import os
import numpy as np
import pandas as pd
import pypsa
import scipy.io
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore
from scipy.io import loadmat
from sklearn.metrics import pairwise_distances
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import Sampler
from sklearn.metrics import mean_squared_error
from IPython.display import display
import plotly.express as px
import zipfile
import plotly.graph_objects as go
import random
from sklearn.metrics import pairwise_distances
np.random.seed(30)

def load_ieee_network(case_filename):
    data = scipy.io.loadmat(case_filename)
    mpc = data.get('mpc')
    if mpc is None:
        raise ValueError("File does not contain 'mpc' key")

    mpc = mpc[0, 0]  # Extract the struct fields
    network_data = {
        'bus': mpc['bus'],
        'branch': mpc['branch'],
        'gen': mpc['gen']
    }
    return network_data


def load_csv_data(csv_file_path):
    with zipfile.ZipFile(csv_file_path, 'r') as z:
        csv_filename = next(f for f in z.namelist() if f.endswith('.csv'))
        with z.open(csv_filename) as f:
            df = pd.read_csv(
                f,
                parse_dates=['utc_timestamp'],
                index_col='utc_timestamp'
            )
    return df


# In[7]:


def remove_outliers_iqr(df, column, multiplier=1.27):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    print(f"IQR bounds for {column}: {lower_bound:.2f} to {upper_bound:.2f}")
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def simulate_hydro_profile(index, capacity=8000, base_cf=0.85, amp=0.15, offset=80):
    day_of_year = index.dayofyear
    hydro_cf = base_cf - amp * np.sin(2 * np.pi * (day_of_year - offset) / 365)
    return hydro_cf * capacity


# Load load and wind profiles from OPSD CSV file, select columns of interest, clean, and rename columns. Remove the outliers from the load data keeping the outliers for wind data to capture extreme situations.
def load_profiles():
    load_csv = 'data/opsd-time_series-2020-10-06/opsd-time_series-2020-10-06/time_series_60min_singleindex.zip'
    load_data = load_csv_data(load_csv)

    columns_of_interest = [
        'AT_load_actual_entsoe_transparency',
        'AT_wind_onshore_generation_actual'
    ]

    data_at = load_data[columns_of_interest].copy()

    data_at.interpolate(method='time', inplace=True)
    data_at.bfill(inplace=True)
    data_at.ffill(inplace=True)

    data_at.rename(columns={
        'AT_load_actual_entsoe_transparency': 'Load (MW)',
        'AT_wind_onshore_generation_actual': 'Wind (MW)'
    }, inplace=True)

    plt.figure(figsize=(6, 4))
    sns.boxplot(data=data_at[['Load (MW)', 'Wind (MW)']])
    plt.title("Before Outlier Removal")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    filtered = remove_outliers_iqr(data_at, 'Load (MW)', multiplier=1.27)
    data_at = filtered.copy()

    plt.figure(figsize=(6, 4))
    sns.boxplot(data=data_at[['Load (MW)', 'Wind (MW)']])
    plt.title("After Outlier Removal (IQR method)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    load_profiles_df = data_at[['Load (MW)']].copy()
    wind_profiles_df = data_at[['Wind (MW)']].copy()

    return load_profiles_df, wind_profiles_df

def plot_yearly_profiles(load_profiles_df, wind_profiles_df):
    start_date = pd.to_datetime("2018-12-01")
    end_date = pd.to_datetime("2019-11-30")

    load_profiles_df = load_profiles_df[(load_profiles_df.index >= start_date) & (load_profiles_df.index <= end_date)]
    wind_profiles_df = wind_profiles_df[(wind_profiles_df.index >= start_date) & (wind_profiles_df.index <= end_date)]

    hydro_series = simulate_hydro_profile(load_profiles_df.index)

    # Create DataFrame for plotting
    combined_df = pd.DataFrame({
        'Load (MW)': load_profiles_df['Load (MW)'].values,
        'Wind (MW)': wind_profiles_df['Wind (MW)'].values,
        'Hydro (MW)': hydro_series
    }, index=load_profiles_df.index)

    daily_df = combined_df.resample('D').mean()

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(daily_df.index, daily_df['Load (MW)'], label='Load (MW)', linewidth=1.2)
    plt.plot(daily_df.index, daily_df['Wind (MW)'], label='Wind (MW)', linewidth=1.2)
    plt.plot(daily_df.index, daily_df['Hydro (MW)'], label='Hydro (MW)', linewidth=1.2)
    plt.title("Daily Average Load, Wind, and Hydro Profiles - Year 2019")
    plt.xlabel("Date")
    plt.ylabel("Power (MW)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("Yearly_Load_Wind_Hydro_Profiles_2019.png")
    plt.show()
    print("Saved plot as 'Yearly_Load_Wind_Hydro_Profiles_2019.png'")


# # Clustering & QUBO Functions
# Construct a QUBO for representative day selection.Solve the QUBO using QAOA via Qiskit.
def get_rep_days_qubo(daily_features, k, penalty=1e4, reps=2):
    dist_matrix = pairwise_distances(daily_features, metric='euclidean')
    qp = QuadraticProgram()
    N = len(daily_features)
    for i in range(N):
        qp.binary_var(name=f"x_{i}")
    linear = {i: penalty * (1 - 2 * k) for i in range(N)}
    quadratic = {}
    for i in range(N):
        for j in range(i + 1, N):
            quadratic[(i, j)] = dist_matrix[i, j] + 2 * penalty
    qp.minimize(linear=linear, quadratic=quadratic)
    optimizer = COBYLA(maxiter=1000)
    sampler = Sampler()
    qaoa = QAOA(optimizer=optimizer, reps=reps, sampler=sampler)
    meo = MinimumEigenOptimizer(qaoa)
    result = meo.solve(qp)
    rep_days = [i for i, bit in enumerate(result.x) if bit > 0.5]
    return rep_days


# Compute non-uniform weights: count how many full days are assigned to each rep day.
def compute_day_weights(full_features, rep_days):
    rep_day_weights = {r: 0 for r in rep_days}
    for i in range(len(full_features)):
        distances = [np.linalg.norm(full_features[i] - full_features[r]) for r in rep_days]
        closest_rep = rep_days[np.argmin(distances)]
        rep_day_weights[closest_rep] += 1
    return rep_day_weights


#
# # PyPSA Network Functions
#
#

# Construct a PyPSA network from IEEE network data, load profiles, and wind profiles, hydro profile.
#     'rep_days' is a list of representative day indices.

def build_pypsa_network(network_data, load_profiles, wind_profiles, rep_days, hydro_series, day_weights=None):
    n = pypsa.Network()
    n.add("Carrier",
          name=["AC", "wind", "hydro"],
          co2_emissions=[0.0, 0.0, 0.0],
          max_growth=[np.inf, np.inf, np.inf],
          max_relative_growth=[np.inf, np.inf, np.inf])

    buses = network_data.get('bus')
    if buses is None:
        num_buses = network_data.get('num_buses', 3)
        buses = np.arange(1, num_buses + 1)
    else:
        buses = buses[:, 0]

    for bus in buses:
        n.add("Bus", f"Bus_{int(bus)}", carrier="AC")

    lines = network_data.get('branch')
    if lines is not None:
        for line in lines:
            bus0 = f"Bus_{int(line[0])}"
            bus1 = f"Bus_{int(line[1])}"
            x = line[3] if len(line) > 3 and line[3] > 0 else 0.1
            r = line[2] if len(line) > 2 and line[2] > 0 else 0.001
            n.add("Line", f"Line_{int(line[0])}_{int(line[1])}",
                  bus0=bus0, bus1=bus1, x=x, r=r, s_nom=500, carrier="AC")

    # Create snapshots and assign weights from day_weights
    rep_snapshots = []
    weights = {}
    if day_weights is None:
        day_weights = {d: 1.0 for d in rep_days}
    for d in rep_days:
        day_snaps = load_profiles.index[d * 24:(d + 1) * 24]
        rep_snapshots.extend(day_snaps)
        for snap in day_snaps:
            weights[snap] = float(day_weights[d])
    n.set_snapshots(rep_snapshots)
    n.snapshot_weightings = pd.Series(weights)

    # Add loads
    for bus in buses:
        bus_name = f"Bus_{int(bus)}"
        n.add("Load", f"Load_{bus_name}", bus=bus_name,
              p_set=load_profiles.iloc[:len(rep_snapshots), 0].values)

    generator_types = ["hydro", "wind"]
    for bus in buses:
        bus_name = f"Bus_{int(bus)}"
        # Randomly select a generator type for each bus
        gen_type = np.random.choice(generator_types)
        if gen_type == "hydro":
            n.add("Generator", f"Hydro_{bus_name}", bus=bus_name,
                  p_nom=8000, marginal_cost=10,
                  p_max_pu=hydro_series.loc[rep_snapshots].values if hydro_series is not None else 1.0,
                  carrier="hydro")
        else:
            n.add("Generator", f"Wind_{bus_name}", bus=bus_name,
                  p_nom_extendable=True, marginal_cost=7,
                  p_max_pu=wind_profiles.iloc[:len(rep_snapshots), 0].values,
                  carrier="wind")

    return n


# Solve the network's LOPF problem using the HiGHS solver.
def run_lopf(network):
    network.optimize(solver_name='highs')
    return network


def plot_season_profiles(season, load_df, wind_df, hydro_series, rep_days, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(load_df.index, load_df['Load (MW)'], label='Load', color='blue')
    ax.plot(wind_df.index, wind_df['Wind (MW)'], label='Wind', color='green')
    ax.plot(hydro_series.index, hydro_series.values, label='Hydro', color='cyan')

    rep_dates = [load_df.index[d * 24] for d in rep_days]
    for date in rep_dates:
        ax.axvline(date, color='red', linestyle='--', alpha=0.5)

    ax.set_title(f"Seasonal Profile with QUBO Representative Days: {season}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Power (MW)")
    ax.legend()
    ax.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()


def main():
    load_profiles_df, wind_profiles_df = load_profiles()

    load_profiles_df.index = pd.to_datetime(load_profiles_df.index).tz_localize(None)
    wind_profiles_df.index = pd.to_datetime(wind_profiles_df.index).tz_localize(None)

    start_date = pd.to_datetime("2018-12-01")
    end_date = pd.to_datetime("2019-11-30")

    load_profiles_df = load_profiles_df[(load_profiles_df.index >= start_date) & (load_profiles_df.index <= end_date)]
    wind_profiles_df = wind_profiles_df[(wind_profiles_df.index >= start_date) & (wind_profiles_df.index <= end_date)]

    plot_yearly_profiles(load_profiles_df, wind_profiles_df)

    def assign_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Autumn'

    load_profiles_df['Month'] = load_profiles_df.index.month
    wind_profiles_df['Month'] = wind_profiles_df.index.month
    load_profiles_df['season'] = load_profiles_df['Month'].map(assign_season)
    wind_profiles_df['season'] = wind_profiles_df['Month'].map(assign_season)

    seasonal_loads, seasonal_wind, seasonal_hydro = {}, {}, {}
    for season in ['Winter', 'Spring', 'Summer', 'Autumn']:
        seasonal_loads[season] = load_profiles_df[load_profiles_df['season'] == season].copy()
        seasonal_wind[season] = wind_profiles_df[wind_profiles_df['season'] == season].copy()
        seasonal_hydro[season] = pd.Series(simulate_hydro_profile(seasonal_loads[season].index),
                                           index=seasonal_loads[season].index)

    ieee_cases = {
        'ieee9bus': 'data/case9.mat',
        'ieee30bus': 'data/case30.mat',
        'ieee118bus': 'data/case118.mat'
    }

    summary_data = []

    for case_name, case_filename in ieee_cases.items():
        print(f"\nStarting {case_name} simulations...")
        network_data = load_ieee_network(case_filename)

        for season in ['Winter', 'Spring', 'Summer', 'Autumn']:
            print(f"\nSimulating {season}...")
            load_season = seasonal_loads[season]
            wind_season = seasonal_wind[season]
            hydro_season = seasonal_hydro[season]
            num_days = len(load_season) // 24
            if num_days < 1:
                print(f"Not enough data for {season} in {case_name}. Skipping...")
                continue

            # Full Simulation
            rep_days_full = list(range(num_days))
            hydro_series_full = pd.Series(hydro_season.values, index=load_season.index).astype(float)
            net_full = build_pypsa_network(network_data, load_season, wind_season, rep_days_full,
                                           hydro_series=hydro_series_full)
            net_full = run_lopf(net_full)
            cost_full = net_full.objective
            full_rep_dates = [load_season.index[d * 24].strftime('%Y-%m-%d') for d in rep_days_full]
            print(f"[{case_name} - {season}] Full Simulation Cost: {cost_full}")
            print(f"[{case_name} - {season}] Representative Days (Full): {full_rep_dates}")

            full_daily_features = []
            for d in range(num_days):
                day_load = load_season.iloc[d * 24:(d + 1) * 24]['Load (MW)'].mean()
                day_wind = wind_season.iloc[d * 24:(d + 1) * 24]['Wind (MW)'].mean()
                day_hydro = np.mean(hydro_season.iloc[d * 24:(d + 1) * 24])
                full_daily_features.append([day_load, day_wind, day_hydro])
            full_daily_features = np.array(full_daily_features)

            # QUBO-based Simulation (Non-Uniform Weights)
            rep_days_qubo = []
            for month in sorted(load_season['Month'].unique()):
                print(f"\nProcessing Month: {month} in Season: {season}")
                load_month = load_season[load_season['Month'] == month].copy()
                wind_month = wind_season[wind_season['Month'] == month].copy()
                hydro_series_season = pd.Series(hydro_season.values, index=load_season.index).astype(float)
                hydro_month = hydro_series_season[load_season['Month'] == month]
                num_days_month = len(load_month) // 24
                split_points = [0, num_days_month // 2, num_days_month]
                for i in range(2):
                    start, end = split_points[i], split_points[i + 1]
                    if end - start < 1:
                        continue
                    daily_features = []
                    for d in range(start, end):
                        day_load = load_month.iloc[d * 24:(d + 1) * 24]['Load (MW)'].mean()
                        day_wind = wind_month.iloc[d * 24:(d + 1) * 24]['Wind (MW)'].mean()
                        day_hydro = np.mean(hydro_month[d * 24:(d + 1) * 24])
                        daily_features.append([day_load, day_wind, day_hydro])
                    daily_features = np.array(daily_features)
                    # Select 3 representative days per half-month via QUBO
                    selected_days = get_rep_days_qubo(daily_features, k=3, penalty=1e4, reps=1)
                    month_offset = (load_season['Month'] < month).sum() // 24
                    rep_days_qubo.extend([month_offset + start + idx for idx in selected_days])

            rep_dates_qubo = [load_season.index[d * 24].strftime('%Y-%m-%d') for d in rep_days_qubo]
            # Compute non-uniform weights based on full season features
            weights_qubo = compute_day_weights(full_daily_features, rep_days_qubo)
            print("Representative Day Weights:")
            for day_index, weight in weights_qubo.items():
                rep_date = load_season.index[day_index * 24].strftime('%Y-%m-%d')
                print(f"{rep_date}: weight = {weight}")
            net_qubo = build_pypsa_network(network_data, load_season, wind_season, rep_days_qubo,
                                           hydro_series=hydro_series_full, day_weights=weights_qubo)
            net_qubo = run_lopf(net_qubo)
            cost_qubo = net_qubo.objective

            plot_season_profiles(season, load_season, wind_season, hydro_series_full, rep_days_qubo)

            deviation_qubo = abs(cost_qubo - cost_full) / cost_full * 100 if cost_full != 0 else np.nan
            print(f"[{case_name} - {season}] QUBO Simulation Cost: {cost_qubo} (Deviation: {deviation_qubo:.2f}%)")
            print(f"[{case_name} - {season}] Representative Days (QUBO): {rep_dates_qubo}")

            # Annualized costs
            annualized_cost_full = cost_full * (365 / num_days)
            annualized_cost_qubo = cost_qubo * (365 / len(rep_days_qubo)) if rep_days_qubo else np.nan

            summary_data.append({
                "Case": case_name,
                "Season": season,
                "Full Cost": cost_full,
                "QUBO Cost": cost_qubo,
                "QUBO Deviation (%)": deviation_qubo
            })

            print(f"\n[{case_name} - {season}] Summary:")
            print(f" Full Cost: {cost_full:.2f}, Annualized: {annualized_cost_full:.2f}")
            print(
                f" QUBO Cost: {cost_qubo:.2f}, Annualized: {annualized_cost_qubo:.2f}, Deviation: {deviation_qubo:.2f}%")

    results_df = pd.DataFrame(summary_data)
    print("\nSummary of All Simulation Results:")
    print(results_df)

    results_df.to_excel("annual_cost_deviation_summary_set_2.xlsx", index=False)
    print("\nResults saved to 'annual_cost_deviation_summary_set_2.xlsx'")

    # ---------------- Plotting Section ----------------
    sns.set(style="whitegrid")

    # 1. Bar Plot: QUBO Deviation by Case and Season
    plt.figure(figsize=(10, 6))
    sns.barplot(data=results_df, x='Case', y='QUBO Deviation (%)', hue='Season')
    plt.title('QUBO Cost Deviation by Case and Season')
    plt.ylabel('Deviation (%)')
    plt.tight_layout()
    plt.savefig("qubo_deviation_by_case_season.png")
    plt.show()

    # 2. Grouped Bar: Full vs QUBO Cost
    plt.figure(figsize=(12, 6))
    melted = results_df.melt(id_vars=['Case', 'Season'], value_vars=['Full Cost', 'QUBO Cost'],
                             var_name='Cost Type', value_name='Cost')
    sns.barplot(data=melted, x='Case', y='Cost', hue='Cost Type', ci=None)
    plt.title('Full vs QUBO Cost by Case (Aggregated Across Seasons)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("full_vs_qubo_cost_by_case.png")
    plt.show()

    # 3. Line Plot of Deviation over Seasons by Case
    plt.figure(figsize=(10, 6))
    for case in results_df['Case'].unique():
        subset = results_df[results_df['Case'] == case]
        plt.plot(subset['Season'], subset['QUBO Deviation (%)'], marker='o', label=case)
    plt.title("QUBO Deviation (%) Across Seasons by Case")
    plt.xlabel("Season")
    plt.ylabel("Deviation (%)")
    plt.legend(title="Case")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("qubo_deviation_lineplot.png")
    plt.show()

    # 4. Average Annual Deviation by Case
    avg_dev = results_df.groupby("Case")["QUBO Deviation (%)"].mean().reset_index()
    plt.figure(figsize=(8, 5))
    sns.barplot(data=avg_dev, x="Case", y="QUBO Deviation (%)", palette="coolwarm")
    plt.title("Average Annual QUBO Deviation (%) by Case")
    plt.tight_layout()
    plt.savefig("average_annual_deviation_by_case.png")
    plt.show()

    # 5. Stacked Area Chart of QUBO Costs by Season and Case
    season_order = ['Winter', 'Spring', 'Summer', 'Autumn']
    pivot_df = results_df.pivot(index='Season', columns='Case', values='QUBO Cost')
    pivot_df = pivot_df.reindex(season_order)
    plt.figure(figsize=(10, 6))
    pivot_df.plot(kind='area', stacked=True, colormap='Set3', alpha=0.9)
    plt.title("Stacked Area Chart of QUBO Costs by Season and Case")
    plt.xlabel("Season")
    plt.ylabel("QUBO Cost")
    plt.tight_layout()
    plt.grid(True)
    plt.savefig("stacked_qubo_costs_by_season.png")
    plt.show()

    print("\nSimulation and analysis completed successfully.")


if __name__ == '__main__':
    main()