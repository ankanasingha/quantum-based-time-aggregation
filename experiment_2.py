import os
import numpy as np
import pandas as pd
import pypsa
import scipy.io
import seaborn as sns
import matplotlib.pyplot as plt
import zipfile
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
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error
from IPython.display import display
import plotly.graph_objects as go
from matplotlib.backends.backend_pdf import PdfPages
from pandas.plotting import parallel_coordinates
import seaborn as sns
from sklearn.manifold import TSNE

os.environ['OMP_NUM_THREADS'] = '1'

def load_ieee_network(case_filename):
    data = scipy.io.loadmat(case_filename)
    mpc = data.get('mpc')  # <== This returns a numpy structured array
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

    # Handling missing values
    data_at.interpolate(method='time', inplace=True)
    data_at.bfill(inplace=True)
    data_at.ffill(inplace=True)

    data_at.rename(columns={
        'AT_load_actual_entsoe_transparency': 'Load (MW)',
        'AT_wind_onshore_generation_actual': 'Wind (MW)'
    }, inplace=True)

    # --- Before Outlier Removal Boxplot ---
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=data_at[['Load (MW)', 'Wind (MW)']])
    plt.title("Before Outlier Removal")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Apply IQR outlier removal for Load (MW)
    filtered = remove_outliers_iqr(data_at, 'Load (MW)', multiplier=1.27)
    data_at = filtered.copy()

    # --- After Outlier Removal Boxplot ---
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
    end_date   = pd.to_datetime("2019-11-30")

    load_profiles_df = load_profiles_df.loc[start_date:end_date]
    wind_profiles_df = wind_profiles_df.loc[start_date:end_date]

    hydro_series = simulate_hydro_profile(load_profiles_df.index)

    combined_df = pd.DataFrame({
        'Load (MW)':  load_profiles_df['Load (MW)'].values,
        'Wind (MW)':  wind_profiles_df['Wind (MW)'].values,
        'Hydro (MW)': hydro_series
    }, index=load_profiles_df.index)

    daily_df = combined_df.resample('D').mean()

    # --- here’s the only part that changes ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(daily_df.index, daily_df['Load (MW)']  / 1e3, label='Load (10³ MW)')
    ax.plot(daily_df.index, daily_df['Wind (MW)']  / 1e3, label='Wind (10³ MW)')
    ax.plot(daily_df.index, daily_df['Hydro (MW)'] / 1e3, label='Hydro (10³ MW)')

    # force the axis to start/end exactly at your data
    ax.set_xlim(daily_df.index.min(), daily_df.index.max())

    # optional: nice month‐ticks on the 1st of each month
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

    ax.set_xlabel("Date")
    ax.set_ylabel("Power (×10³ MW)")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("Yearly_Load_Wind_Hydro_Profiles_2019.pdf", format="pdf")
    plt.show()


# # Clustering & QUBO Functions

# Construct a QUBO for representative day selection.Solve the QUBO using QAOA via Qiskit.
def get_rep_days_qubo(daily_features, k=3, penalty=1e4, reps=2):
    from sklearn.metrics import pairwise_distances
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
            r = line[2] if len(line) > 2 and line[2] > 0 else 0.001  # Set minimal resistance
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

    # Add generators
    for bus in buses:
        bus_name = f"Bus_{int(bus)}"
        if int(bus) % 2 == 1:
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



def plot_season_profiles(case_name, season, load_df, wind_df, hydro_series, rep_days, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    load_k  = load_df['Load (MW)']  / 1e3
    wind_k  = wind_df['Wind (MW)']  / 1e3
    hydro_k = hydro_series           / 1e3

    ax.plot(load_df.index,      load_k,  label='Load (10³ MW)',  linewidth=1.2)
    ax.plot(wind_df.index,      wind_k,  label='Wind (10³ MW)',  linewidth=1.2)
    ax.plot(hydro_series.index, hydro_k, label='Hydro (10³ MW)', linewidth=1.2)

    rep_dates = [load_df.index[d * 24] for d in rep_days]
    for date in rep_dates:
        ax.axvline(date, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel("Date")
    ax.set_ylabel("Power (×10³ MW)")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(f"qubo_allcases_{season}_k_3.pdf", format="pdf")
    plt.show()



def plot_tsne(case_name, season, full_daily_features, rep_days,
              perplexity=30, learning_rate=200, init='pca', n_iter=1000,
              random_state=42, save_path=None):
    """
    Perform a 2D t-SNE embedding of daily features and plot.

    Saves both PNG and PDF versions.
    """
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=learning_rate,
        init=init,
        n_iter=n_iter,
        random_state=random_state
    )
    embeddings = tsne.fit_transform(full_daily_features)

    plt.figure(figsize=(8, 6))
    plt.scatter(embeddings[:, 0], embeddings[:, 1],
                c='lightgray', label='All days', s=20, alpha=0.6)
    if rep_days:
        reps = np.array(rep_days)
        plt.scatter(embeddings[reps, 0], embeddings[reps, 1],
                    c='red', label='QUBO rep days', s=50)

    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{season}_tsne_k_3.pdf", format="pdf")
    plt.show()


def main():
    plt.rcParams.update({
        'lines.linewidth': 2.5,
        'lines.markersize': 8,
        'lines.markeredgewidth': 1.5,
        'patch.linewidth': 1.5,
        'font.weight': 'bold',
        'axes.labelweight': 'bold',
        'axes.titlesize': 0,
        'xtick.labelsize': 10,
        'ytick.labelsize': 14,
        'xtick.major.width': 1.5,
        'ytick.major.width': 1.5,
        'xtick.major.size': 6,
        'ytick.major.size': 6,
        'axes.labelsize': 20,
        'axes.labelweight': 'bold',
        'legend.fontsize': 16,
        'legend.title_fontsize': 18,
    })
    # 1. Load and preprocess profiles
    load_profiles_df, wind_profiles_df = load_profiles()
    load_profiles_df.index = pd.to_datetime(load_profiles_df.index).tz_localize(None)
    wind_profiles_df.index = pd.to_datetime(wind_profiles_df.index).tz_localize(None)

    start_date = pd.to_datetime("2018-12-01")
    end_date = pd.to_datetime("2019-11-30")
    load_profiles_df = load_profiles_df.loc[start_date:end_date]
    wind_profiles_df = wind_profiles_df.loc[start_date:end_date]

    plot_yearly_profiles(load_profiles_df, wind_profiles_df)

    # Season assignment
    def assign_season(month):
        if month in [12, 1, 2]:   return 'Winter'
        if month in [3, 4, 5]:    return 'Spring'
        if month in [6, 7, 8]:    return 'Summer'
        return 'Autumn'

    for df in (load_profiles_df, wind_profiles_df):
        df['Month'] = df.index.month
        df['Season'] = df['Month'].map(assign_season)

    seasons = ['Winter', 'Spring', 'Summer', 'Autumn']
    seasonal_loads = {s: load_profiles_df[load_profiles_df['Season'] == s].copy() for s in seasons}
    seasonal_wind = {s: wind_profiles_df[wind_profiles_df['Season'] == s].copy() for s in seasons}
    seasonal_hydro = {}
    for s in seasons:
        dates = seasonal_loads[s].index
        seasonal_hydro[s] = pd.Series(simulate_hydro_profile(dates), index=dates)
    rep_days_by_season = {season: [] for season in seasons}
    features_by_season = {season: None for season in seasons}
    ieee_cases = {
        'ieee9bus': 'data/case9.mat',
        'ieee30bus': 'data/case30.mat',
        'ieee118bus': 'data/case118.mat'
    }

    summary_data = []

    # Main simulation loop
    for case_name, case_filename in ieee_cases.items():
        print(f"\nStarting {case_name} simulations...")
        network_data = load_ieee_network(case_filename)

        for season in seasons:
            print(f"\nSimulating {season}...")
            load_season = seasonal_loads[season]
            wind_season = seasonal_wind[season]
            hydro_season = seasonal_hydro[season]
            num_days = len(load_season) // 24
            if num_days < 1:
                print(f"Not enough data for {season} in {case_name}. Skipping...")
                continue

            # Full simulation
            rep_days_full = list(range(num_days))
            hydro_series_full = pd.Series(hydro_season.values, index=load_season.index).astype(float)
            net_full = build_pypsa_network(
                network_data, load_season, wind_season, rep_days_full,
                hydro_series=hydro_series_full
            )
            net_full = run_lopf(net_full)
            cost_full = net_full.objective

            # Prepare features for QUBO
            full_daily_features = []
            for d in range(num_days):
                day_load = load_season.iloc[d * 24:(d + 1) * 24]['Load (MW)'].mean()
                day_wind = wind_season.iloc[d * 24:(d + 1) * 24]['Wind (MW)'].mean()
                day_hydro = np.mean(hydro_season.iloc[d * 24:(d + 1) * 24])
                full_daily_features.append([day_load, day_wind, day_hydro])
            full_daily_features = np.array(full_daily_features)

            # QUBO-based representative day selection
            rep_days_qubo = []
            for month in sorted(load_season['Month'].unique()):
                load_month = load_season[load_season['Month'] == month]
                wind_month = wind_season[wind_season['Month'] == month]
                hydro_month = hydro_series_full[load_season['Month'] == month]
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
                    selected = get_rep_days_qubo(daily_features, k=3, penalty=1e4, reps=1)
                    offset = (load_season['Month'] < month).sum() // 24
                    rep_days_qubo.extend([offset + start + idx for idx in selected])

            weights_qubo = compute_day_weights(full_daily_features, rep_days_qubo)
            print("Representative Day Weights:")
            for idx, w in weights_qubo.items():
                date = load_season.index[idx * 24].strftime('%Y-%m-%d')
                print(f"  {date}: {w:.4f}")

            net_qubo = build_pypsa_network(
                network_data, load_season, wind_season, rep_days_qubo,
                hydro_series=hydro_series_full,
                day_weights=weights_qubo
            )
            net_qubo = run_lopf(net_qubo)
            cost_qubo = net_qubo.objective
            deviation = abs(cost_qubo - cost_full) / cost_full * 100 if cost_full else np.nan

            features_by_season[season] = full_daily_features

            rep_days_by_season[season] = rep_days_qubo
            summary_data.append({
                "Case": case_name,
                "Season": season,
                "Full Cost": cost_full,
                "QUBO Cost": cost_qubo,
                "QUBO Deviation (%)": deviation
            })

            print(f"[{case_name} - {season}] Full: {cost_full:.2f}, QUBO: {cost_qubo:.2f}, Dev: {deviation:.2f}%")

    # Compile and save summary
    results_df = pd.DataFrame(summary_data)
    results_df.to_excel("annual_cost_deviation_summary_k_3.xlsx", index=False)
    print(results_df.to_markdown(index=False))

    for season in seasons:
        plot_season_profiles('all_cases', season,
                             seasonal_loads[season], seasonal_wind[season], seasonal_hydro[season],
                             rep_days_by_season[season])
        plot_tsne('all_cases', season,
                  features_by_season[season], rep_days_by_season[season])

    # 8. Static summary plots
    sns.set(style="whitegrid")
    # Bar: deviation by case & season
    plt.figure(figsize=(10, 6))
    sns.barplot(data=results_df, x='Case', y='QUBO Deviation (%)', hue='Season')
    plt.tight_layout()
    plt.savefig("qubo_deviation_by_case_season_k_3_even_odd.pdf")
    plt.show()
    # Grouped bar
    plt.figure(figsize=(12, 6))
    melt = results_df.melt(['Case', 'Season'], ['Full Cost', 'QUBO Cost'],
                           var_name='Cost Type', value_name='Cost')
    sns.barplot(data=melt, x='Case', y='Cost', hue='Cost Type', ci=None)
    plt.tight_layout()
    plt.savefig("full_vs_qubo_cost_by_case_k_3_even_odd.pdf")
    plt.show()

    plt.figure(figsize=(10, 6))

    for c in results_df['Case'].unique():
        sub = results_df[results_df['Case'] == c]
        plt.plot(
            sub['Season'],
            sub['QUBO Deviation (%)'],
            marker='o',
            lw=3,  # thicker line
            markersize=10,  # bigger markers
            label=c
        )

    plt.xlabel('Season', fontsize=22, fontweight='bold')
    plt.ylabel('QUBO Deviation (%)', fontsize=22, fontweight='bold')

    plt.xticks(fontsize=16, fontweight='bold')
    plt.yticks(fontsize=16, fontweight='bold')

    leg = plt.legend(
        title='System Case',
        fontsize=18,
        title_fontsize=20,
        frameon=True
    )

    for legline in leg.get_lines():
        legline.set_linewidth(3.0)

    plt.tight_layout()
    plt.savefig("qubo_deviation_lineplot_k_3_even_odd.pdf", format='pdf', bbox_inches='tight')
    plt.show()

    # Avg deviation
    plt.figure(figsize=(8, 5))
    avg = results_df.groupby('Case')['QUBO Deviation (%)'].mean().reset_index()
    sns.barplot(data=avg, x='Case', y='QUBO Deviation (%)')
    plt.tight_layout()
    plt.savefig("average_annual_deviation_by_case_k_3_even_odd.pdf")
    plt.show()

    dev_mat = results_df.pivot(index='Case', columns='Season', values='QUBO Deviation (%)')
    dev_mat = dev_mat[seasons]

    plt.figure(figsize=(6, 4))
    sns.heatmap(dev_mat, annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={'label': 'Deviation (%)'})
    plt.ylabel("Case")
    plt.xlabel("Season")
    plt.tight_layout()
    plt.savefig("deviation_heatmap_k_3_even_odd.pdf", format="pdf")
    plt.show()

    print("\nSimulation and analysis completed successfully.")


if __name__ == '__main__':
    main()