import os
import pypsa
import scipy.io
import seaborn as sns
from scipy.io import loadmat
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from qiskit_optimization import QuadraticProgram
from sklearn.manifold import TSNE
import itertools
import time
import json
from collections import defaultdict
from qiskit.primitives import Sampler
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import scipy.io
import pypsa
from sklearn.metrics import pairwise_distances
from sklearn.manifold import TSNE
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit.algorithms.minimum_eigensolvers import QAOA, NumPyMinimumEigensolver
from qiskit.algorithms.optimizers import COBYLA
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from pandas.plotting import parallel_coordinates

os.environ['OMP_NUM_THREADS'] = '1'

def load_ieee_network(case_filename):
    data = scipy.io.loadmat(case_filename)
    mpc = data.get('mpc')
    if mpc is None:
        raise ValueError("File does not contain 'mpc' key")

    mpc = mpc[0, 0]
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

    load_profiles_df = load_profiles_df[(load_profiles_df.index >= start_date) & (load_profiles_df.index <= end_date)]
    wind_profiles_df = wind_profiles_df[(wind_profiles_df.index >= start_date) & (wind_profiles_df.index <= end_date)]

    hydro_series = simulate_hydro_profile(load_profiles_df.index)

    combined_df = pd.DataFrame({
        'Load (MW)': load_profiles_df['Load (MW)'].values,
        'Wind (MW)': wind_profiles_df['Wind (MW)'].values,
        'Hydro (MW)': hydro_series
    }, index=load_profiles_df.index)

    daily_df = combined_df.resample('D').mean()

    plt.figure(figsize=(10, 5))
    plt.plot(daily_df.index, daily_df['Load (MW)'] / 1e3, label='Load (10³ MW)')
    plt.plot(daily_df.index, daily_df['Wind (MW)'] / 1e3, label='Wind (10³ MW)')
    plt.plot(daily_df.index, daily_df['Hydro (MW)'] / 1e3, label='Hydro (10³ MW)')
    plt.xlabel("Date")
    plt.ylabel("Power (×10³ MW)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("Yearly_Load_Wind_Hydro_Profiles_2019.pdf", format="pdf")
    plt.show()

def build_daily_72d_matrix(load_season, wind_season, hydro_series):
    """Return (N_days, 72) array concatenating 24h load, 24h wind, 24h hydro per day."""
    num_days = len(load_season) // 24
    X = np.zeros((num_days, 72), dtype=float)
    for d in range(num_days):
        sl = slice(d * 24, (d + 1) * 24)
        L = load_season.iloc[sl]['Load (MW)'].values.astype(float)
        W = wind_season.iloc[sl]['Wind (MW)'].values.astype(float)
        H = hydro_series.iloc[sl].values.astype(float)
        X[d, :24] = L
        X[d, 24:48] = W
        X[d, 48:72] = H
    return X

def pca_reduce_to_3(X, scaler=None, pca=None):
    """
    Standardize then reduce 72D daily vectors to 3D via PCA.
    Returns reduced features and fitted scaler/pca (if not provided).
    """
    own_scaler = False
    own_pca = False
    if scaler is None:
        scaler = StandardScaler(with_mean=True, with_std=True)
        own_scaler = True
        Xs = scaler.fit_transform(X)
    else:
        Xs = scaler.transform(X)

    if pca is None:
        pca = PCA(n_components=3, random_state=42)
        own_pca = True
        Z = pca.fit_transform(Xs)
    else:
        Z = pca.transform(Xs)

    return Z, (scaler if own_scaler else None), (pca if own_pca else None)

def get_rep_days_qubo(daily_features, k, penalty=1e4, reps=1, maxiter=500):
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

    optimizer = COBYLA(maxiter=maxiter)
    sampler = Sampler()
    qaoa = QAOA(optimizer=optimizer, reps=reps, sampler=sampler)
    meo = MinimumEigenOptimizer(qaoa)
    result = meo.solve(qp)
    rep_days = [i for i, bit in enumerate(result.x) if bit > 0.5]
    return rep_days

def compute_day_weights(full_features, rep_days):
    rep_day_weights = {r: 0 for r in rep_days}
    for i in range(len(full_features)):
        distances = [np.linalg.norm(full_features[i] - full_features[r]) for r in rep_days]
        closest_rep = rep_days[np.argmin(distances)]
        rep_day_weights[closest_rep] += 1
    return rep_day_weights

from collections import defaultdict


def build_pypsa_network(network_data, load_profiles, wind_profiles, rep_days, hydro_series, day_weights=None):
    n = pypsa.Network()
    for cname in ["AC", "wind", "hydro"]:
        n.add("Carrier", name=cname)
    if "co2_emissions" in n.carriers.columns:
        n.carriers.loc[["AC", "wind", "hydro"], "co2_emissions"] = 0.0

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
        pair_counts = defaultdict(int)  # counts per (i,j) (directional) pair
        for idx, line in enumerate(lines, start=1):
            i = int(line[0]);
            j = int(line[1])
            bus0 = f"Bus_{i}"
            bus1 = f"Bus_{j}"

            # MATPOWER columns: r=x[2], x=x[3]; fallbacks stay the same
            x = line[3] if len(line) > 3 and line[3] > 0 else 0.1
            r = line[2] if len(line) > 2 and line[2] > 0 else 0.001

            pair_counts[(i, j)] += 1
            name = f"Line_{i}_{j}_{pair_counts[(i, j)]}"  # ensure uniqueness

            n.add("Line", name, bus0=bus0, bus1=bus1, x=x, r=r, s_nom=500, carrier="AC")
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

    for bus in buses:
        bus_name = f"Bus_{int(bus)}"
        n.add("Load", f"Load_{bus_name}", bus=bus_name,
              p_set=load_profiles.iloc[:len(rep_snapshots), 0].values)

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

def run_lopf(network):
    network.optimize(solver_name='highs')
    return network

def plot_pca_explained_variance(pca, season):
    """Bar plot of variance explained by 3 PCA components."""
    ev = pca.explained_variance_ratio_
    plt.figure(figsize=(6, 4))
    plt.bar([1, 2, 3], ev, tick_label=['PC1', 'PC2', 'PC3'])
    plt.ylim(0, 1)
    plt.ylabel("Explained variance ratio")
    # plt.title(f"PCA explained variance — {season}")
    for i, v in enumerate(ev, start=1):
        plt.text(i, v + 0.02, f"{v * 100:.1f}%", ha='center')
    plt.tight_layout()
    plt.savefig(f"pca_explained_{season}.pdf", dpi=200)
    plt.show()

def plot_pca3_scatter(Z3, rep_days, case_name, season):
    """3D scatter of PCA(3) with representative days highlighted."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(Z3[:, 0], Z3[:, 1], Z3[:, 2], s=18, alpha=0.35, label='All days')
    if len(rep_days):
        ax.scatter(Z3[rep_days, 0], Z3[rep_days, 1], Z3[rep_days, 2],
                   s=60, marker='o', label='Rep days')
    ax.set_xlabel('PC1');
    ax.set_ylabel('PC2');
    ax.set_zlabel('PC3')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"pca3_scatter_{case_name}_{season}.pdf", dpi=220)
    plt.show()

def plot_season_profiles_with_reps(load_season, wind_season, hydro_season, rep_days, case_name, season):
    """Time series for the season with vertical lines at representative-day starts."""
    fig, ax = plt.subplots(figsize=(10, 5))
    # aggregate by hour to MW×1e3
    ax.plot(load_season.index, load_season['Load (MW)'] / 1e3, label='Load (10³ MW)', linewidth=2)
    ax.plot(wind_season.index, wind_season['Wind (MW)'] / 1e3, label='Wind (10³ MW)', linewidth=2)
    ax.plot(hydro_season.index, hydro_season / 1e3, label='Hydro (10³ MW)', linewidth=2)

    for d in rep_days:
        date = load_season.index[d * 24]
        ax.axvline(date, color='red', linestyle='--', alpha=0.6)

    ax.set_xlabel("Date")
    ax.set_ylabel("Power (×10³ MW)")
    ax.legend();
    ax.grid(True);
    plt.tight_layout()
    plt.savefig(f"season_profiles_with_reps_{case_name}_{season}.pdf", format="pdf")
    plt.show()

# t-SNE plot (2D) highlighting representative days ===
def plot_tsne_2d_from_pca3(Z3, rep_days, case_name, season, perplexity=15):
    """
    Project PCA(3) -> t-SNE(2) and highlight representative days.
    Saves as '<Season>_tsne_k3.pdf' to match the attached figure set.
    """
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42,
                init='random', learning_rate='auto')
    Y = tsne.fit_transform(Z3)

    plt.figure(figsize=(7, 6))
    plt.scatter(Y[:, 0], Y[:, 1], s=20, alpha=0.35, label='All days')
    if rep_days:
        plt.scatter(Y[rep_days, 0], Y[rep_days, 1],
                    s=60, marker='o', label='Rep days')
    plt.xlabel('t-SNE 1');
    plt.ylabel('t-SNE 2')
    plt.legend();
    plt.tight_layout()
    plt.savefig(f"{season}_tsne_k3.pdf", format="pdf")
    plt.show()


# summary figure set (5 charts) created from results_df
def plot_summary_figures(results_df):
    """
    Expects columns: Case, Season, Full Cost, PCA3-QUBO Cost, Deviation (%), Rep days
    Produces the five summary charts to match the attached PDF.
    """

    # Ensure nice ordering of seasons on plots
    season_order = ['Winter', 'Spring', 'Summer', 'Autumn']
    results_df = results_df.copy()
    results_df['Season'] = pd.Categorical(results_df['Season'], categories=season_order, ordered=True)

    # 1) Deviation bar chart by Case × Season
    plt.figure(figsize=(9, 5))
    for i, (case, sub) in enumerate(results_df.groupby('Case'), start=1):
        plt.bar([f"{case}\n{s}" for s in sub.sort_values('Season')['Season']],
                sub.sort_values('Season')['Deviation (%)'], label=case)
    plt.ylabel("Deviation (%)")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("qubo_deviation_by_case_season_k_3_even_odd.pdf", format="pdf")
    plt.show()

    # 2) Full vs QUBO cost (grouped bars) per Case (aggregated over seasons)
    agg = results_df.groupby('Case', as_index=False).agg({
        'Full Cost': 'mean',
        'PCA3-QUBO Cost': 'mean'
    })
    x = np.arange(len(agg))
    w = 0.35
    plt.figure(figsize=(8, 5))
    plt.bar(x - w / 2, agg['Full Cost'], width=w, label='Full')
    plt.bar(x + w / 2, agg['PCA3-QUBO Cost'], width=w, label='PCA3-QUBO')
    plt.xticks(x, agg['Case'])
    plt.ylabel("Objective (mean over seasons)")
    plt.legend();
    plt.tight_layout()
    plt.savefig("full_vs_qubo_cost_by_case_k_3_even_odd.pdf", format="pdf")
    plt.show()

    # 3) Deviation line plot across seasons per Case
    plt.figure(figsize=(8, 5))
    for case, sub in results_df.groupby('Case'):
        sub = sub.sort_values('Season')
        plt.plot(sub['Season'], sub['Deviation (%)'], marker='o', label=case)
    plt.ylabel("Deviation (%)")
    plt.tight_layout()
    plt.legend()
    plt.savefig("qubo_deviation_lineplot_k_3_even_odd.pdf", format="pdf")
    plt.show()

    # 4) Average annual deviation by Case
    mean_dev = results_df.groupby('Case', as_index=False)['Deviation (%)'].mean()
    plt.figure(figsize=(7, 5))
    plt.bar(mean_dev['Case'], mean_dev['Deviation (%)'])
    plt.ylabel("Avg Deviation (%)")
    plt.tight_layout()
    plt.savefig("average_annual_deviation_by_case_k_3_even_odd.pdf", format="pdf")
    plt.show()

    # 5) Deviation heatmap (Case × Season)
    # pivot to (Case × Season) matrix
    pivot = results_df.pivot(index='Case', columns='Season', values='Deviation (%)').reindex(columns=season_order)
    plt.figure(figsize=(7, 4.5))
    im = plt.imshow(pivot.values, aspect='auto')
    plt.colorbar(im, fraction=0.046, pad=0.04, label="Deviation (%)")
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.tight_layout()
    plt.savefig("deviation_heatmap_k_3_even_odd.pdf", format="pdf")
    plt.show()

def main():
    plt.rcParams.update({
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'font.size': 16,
        'font.weight': 'bold',
        'axes.labelweight': 'bold',
        'axes.titlesize': 18,
        'axes.titleweight': 'bold',
        'axes.linewidth': 2,
        'grid.linewidth': 1.5,
        'lines.linewidth': 3,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14
    })

    # 1) Load and trim data
    load_profiles_df, wind_profiles_df = load_profiles()
    load_profiles_df.index = pd.to_datetime(load_profiles_df.index).tz_localize(None)
    wind_profiles_df.index = pd.to_datetime(wind_profiles_df.index).tz_localize(None)

    start_date = pd.to_datetime("2018-12-01")
    end_date = pd.to_datetime("2019-11-30")
    load_profiles_df = load_profiles_df.loc[start_date:end_date]
    wind_profiles_df = wind_profiles_df.loc[start_date:end_date]

    # Yearly profiles
    plot_yearly_profiles(load_profiles_df, wind_profiles_df)

    # 2) Season tags
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

    # 3) IEEE cases (enable/extend as needed)
    ieee_cases = {
        'ieee9bus': 'C:/Users/Lenovo/Documents/case9.mat',
        'ieee30bus': 'C:/Users/Lenovo/Documents/case30.mat',
        'ieee118bus': 'C:/Users/Lenovo/Documents/case118.mat'
    }

    summary_data = []

    # 4) Outer loop: cases → seasons
    for case_name, case_filename in ieee_cases.items():
        print(f"\n=== {case_name} ===")
        network_data = load_ieee_network(case_filename)

        for season in seasons:
            print(f"\n-- Season: {season} --")
            if seasonal_loads[season].empty:
                print("No data for this season; skipping.")
                continue

            load_season = seasonal_loads[season].drop(columns=['Month', 'Season'])
            wind_season = seasonal_wind[season].drop(columns=['Month', 'Season'])
            hydro_season = seasonal_hydro[season]

            num_days = len(load_season) // 24
            if num_days < 1:
                print("Not enough data; skipping.")
                continue

            # 4a) Baseline full model
            hydro_series_full = pd.Series(hydro_season.values, index=load_season.index).astype(float)
            rep_days_full = list(range(num_days))
            net_full = build_pypsa_network(
                network_data,
                load_season,
                wind_season,
                rep_days_full,
                hydro_series=hydro_series_full
            )
            net_full = run_lopf(net_full)
            cost_full = net_full.objective

            # 4b) PCA(3) features from 72D daily vectors
            X72 = build_daily_72d_matrix(load_season, wind_season, hydro_season)
            Z3, scaler, pca = pca_reduce_to_3(X72)

            # Explained variance (saved per case×season by helper)
            plot_pca_explained_variance(pca, f"{case_name}_{season}")

            # 4c) QUBO-based representative-day selection (split by half-month windows)
            rep_days_qubo = []

            def month_day_indices(m):
                idx = pd.Index(load_season.index)
                mask = (idx.month == m) & (idx.hour == 0)
                day_idx = np.unique(np.where(mask)[0] // 24)
                num_days = len(load_season) // 24
                return day_idx[day_idx < num_days]

            unique_months = sorted(pd.unique(pd.Index(load_season.index).month))
            for m in unique_months:
                idx_days = month_day_indices(m)
                if len(idx_days) < 1:
                    continue
                split = len(idx_days) // 2
                halves = [idx_days[:split], idx_days[split:]]
                for half in halves:
                    if len(half) < 1:
                        continue
                    feats_half = Z3[half, :]
                    selected_local = get_rep_days_qubo(
                        feats_half, k=3, penalty=1e4, reps=1, maxiter=500
                    )
                    rep_days_qubo.extend([half[i] for i in selected_local])

            rep_days_qubo = sorted(set(rep_days_qubo))
            if len(rep_days_qubo) == 0:
                rep_days_qubo = list(np.linspace(0, num_days - 1, 3, dtype=int))

            # 4d) Weights in PCA space for aggregated run
            weights_qubo = compute_day_weights(Z3, rep_days_qubo)
            print("Representative Day Weights (date: weight):")
            for idx, w in weights_qubo.items():
                date = load_season.index[idx * 24].strftime('%Y-%m-%d')
                print(f"  {date}: {w}")

            # 4e) Plots using selected reps
            # PCA(3) scatter with reps (existing helper)
            plot_pca3_scatter(Z3, rep_days_qubo, case_name, season)
            # Seasonal time-series with rep-day markers (existing helper)
            plot_season_profiles_with_reps(load_season, wind_season, hydro_season, rep_days_qubo, case_name, season)
            # t-SNE with case name in the filename/title
            plot_tsne_2d_from_pca3(Z3, rep_days_qubo, case_name, season, perplexity=15)

            # 4f) Aggregated model using representative days + weights
            net_qubo = build_pypsa_network(
                network_data,
                load_season,
                wind_season,
                rep_days_qubo,
                hydro_series=hydro_series_full,
                day_weights=weights_qubo
            )
            net_qubo = run_lopf(net_qubo)
            cost_qubo = net_qubo.objective
            deviation = abs(cost_qubo - cost_full) / cost_full * 100 if cost_full else np.nan

            print(f"[{case_name} - {season}] Full: {cost_full:.2f}, PCA3-QUBO: {cost_qubo:.2f}, Dev: {deviation:.2f}%")
            summary_data.append({
                "Case": case_name,
                "Season": season,
                "Full Cost": cost_full,
                "PCA3-QUBO Cost": cost_qubo,
                "Deviation (%)": deviation,
                "Rep days": len(rep_days_qubo)
            })

    # 5) Summaries and the 5 missing summary charts
    results_df = pd.DataFrame(summary_data)
    results_df.to_excel("pca3_qubo_summary.xlsx", index=False)

    # Generate the five summary figures (bar/line/heatmap set)
    if not results_df.empty:
        plot_summary_figures(results_df)

    print("\n=== Summary ===")
    try:
        print(results_df.to_markdown(index=False))
    except Exception:
        print(results_df)
    print("Process completed successfully!")


if __name__ == "__main__":
    main()