import os
import numpy as np
import pandas as pd
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
from itertools import combinations
import json
from qiskit.primitives import Sampler
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit.algorithms.minimum_eigensolvers import QAOA, NumPyMinimumEigensolver
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from pandas.plotting import parallel_coordinates
from itertools import combinations

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
    end_date = pd.to_datetime("2019-02-28")

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


# PCA(3) feature extraction on 24h × [L,W,H]

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


def build_qubo_for_rep_days(daily_features, k, penalty=1e4):
    """Return QuadraticProgram that selects k representative days using a QUBO."""
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
    return qp


def solve_qubo_with_qaoa(
        qp,
        reps=1,
        maxiter=300,
        tol=None,
        initial_point=None,
        sampler=None,
        record_history=True,
        run_tag="run"
):
    from qiskit.primitives import Sampler
    if sampler is None:
        sampler = Sampler()

    history = {
        "eval_count": [],
        "objective": [],
        "stepsize": [],
        "accepted": [],
        "params": []
    }

    def cb(*cb_args):
        if not record_history:
            return
        if len(cb_args) == 1:
            # SciPy COBYLA: callback(xk)
            xk = np.array(cb_args[0])
            history["eval_count"].append(len(history["eval_count"]) + 1)
            history["params"].append(xk.tolist())
            history["objective"].append(None)  # unknown in SciPy callback
            history["stepsize"].append(None)
            history["accepted"].append(None)
        elif len(cb_args) == 5:
            # Qiskit COBYLA: callback(nfev, params, value, stepsize, accepted)
            nfev, parameters, value, stepsize, accepted = cb_args
            history["eval_count"].append(int(nfev))
            history["params"].append(np.array(parameters).tolist())
            history["objective"].append(float(value))
            history["stepsize"].append(float(stepsize) if stepsize is not None else None)
            history["accepted"].append(bool(accepted))
        else:
            # Unknown signature; store what we can
            history["eval_count"].append(len(history["eval_count"]) + 1)
            history["params"].append([float(x) for x in np.ravel(cb_args)])
            history["objective"].append(None)
            history["stepsize"].append(None)
            history["accepted"].append(None)

    optimizer = COBYLA(maxiter=maxiter, tol=tol, callback=cb)

    qaoa = QAOA(optimizer=optimizer, reps=reps, sampler=sampler, initial_point=initial_point)
    meo = MinimumEigenOptimizer(qaoa)
    result = meo.solve(qp)

    rep_days = [i for i, bit in enumerate(result.x) if bit > 0.5]
    hist_df = pd.DataFrame(history) if record_history and history["eval_count"] else pd.DataFrame(
        columns=["eval_count", "objective", "stepsize", "accepted", "params"]
    )
    final_val = result.fval
    return rep_days, hist_df, final_val


def compute_day_weights(full_features, rep_days):
    rep_day_weights = {r: 0 for r in rep_days}
    for i in range(len(full_features)):
        distances = [np.linalg.norm(full_features[i] - full_features[r]) for r in rep_days]
        closest_rep = rep_days[np.argmin(distances)]
        rep_day_weights[closest_rep] += 1
    return rep_day_weights

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
        for line in lines:
            bus0 = f"Bus_{int(line[0])}"
            bus1 = f"Bus_{int(line[1])}"
            x = line[3] if len(line) > 3 and line[3] > 0 else 0.1
            r = line[2] if len(line) > 2 and line[2] > 0 else 0.001
            n.add("Line", f"Line_{int(line[0])}_{int(line[1])}",
                  bus0=bus0, bus1=bus1, x=x, r=r, s_nom=500, carrier="AC")

    # Build snapshot list & weights from representative days (each is 24 hours)
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

# -----------------------------
# Plots
# -----------------------------
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
    plt.savefig(f"pca_explained_{season}.png", dpi=200)
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
    # ax.set_title(f"PCA(3) space — {case_name} / {season}")
    plt.tight_layout()
    plt.savefig(f"pca3_scatter_{case_name}_{season}.png", dpi=220)
    plt.show()


def plot_season_profiles_with_reps(load_season, wind_season, hydro_season, rep_days, case_name, season):
    """Time series for the season with vertical lines at representative-day starts."""
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(load_season.index, load_season['Load (MW)'] / 1e3, label='Load (10³ MW)', linewidth=2)
    ax.plot(wind_season.index, wind_season['Wind (MW)'] / 1e3, label='Wind (10³ MW)', linewidth=2)
    ax.plot(hydro_season.index, hydro_season / 1e3, label='Hydro (10³ MW)', linewidth=2)

    for d in rep_days:
        date = load_season.index[d * 24]  # first hour of that day
        ax.axvline(date, color='red', linestyle='--', alpha=0.6)

    ax.set_xlabel("Date")
    ax.set_ylabel("Power (×10³ MW)")
    ax.legend();
    ax.grid(True);
    plt.tight_layout()
    plt.savefig(f"season_profiles_with_reps_{case_name}_{season}.pdf", format="pdf")
    plt.show()


def plot_convergence(hist_df, title_stub):
    if hist_df.empty:
        return
    df = hist_df.dropna(subset=["objective"])
    if df.empty:
        return
    plt.figure(figsize=(7, 4))
    plt.plot(df["eval_count"], df["objective"], marker='o')
    plt.xlabel("Function evaluations")
    plt.ylabel("Objective (COBYLA)")
    # plt.title(f"Convergence — {title_stub}")
    plt.grid(True)
    safe = "".join([c if c.isalnum() or c in "-_." else "_" for c in title_stub])
    plt.tight_layout()
    plt.savefig(f"convergence_{safe}.png", dpi=200)
    plt.close()

# ---------------------------
# Main: Sensitivity study
# ---------------------------

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
    end_date = pd.to_datetime("2018-12-31")
    load_profiles_df = load_profiles_df.loc[start_date:end_date]
    wind_profiles_df = wind_profiles_df.loc[start_date:end_date]

    plot_yearly_profiles(load_profiles_df, wind_profiles_df)

    # Season
    def assign_season(month):
        if month in [12, 1, 2]:   return 'Winter'
        return 'Autumn'  # (only 'Winter' is used below)

    for df in (load_profiles_df, wind_profiles_df):
        df['Month'] = df.index.month
        df['Season'] = df['Month'].map(assign_season)

    seasons = ['Winter']
    seasonal_loads = {s: load_profiles_df[load_profiles_df['Season'] == s].copy() for s in seasons}
    seasonal_wind = {s: wind_profiles_df[wind_profiles_df['Season'] == s].copy() for s in seasons}
    seasonal_hydro = {}
    for s in seasons:
        dates = seasonal_loads[s].index
        seasonal_hydro[s] = pd.Series(simulate_hydro_profile(dates), index=dates)

    # IEEE cases
    ieee_cases = {
        'ieee30bus': 'C:/Users/Lenovo/Documents/case30.mat'
    }

    # QAOA sensitivity grids (COBYLA only)
    reps_grid = [1, 2, 3]  # QAOA depth P
    maxiter_grid = [100, 300, 1000]  # COBYLA settings
    rng = np.random.default_rng(42)  # for random initial points

    summary_rows = []

    for case_name, case_filename in ieee_cases.items():
        print(f"\n=== {case_name} ===")
        network_data = load_ieee_network(case_filename)

        for season in seasons:
            print(f"\n-- Season: {season} --")

            load_season = seasonal_loads[season].drop(columns=['Month', 'Season'], errors='ignore')
            wind_season = seasonal_wind[season].drop(columns=['Month', 'Season'], errors='ignore')
            if len(load_season) < 24:
                print("Not enough data; skipping.")
                continue
            hydro_season = seasonal_hydro[season]

            # Baseline FULL run (all days of the month)
            num_days = len(load_season) // 24
            hydro_series_full = pd.Series(hydro_season.values, index=load_season.index).astype(float)
            rep_days_full = list(range(num_days))
            net_full = build_pypsa_network(network_data, load_season, wind_season, rep_days_full,
                                           hydro_series=hydro_series_full)
            net_full = run_lopf(net_full)
            cost_full = float(net_full.objective)
            print(f"[Baseline FULL] cost = {cost_full:.4e}")

            # PCA(3) features
            X72 = build_daily_72d_matrix(load_season, wind_season, hydro_season)
            Z3, scaler, pca = pca_reduce_to_3(X72)
            plot_pca_explained_variance(pca, f"{case_name}_{season}")

            # Helper to list day indices per month, split halves
            def month_day_indices(m):
                idx = pd.Index(load_season.index)
                mask = (idx.month == m)
                hour0_mask = mask & (idx.hour == 0)
                day_indices = np.where(hour0_mask)[0] // 24
                return np.unique(day_indices)

            unique_months = sorted(pd.unique(pd.Index(load_season.index).month))

            # Sensitivity sweep
            for reps in reps_grid:
                # build initial_point grid matching the current reps (2*reps = gammas+betas)
                init_grid = [
                    [0.1] * (2 * reps),  # small angles (γ,β)
                    [np.pi / 4] * (2 * reps),  # mid angles
                    rng.uniform(0, np.pi, 2 * reps).tolist(),  # random
                ]
                init_grid = [ip for ip in init_grid if len(ip) == 2 * reps]

                for maxiter in maxiter_grid:
                    for init in init_grid:
                        # --- representative day selection per half-month using current hyperparams
                        rep_days_all_halves = []

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
                                qp = build_qubo_for_rep_days(feats_half, k=3, penalty=1e4)
                                rep_days_local, hist_df, fval = solve_qubo_with_qaoa(
                                    qp,
                                    reps=reps,
                                    maxiter=maxiter,
                                    tol=None,
                                    initial_point=init,
                                    sampler=Sampler(),
                                    record_history=True,
                                    run_tag=f"{case_name}_{season}_P{reps}_iter{maxiter}"
                                )
                                # Map local indices back to season-day indices
                                rep_days_all_halves.extend([half[i] for i in rep_days_local])

                                # Save convergence curve for this half
                                title_stub = f"{case_name}-{season}-P{reps}-iter{maxiter}-half{len(rep_days_all_halves)}"
                                plot_convergence(hist_df, title_stub)

                        rep_days_qubo = sorted(set(rep_days_all_halves))
                        if len(rep_days_qubo) == 0:
                            rep_days_qubo = list(np.linspace(0, num_days - 1, 3, dtype=int))

                        # Weights & PyPSA for the aggregated representative set
                        weights_qubo = compute_day_weights(Z3, rep_days_qubo)

                        net_qubo = build_pypsa_network(
                            network_data, load_season, wind_season, rep_days_qubo,
                            hydro_series=hydro_series_full, day_weights=weights_qubo
                        )
                        net_qubo = run_lopf(net_qubo)
                        cost_qubo = float(net_qubo.objective)
                        deviation = abs(cost_qubo - cost_full) / cost_full * 100 if cost_full else np.nan

                        print(f"[{case_name} - {season}] P={reps}, COBYLA(maxiter={maxiter}), "
                              f"init[:2]={init[:2]}... | "
                              f"cost={cost_qubo:.4e} | dev={deviation:.2f}%")

                        summary_rows.append({
                            "Case": case_name,
                            "Season": season,
                            "P (reps)": reps,
                            "COBYLA maxiter": maxiter,
                            "initial_point": init,
                            "Rep days (count)": len(rep_days_qubo),
                            "Cost (QUBO)": cost_qubo,
                            "Cost (FULL)": cost_full,
                            "Deviation (%)": deviation,
                            "Rep days": rep_days_qubo
                        })

    # Summarize results
    results_df = pd.DataFrame(summary_rows)

    # ---------- Pairwise overlap across sensitivity runs (equal set sizes) ----------

    def _run_key(row, idx):

        return f"P{row['P (reps)']}-it{row['COBYLA maxiter']}-{idx}"

    pairwise_outputs = []

    # Build per (Case, Season) so different datasets don't mix
    for (case_name, season), sub in results_df.groupby(["Case", "Season"], dropna=False):
        sub = sub.reset_index(drop=True).copy()
        keys = [_run_key(sub.loc[i], i) for i in range(len(sub))]
        sets = [set(sub.loc[i, "Rep days"]) for i in range(len(sub))]

        if not sets:
            continue

        k = len(next(iter(sets)))  # all sets same size
        n = len(sets)

        overlap_pct = np.zeros((n, n), dtype=float)
        jaccard = np.zeros((n, n), dtype=float)

        for i in range(n):
            A = sets[i]
            for j in range(n):
                B = sets[j]
                inter = len(A & B)
                # Equal-size simplifications
                overlap_pct[i, j] = (inter / k) * 100.0
                jaccard[i, j] = inter / (2 * k - inter) if (2 * k - inter) > 0 else np.nan

        Overlap_df = pd.DataFrame(overlap_pct, index=keys, columns=keys)
        Jaccard_df = pd.DataFrame(jaccard, index=keys, columns=keys)

        pairwise_outputs.append(((case_name, season), Overlap_df, Jaccard_df))

    # Save to a workbook
    with pd.ExcelWriter("qaoa_pairwise_overlap_equal_k.xlsx", engine="xlsxwriter") as xlw:
        for (case_name, season), Overlap_df, Jaccard_df in pairwise_outputs:
            base = f"{case_name}_{season}"
            Overlap_df.to_excel(xlw, sheet_name=f"{base}_Overlap%")
            Jaccard_df.to_excel(xlw, sheet_name=f"{base}_Jaccard")

    print("Saved: qaoa_pairwise_overlap_equal_k.xlsx")

    print("Process completed successfully!")


if __name__ == "__main__":
    main()






