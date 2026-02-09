# Quantum-Inspired Temporal Aggregation for Power System Capacity Expansion

> A quantum-compatible workflow that selects non-uniformly weighted representative days via QUBO->QAOA and plugs them into a PyPSA-based capacity expansion model.

## Latest Abstract

This study extends previous work on the application of quantum optimization to temporal aggregation in capacity expansion planning (CEP) for power systems. The previous approach used fixed time windows to limit problem size, allowing the Quantum Approximate Optimization Algorithm (QAOA) to identify representative days within each partition. While effective, this strategy introduced artificial boundaries that weakened the representation of chronological patterns across the year.

The current approach reconstructs the problem as a Max-Cut equivalent quadratic unconstrained binary optimization (QUBO) model to overcome this limitation. Warm-starts based on semidefinite programming (SDP) are used to enhance convergence and solution quality in shallow-depth QAOA circuits. Piecewise-linear transitions between representative days and a-posteriori refinements informed by energy storage and network congestion behavior retain chronological dynamics. The study tests this framework on standard IEEE test systems to evaluate representativeness, computational efficiency, and chronological accuracy. By removing the need for temporal partitioning, the work aims to enhance the scalability, realism, and practical integration of quantum-assisted temporal aggregation within next-generation energy system planning tools.

## Latest Details

- Latest runtime (seconds): 687.8859031 (about 11.5 minutes) from `run_runtime_seconds.txt`.
- Production time range: 2018-12-01 to 2019-11-30 (inclusive).
- Time-series inputs: `research/datasets/time_series_60min_singleindex.csv` (AT load/wind) plus synthetic hydro.
- IEEE cases: `research/datasets/case9.mat`, `case30.mat`, `case118.mat`.
- Outputs: `warm-start-qaoa/results/` with plots in `results/plots/` and figures in `results/figures/`.

## Citation

If you use this repository in academic or industrial research, please cite:

**A. Singha, S. Mishra, M. Shafie-khah,  
_Application of quantum computing to temporal aggregation for efficient capacity expansion in power systems_,  
International Journal of Electrical Power & Energy Systems, vol. 173, 2025.**  
https://doi.org/10.1016/j.ijepes.2025.111355


```bibtex
@article{Singha2025_QuantumAggregation,
  title   = {Application of quantum computing to temporal aggregation for efficient capacity expansion in power systems},
  author  = {Singha, Ankana and Mishra, Sambeet and Shafie-khah, Miadreza},
  journal = {International Journal of Electrical Power & Energy Systems},
  year    = {2025},
  volume  = {173},
  pages   = {111355},
  issn    = {0142-0615},
  doi     = {10.1016/j.ijepes.2025.111355},
  url     = {https://www.sciencedirect.com/science/article/pii/S0142061525009032},
  keywords = {Quantum computing, Representative days, Capacity expansion planning, Time aggregation, Renewable energy, Optimization}
}

```
## Context

Modern power system planning requires high-resolution temporal data to capture variability in load and renewable resources, but full-year hourly simulations are expensive. This repository integrates **QUBO-formulated** representative-day selection solved by **QAOA**, then evaluates investment and operations using **PyPSA (LOPF)**. Across IEEE test systems and seasonal demand types, the approach preserves planning-cost fidelity while drastically shrinking the temporal dimension.

## Features

1. **QUBO->QAOA Representative-Day Selection** -- Day selection is posed as a QUBO and optimized with QAOA.
2. **Non-Uniform Weights** -- Each chosen day stands in for multiple real days via cluster-size weights.
3. **CEP Integration (PyPSA LOPF)** -- Weighted representative days drive capacity investment and dispatch.
4. **Seasonal Windowing** -- Half-month windows bound each QUBO to a tractable size for simulators and near-term hardware.

## Experiments & Evaluation

Three presets are provided (per season):

- **Experiment 1** -- `k = 2`, **12** representative days/season -- [`experiment_1.py`](experiment_1.py)  
- **Experiment 2** -- `k = 3`, **18** representative days/season -- [`experiment_2.py`](experiment_2.py)  
- **Experiment 3** -- `k = 4`, **24** representative days/season -- [`experiment_3.py`](experiment_3.py)

**Systems & Seasons:** Winter, Spring, Summer, Autumn on **IEEE 9-bus**, **IEEE 30-bus**, and **IEEE 118-bus** networks (MATPOWER canonical cases). Full-resolution vs. aggregated runs are compared for total system cost deviation and dispatch fidelity.

## Sensitivity Analysis

The paper performs a dedicated sensitivity study to test robustness of the QAOA-based selection and the accuracy-compression tradeoff:
**Sensitivity Analysis** -- `k = 3`, [`sensitivity_analysis.py`](sensitivity_analysis.py)  

### A. QAOA Settings (fixed `k = 3`)
- **Circuit depth (`p`)**: {1, 2, 3}  
- **Initialization of variational angles (`gamma, beta`)**: random, warm-start, and zero-init  
- **Optimizer (COBYLA) iteration limits**: 100, 300, 1000  

**Findings (IEEE-30, Winter):**
- The **selected representative-day sets are identical** across all tested configurations (pairwise overlap **100%**).  
- **Cost deviation remains constant at 4.53%** across depths, inits, and iterations, indicating a **stable QAOA landscape** with convergence to a consistent near-global solution.

### B. Number of Representative Days (`k in {2, 3, 4}`)
- **k = 2** (12 days/season): largest deviations; Spring worst (~8-9%), Summer/Autumn >6% in multiple systems.  
- **k = 3** (18 days/season): sharp improvement; deviations cluster narrowly around **~4.5-5.1%** across systems and seasons (good balance of fidelity and size).  
- **k = 4** (24 days/season): **incremental** gains beyond k=3; seasonal peaks reduce modestly (e.g., Autumn ~4.2%).  
- Overall: the **biggest step-change is k=2->3**; **diminishing returns** beyond k=3.

### C. Seasonal & System Effects
- With **k = 3**, deviations are ~**5%** across **IEEE-9/30/118** and seasons; Spring tends to be lowest (~4.5-4.9%), Winter/Autumn slightly higher (~5.0-5.1%).  
- Representative days span both **typical clusters and peripheral transitional regimes**, capturing peaks, troughs, and ramps.


## Workflow

![Flow Chart](docs/images/program_flow.png "Flow Chart")

1. **Preprocess** hourly load, wind, hydro; construct daily feature vectors.
2. **Seasonal windowing + PCA** in each half-month segment.
3. **QUBO formulation** + **QAOA solve** to pick representative days.
4. **Apply non-uniform weights** from cluster sizes.
5. **Run PyPSA CEP (LOPF)** with weighted snapshots.
6. **Compare** against full-resolution baselines (cost, dispatch, runtime).

## Reference Data

- **Load & Wind:** Open Power System Data -- Time Series  
- **Hydro:** Synthetic seasonal profile based on capacity parameters  
- **Horizon:** **Dec 2018 -> Nov 2019**, split by seasons; each season further segmented into **half-month windows** so each QUBO ~**15-16 days** (~**2^15-2^16** scale) for current simulators.

## Installation

### Prerequisites
- Python **3.10+**
- `pip`

### Setup

```bash
# Clone
git clone <your-repo-url>.git
cd <repo-folder>

# Create & activate venv
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Known Limitations

- Current workflow uses **classical QAOA simulation**; speedups are expected with quantum/hybrid backends.
- Accuracy-compression tradeoff depends on **k** and seasonal variability; **k = 3** is a strong default, but users should validate for their systems.

## Acknowledgements

- **PyPSA** -- https://pypsa.readthedocs.io/
- **Qiskit** -- https://qiskit.org/
- **Open Power System Data (OPSD)** -- https://open-power-system-data.org/