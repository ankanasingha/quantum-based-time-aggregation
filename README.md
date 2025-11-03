# Quantum-Inspired Temporal Aggregation for Power System Capacity Expansion

> A quantum-compatible workflow that selects non-uniformly weighted representative days via QUBO→QAOA and plugs them into a PyPSA-based capacity expansion model.

## Context

Modern power system planning requires high-resolution temporal data to capture variability in load and renewable resources, but full-year hourly simulations are expensive. This repository integrates **QUBO-formulated** representative-day selection solved by **QAOA**, then evaluates investment and operations using **PyPSA (LOPF)**. Across IEEE test systems and seasonal demand types, the approach preserves planning-cost fidelity while drastically shrinking the temporal dimension.

## Features

1. **QUBO→QAOA Representative-Day Selection** — Day selection is posed as a QUBO and optimized with QAOA.
2. **Non-Uniform Weights** — Each chosen day stands in for multiple real days via cluster-size weights.
3. **CEP Integration (PyPSA LOPF)** — Weighted representative days drive capacity investment and dispatch.
4. **Seasonal Windowing** — Half-month windows bound each QUBO to a tractable size for simulators and near-term hardware.

## Experiments & Evaluation

Three presets are provided (per season):

- **Experiment 1** — `k = 2`, **12** representative days/season — [`experiment_1.py`](experiment_1.py)  
- **Experiment 2** — `k = 3`, **18** representative days/season — [`experiment_2.py`](experiment_2.py)  
- **Experiment 3** — `k = 4`, **24** representative days/season — [`experiment_3.py`](experiment_3.py)

**Systems & Seasons:** Winter, Spring, Summer, Autumn on **IEEE 9-bus**, **IEEE 30-bus**, and **IEEE 118-bus** networks (MATPOWER canonical cases). Full-resolution vs. aggregated runs are compared for total system cost deviation and dispatch fidelity.

## Sensitivity Analysis

The paper performs a dedicated sensitivity study to test robustness of the QAOA-based selection and the accuracy–compression tradeoff:

### A. QAOA Settings (fixed `k = 3`)
- **Circuit depth (`p`)**: {1, 2, 3}  
- **Initialization of variational angles (`γ, β`)**: random, warm-start, and zero-init  
- **Optimizer (COBYLA) iteration limits**: 100, 300, 1000  

**Findings (IEEE-30, Winter):**
- The **selected representative-day sets are identical** across all tested configurations (pairwise overlap **100%**).  
- **Cost deviation remains constant at 4.53%** across depths, inits, and iterations, indicating a **stable QAOA landscape** with convergence to a consistent near-global solution.

### B. Number of Representative Days (`k ∈ {2, 3, 4}`)
- **k = 2** (12 days/season): largest deviations; Spring worst (~8–9%), Summer/Autumn >6% in multiple systems.  
- **k = 3** (18 days/season): sharp improvement; deviations cluster narrowly around **~4.5–5.1%** across systems and seasons (good balance of fidelity and size).  
- **k = 4** (24 days/season): **incremental** gains beyond k=3; seasonal peaks reduce modestly (e.g., Autumn ≈4.2%).  
- Overall: the **biggest step-change is k=2→3**; **diminishing returns** beyond k=3.

### C. Seasonal & System Effects
- With **k = 3**, deviations are ≈**5%** across **IEEE-9/30/118** and seasons; Spring tends to be lowest (~4.5–4.9%), Winter/Autumn slightly higher (~5.0–5.1%).  
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

- **Load & Wind:** Open Power System Data — Time Series  
- **Hydro:** Synthetic seasonal profile based on capacity parameters  
- **Horizon:** **Dec 2018 → Nov 2019**, split by seasons; each season further segmented into **half-month windows** so each QUBO ~**15–16 days** (≈**2^15–2^16** scale) for current simulators.

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
- Accuracy–compression tradeoff depends on **k** and seasonal variability; **k = 3** is a strong default, but users should validate for their systems.

## Acknowledgements

- **PyPSA** — https://pypsa.readthedocs.io/
- **Qiskit** — https://qiskit.org/
- **Open Power System Data (OPSD)** — https://open-power-system-data.org/

