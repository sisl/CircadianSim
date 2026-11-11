# CircadianPOMDP

[![Build Status](https://github.com/sisl/CircadianSim/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/sisl/CircadianSim/actions/workflows/CI.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/sisl/CircadianSim/branch/main/graph/badge.svg)](https://app.codecov.io/gh/sisl/CircadianSim)
[![Julia](https://img.shields.io/badge/Julia-1.11%2B-blue?logo=julia)](https://julialang.org/)
[![License: Stanford Academic](https://img.shields.io/badge/License-Stanford%20Academic%20Software-red)](LICENSE)

Julia package for **circadian rhythm and sleep–wake modeling** with particle-based simulation, including **jet lag experiments**: baseline runs, timezone shifts, light interventions, and adaptation-time estimation.

This repository contains the code for Weed, Jamgochian, Kochenderfer and Zeitzer, *CircadianSim: Population circadian modeling reveals variability in sleep timing, jet lag adaptation, and benefits of personalized light interventions* (manuscript under review at PLOS Computational Biology). The paper calls the forward-simulation model **CircadianSim**. The Julia package in this repository is named `CircadianPOMDP` (`using CircadianPOMDP`).

## Installation

Julia 1.11 or later (CI tests 1.11 and 1.12). From the Julia REPL:

```julia
using Pkg
Pkg.add(url="https://github.com/sisl/CircadianSim")
```

Or clone and use the project:

```bash
git clone https://github.com/sisl/CircadianSim.git
cd CircadianSim
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

### Additional requirements for the paper experiments

- **Python, through PyCall.** The Random Forest and SHAP fits use scikit-learn and shap, and the NHANES file is read with pyreadstat. Install the three packages in one Python environment and point PyCall at it:
  ```julia
  ENV["PYTHON"] = "/path/to/python"   # environment with scikit-learn, shap and pyreadstat
  using Pkg; Pkg.build("PyCall")
  ```
  The paper results were produced with scikit-learn 1.8.0 and shap 0.50.0.
- **NHANES data (Fig 1, S2 Fig).** Download the NHANES August 2021–August 2023 Sleep Disorders file [SLQ_L.xpt](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/SLQ_L.xpt) from the CDC and save it as `Data/NHANES/SLQ_L_2021_2023.xpt`. `2b_SensitivityRevision/calibrate_nhanes_alignment.jl` also reads a CSV export of the same file, `Data/NHANES/SLQ_L_2021_2023.csv`.
- **Compute.** Experiments 1–3 and the reviewer analyses run on a workstation. Experiment 4 (300 single-day light interventions × 23 shifts × 2 light levels) takes roughly 75 hours on 8 CPUs; `experiments/4_JetLagPolicy/sherlock/` contains the Slurm scripts used to run it on a cluster.

## Reproducibility

- **Environment.** `Project.toml` sets compatible version ranges for all dependencies. `Manifest.toml` is not tracked, so `Pkg.instantiate()` resolves the newest compatible versions.
- **Determinism.** Simulations use fixed seeds (`SIMULATION_SEED` in `experiments/jetlag_reference.jl` for the main pipeline) with one random-number stream per participant, so results do not depend on the number of threads.
- **Outputs.** Generated data and figures are written under `Data/JetLag/`, and the reviewer analyses write to `experiments/2b_SensitivityRevision/results/` and `figures/`; none of these are tracked. Paths and constants for every experiment are defined in `experiments/jetlag_reference.jl` and `experiments/revision_baseline.jl`.

## Quickstart (start here)

**[quickstart/](quickstart/)** contains everything needed to get going:

| What | Command (run from repo root) |
|------|------------------------------|
| **Verify install** (no setup, no saved data) | `julia --project=. quickstart/02_minimal_jetlag.jl` |
| **One-time setup** for paper experiments | `julia --project=. quickstart/01_setup_data_dirs.jl` |

See **[quickstart/README.md](quickstart/README.md)** for more detail.

## Reproducing the paper

Run from the repository root, in this order. The pipeline scripts use zsh.

1. **Setup (once):**
   `julia --project=. quickstart/01_setup_data_dirs.jl`
2. **Initial particles** (1,000 virtual participants):
   `julia --project=. experiments/1_GenerateInitialParticles/initializeParticles.jl`
3. **Experiment 2, regular schedule:**
   `zsh experiments/2b_SensitivityRevision/run_exp2_pipeline.sh`
   - Runs the 70-day baseline histories and the per-participant summaries, then selects the shared set of 967 entrained participants used by every later step.
   - Then computes the Spearman correlations and SHAP and draws Fig 1, Fig 2, S2 Fig, S5 Fig and S6 Fig.
   - Pass `core` to stop after the shared participant set.
4. **Experiment 3, jet lag without intervention:**
   `zsh experiments/2b_SensitivityRevision/run_exp3_pipeline.sh trajectories`
   - Runs westward shifts of 1–23 h, adaptation-time detection and the adaptation-time SHAP fits.
   - Draws Fig 3, Fig 4 and S10–S13 Figs.
   - Without `trajectories` the script skips the trajectory figures; `shap` redoes only the SHAP fits and Fig 4.
5. **Experiment 4, single-day light interventions** (needs Experiment 2):
   - Locally: `julia --project=. -t auto experiments/4_JetLagPolicy/rescuingJetLag_unified.jl`. For a short test, add `mini` as the first argument.
   - On a Slurm cluster: `bash experiments/4_JetLagPolicy/sherlock/exp4_sherlock.sh setup|test|submit|status`. See the script header.

   Then draw the figures:
   ```bash
   julia --project=. experiments/4_JetLagPolicy/plot_baseline_vs_intervention_wrapped_quartiles_steadyState.jl  # Fig 5
   julia --project=. experiments/4_JetLagPolicy/plot_personal_best_worst_change.jl                             # S14 Fig
   ```
6. **Reviewer analyses** (after Experiment 3):
   - `zsh experiments/2b_SensitivityRevision/run_2b_final.sh` runs the NHANES alignment calibration, the Spearman effect sizes, the RF out-of-fold R² and SHAP stability, the prior-SD sweep and the ψ_obs spread. It draws S3, S4 and S7–S9 Figs.
   - `julia --project=. experiments/2_RegularScheduleAnalysis/plot_excluded_participants.jl` draws S1 Fig.

   See [experiments/2b_SensitivityRevision/README.md](experiments/2b_SensitivityRevision/README.md).

### Figures and the scripts that draw them

| Figure | Script under `experiments/` (arguments) |
|---|---|
| Fig 1, S2 Fig | `2_RegularScheduleAnalysis/compareWithNHANES.jl` (`LowLight150lux`, `HighLight10000lux`) |
| Fig 2 | `2_RegularScheduleAnalysis/plotCorrelationsAndSHAPCombined.jl` |
| Fig 3 | `3_JetLagAnalysis/plotJetLagTrajectoriesFourShifts.jl` |
| Fig 4 | `3_JetLagAnalysis/plotCombinedLightConditions.jl` (`parameters noR2`) |
| Fig 5 | `4_JetLagPolicy/plot_baseline_vs_intervention_wrapped_quartiles_steadyState.jl` |
| S1 Fig | `2_RegularScheduleAnalysis/plot_excluded_participants.jl` |
| S3, S4, S9 Figs | `2b_SensitivityRevision/plot_sd_sweep.jl` |
| S5, S6 Figs | `2_RegularScheduleAnalysis/computeSHAP.jl` (per light condition) |
| S7, S8 Figs | `2b_SensitivityRevision/plot_rf_fit_and_shap_stability.jl` |
| S10, S11 Figs | `3_JetLagAnalysis/plotJetLagTrajectories.jl` (per light condition) |
| S12 Fig | `3_JetLagAnalysis/plotCombinedLightConditions.jl` (`parameters shaponly`) |
| S13 Fig | `3_JetLagAnalysis/plotCombinedLightConditions.jl` (`outputs shaponly`) |
| S14 Fig | `4_JetLagPolicy/plot_personal_best_worst_change.jl` |

## In-code quick start (REPL or script)

```julia
using CircadianPOMDP
using Dates

params = model_parameters()
particles = initialize_particles(1000, DateTime(2023, 1, 1); state_params_mode=:sampled_normal)
LD = generate_light_diet_basic(days=35, lux_on=10_000.0,
    light_on=REGULAR_SCHEDULE_LIGHT_ON_H, light_off=REGULAR_SCHEDULE_LIGHT_OFF_H,
    lux_off=0.0, sleep_light=missing, sleep_dark=missing)
cbtmin_events, onset_ev, offset_ev = simulate_particle_population_events_only(
    particles, LD, params, circadian_transition_sleep, DateTime(2023, 1, 1); dt=1/60, verbose=false)
base_traj, base_counts = baseline_cbtmin_from_events(cbtmin_events, length(particles), 35*24)
summaries = compute_mle_population_summaries(
    cbtmin_events, onset_ev, offset_ev, particles; verbose=false)
```

## Structure

| Path | Purpose |
|------|---------|
| **quickstart/** | Minimal example and one-time setup; see [quickstart/README.md](quickstart/README.md). |
| **experiments/** | Paper pipeline: `1_GenerateInitialParticles/`, `2_RegularScheduleAnalysis/`, `3_JetLagAnalysis/`, `4_JetLagPolicy/`, and the reviewer analyses in `2b_SensitivityRevision/`. Shared paths and constants are in `jetlag_reference.jl` and `revision_baseline.jl`; figure style is in `figure_style.jl`. |
| **src/** | Package code: circadian model, light diets, simulation, CBTmin and sleep detection, and analysis. |
| **test/** | Test suite; see [test/TEST_DOCUMENTATION.md](test/TEST_DOCUMENTATION.md). |

## Additional documentation

- **Reviewer analyses:** [experiments/2b_SensitivityRevision/README.md](experiments/2b_SensitivityRevision/README.md)
- **Tests:** [test/TEST_DOCUMENTATION.md](test/TEST_DOCUMENTATION.md)
- **In Julia:** `?CircadianPOMDP` or `@doc CircadianPOMDP` for an API overview and examples

## Testing

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

## Contact

For questions, bug reports, collaborations, or consulting, contact [weed@stanford.edu](mailto:weed@stanford.edu).

## License and citation

Stanford Academic Software License (Stanford Docket S26-030) — see [LICENSE](LICENSE). For commercial licensing, contact [otl@stanford.edu](mailto:otl@stanford.edu).  
**Citation:** [CITATION.cff](CITATION.cff). When the accompanying paper is published, add its details to the `preferred-citation` block in CITATION.cff and cite both the paper and the software.
