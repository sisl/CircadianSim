# Quickstart

**Start here** for reproducibility and to try the package with minimal friction.

## What you need for reproducibility

- **Project.toml** (and optionally **Manifest.toml**) in the repo — so `Pkg.instantiate()` reproduces the same environment.
- **Run order** — run the scripts below in the order shown when doing the full paper pipeline.

## Quickstart scripts (run from repo root)

| Order | Script | Purpose |
|-------|--------|---------|
| — | `02_minimal_jetlag.jl` | **No setup required.** Run anytime to verify the install (short simulation, 50 particles, 3 days). Use this first to confirm everything works. |
| 1 | `01_setup_data_dirs.jl` | Run **once** before the paper experiments. Creates `Data/JetLag/` and subdirectories so the experiments can save outputs. |
| then | See [Reproducing the paper](../README.md#reproducing-the-paper) | Paper experiments live under `experiments/`; run them in the order given in the main README. |

### Commands (from repo root)

```bash
# Verify install (no setup, no saved data)
julia --project=. quickstart/02_minimal_jetlag.jl

# One-time setup for paper experiments (creates Data/JetLag/)
julia --project=. quickstart/01_setup_data_dirs.jl
```

After setup, run the paper pipeline in the order given in [Reproducing the paper](../README.md#reproducing-the-paper) in the main README.
Paths and constants are defined in `experiments/jetlag_reference.jl` and `experiments/revision_baseline.jl`.

## Repo structure (where things live)

| Path | Purpose |
|------|---------|
| `quickstart/` | This folder — minimal example + one-time setup. |
| `experiments/` | Paper pipeline: `1_GenerateInitialParticles/`, `2_RegularScheduleAnalysis/`, `3_JetLagAnalysis/`, `4_JetLagPolicy/`, and the pipeline scripts and reviewer analyses in `2b_SensitivityRevision/`. |
| `experiments/jetlag_reference.jl` | Single source for paths and constants. |
| `src/` | Package code. |
| `test/` | Test suite (`julia --project=. -e 'using Pkg; Pkg.test()'`). |
