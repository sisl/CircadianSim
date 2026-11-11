# 2b_SensitivityRevision

Reviewer-response analyses for the PLOS Computational Biology revision (Fig 2; S3, S4 and S7–S9 Figs):

- **1.3**: how much sampling ψ widens the observed phase offset ψ_obs
- **2.4**: do the Spearman and SHAP conclusions depend on the parameter SDs? (SD sweep)
- **2.6**: report effect sizes rather than significance
- **3.6** (R3.7 in the response letter): goodness of fit of the Random Forest surrogates, and SHAP stability across
  seeds and folds, for the regular schedule (Fig 2). The jet-lag out-of-fold R² (S12 Fig) is computed by
  `3_JetLagAnalysis/computeAdaptationSHAPVariants.jl` in experiment 3.

## Revision baseline

Defined in `experiments/revision_baseline.jl` (loaded through `jetlag_reference.jl`), so experiments 2–4 and this
folder use one definition. It differs from the published experiment 2 as follows:

| | Experiment 2 (published) | Revision baseline |
|---|---|---|
| Light | stochastic (10 % noise) | stochastic (same noise), one stream per particle, `StableRNG(SIMULATION_SEED + id)` |
| Run / scoring window | 35 d, days 20.3–35 | 70 d, days 56–70 |
| NHANES wake alignment | 1.143 h | 1.341 h (`calibrate_nhanes_alignment.jl`) |
| Process C phase lag ψ | ψ ~ N(0, 2²) h, entering as a constant offset of C (≡ a shift of A_v) | rotation of the drive by 2πψ/τ, ψ fixed at 0 (the unshifted Phillips–Robinson / St Hilaire drive) |
| Exclusions | < 13 sleep episodes in window, per condition | asleep during part of every night (lights off about 22:40–06:40), and CBTmin and sleep midpoint within ±1 h of their window means, in **both** conditions |
| Code | original submission (before the Vm/Vv fix) | this release (Vm/Vv fix, ψ_obs wrap, Process C rotation) |

The MLE summaries are computed the same way as in experiment 2 (`compute_mle_population_summaries`).

Why each change:

- **70-day run.** In days 20.3–35, about 38 % of LowLight particles are still drifting by ≥ 1 min/day; nearly all of
  them lock by days 52.5–70.
- **Exclusion rule.** In the late window almost every exclusion is a particle that never sleeps, sleeps only every
  other night, or sleeps only during the day ("flipped").
  - **Timing.** Early sleepers (asleep before lights-off) and late sleepers are kept, as long as they are asleep for
    part of every night.
  - **Earlier version.** A "≥ 1 sleep onset per calendar day" test was dropped because it misclassified particles
    whose onset alternates either side of midnight.

**One exclusion set for all analyses.** A particle that fails the rule in either light condition is excluded from
every analysis, so both conditions use the same particles (`2_RegularScheduleAnalysis/selectEntrainedParticles.jl`
writes the set; `included_particle_ids()` loads it).

Main analysis population: **967 of 1,000 particles**, the same in both conditions (33 excluded, 15 of them in both
conditions).

**ψ fixed at 0.** The literature on the sleep–circadian phase angle constrains the observed phase angle ψ_obs (an
output, between-person SD ≈ 1–1.2 h), not a lag of the circadian drive itself, and τ, light and the homeostat already
spread ψ_obs (SD ≈ 0.6 h at ψ = 0). ψ is therefore not sampled; it drops out of the input–output analyses
(`varying_inputs` in `jetlag_reference.jl`), and the SD sweep adds it back at SD 0.125–2 h.

## Scripts

| Script | Purpose |
|---|---|
| `sensitivity_common.jl` | Shared setup on top of the revision baseline: re-simulating particle sets (SD sweep, calibration) and experiment 2's 35-day settings for comparison |
| `build_deterministic_baseline.jl` | Builds the baseline summaries used by 2.6 and 3.6 (`results/deterministic_baseline_summaries.jld2`, gitignored, ~1 min) and compares ρ/SHAP with the 35-day window. The name is from an earlier deterministic-light version; the baseline uses stochastic light |
| `calibrate_nhanes_alignment.jl` | Calibrates the NHANES wake alignment (1.341 h) |
| `effect_sizes_spearman.jl` | 2.6: Spearman ρ with bootstrap CIs and Holm-adjusted p-values |
| `rf_fit_and_shap_stability.jl` | 3.6: regular-schedule RF fit (out-of-fold R²) and SHAP stability; its figures are drawn by `plot_rf_fit_and_shap_stability.jl` from the saved CSVs (run alone to redraw them) |
| `sd_sweep.jl`, `plot_sd_sweep.jl` | 2.4: rescaled parameter SDs and uniform priors; ψ added at SD 0.125–2 h on top of ψ = 0 |
| `psi_obs_spread.jl` | 1.3: spread of ψ_obs (and CBTmin, sleep timing) and the exclusion counts as the ψ SD grows, for the sweep’s ψ scenarios (~5 min on 8 threads) |
| `run_2b_final.sh` | Reruns the 2b analyses above |
| `run_exp3_pipeline.sh` | Regenerates experiment 3 (Figs 3–4 and supplements) on the revision baseline |

Experiment 4 (Fig 5, S14 Fig) runs on Sherlock (`experiments/4_JetLagPolicy/sherlock/`).

Run every script from the repository root with `julia -t auto --project=. <script>`.
