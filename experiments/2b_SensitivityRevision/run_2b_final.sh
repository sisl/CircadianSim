#!/bin/zsh
# Rerun the reviewer analyses at the final revision settings (stochastic light, 70-day baseline, 1.341 h
# alignment, shared particle set). Run from the repository root.
set -e
L=experiments/2b_SensitivityRevision/results/pipeline_logs
mkdir -p $L
B=experiments/2b_SensitivityRevision
J() { julia -t auto --project=. "$@"; }
J $B/build_deterministic_baseline.jl > $B/results/build_deterministic_baseline.log 2>&1
J $B/calibrate_nhanes_alignment.jl > $L/2b_calibrate_nhanes_alignment.log 2>&1
J $B/effect_sizes_spearman.jl > $B/results/effect_sizes_spearman.log 2>&1
J $B/rf_fit_and_shap_stability.jl > $B/results/rf_fit_and_shap_stability.log 2>&1
rm -rf $B/results/sd_sweep   # scenario checkpoints, so sd_sweep.jl reruns every scenario
J $B/sd_sweep.jl > $B/results/sd_sweep.log 2>&1
J $B/plot_sd_sweep.jl
J $B/psi_obs_spread.jl > $B/results/psi_obs_spread.log 2>&1
echo "2b final done"
