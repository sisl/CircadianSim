#!/bin/zsh
# Regenerate experiment 3 (jet lag; Fig 4 and its supplements) on the revision baseline. Run from the repository root.
# Post-shift window per condition (INTERVENTION_TARGET_DAYS: 150 lux 100 d, 10,000 lux 50 d), adaptation target from
# the settled pre-shift phase, and non-adapters counted at the window length (see detectSteadyState.jl). Existing
# history files for a window are kept. The trajectory figures (Fig 3, S10–S11 Figs) show days −7 to 50, which the longer
# 150 lux window does not change; pass "trajectories" to redraw them (and the four-shift figure) anyway. Pass "shap" to skip the simulation,
# detection and statistics steps and redo only the SHAP fits and Fig 4.
set -e
L=experiments/2b_SensitivityRevision/results/pipeline_logs
mkdir -p $L
J() { julia -t auto --project=. "$@"; }
if [[ ${1:-} != shap ]]; then
J experiments/3_JetLagAnalysis/generateJetLagParticleHistories.jl > $L/3_generateJetLag.log 2>&1
for c in LowLight150lux HighLight10000lux; do
  J experiments/3_JetLagAnalysis/detectSteadyState.jl $c > $L/3_detectSteadyState_$c.log 2>&1
  J experiments/3_JetLagAnalysis/summarizeAdaptationStatistics.jl $c > $L/3_summarize_$c.log 2>&1
  if [[ ${1:-} == trajectories ]]; then
    J experiments/3_JetLagAnalysis/plotJetLagTrajectories.jl $c > $L/3_plotTrajectories_$c.log 2>&1
  fi
done
if [[ ${1:-} == trajectories ]]; then
  J experiments/3_JetLagAnalysis/plotJetLagTrajectoriesFourShifts.jl > $L/3_plotFourShifts.log 2>&1
fi
fi
# SHAP / RF fits (all three feature sets) on log(1 + days) (main) and raw days (supplementary check)
for t in logdays days; do
  SHAP_TARGET=$t J experiments/3_JetLagAnalysis/computeAdaptationSHAPVariants.jl > $L/3_computeAdaptationSHAPVariants_$t.log 2>&1
done
# Fig 4 ("parameters noR2"), S12 Fig ("parameters shaponly") and S13 Fig ("outputs shaponly")
for v in "parameters noR2" "parameters shaponly" "outputs shaponly"; do
  J experiments/3_JetLagAnalysis/plotCombinedLightConditions.jl ${=v} > $L/3_plotCombined_${v// /_}.log 2>&1
done
echo "exp3 done"
