#!/bin/zsh
# Regenerate experiment 2 (regular schedule; Figs 1–2, S2, S5–S6 Figs) on the revision baseline. Run from the repository root.
# Pass "core" to stop after the histories, MLE summaries and the shared particle set (what experiment 3 needs), or
# "figures" to skip those steps and redo only the correlations, SHAP, Fig 2 and the NHANES comparison.
set -e
L=experiments/2b_SensitivityRevision/results/pipeline_logs
mkdir -p $L
D=experiments/2_RegularScheduleAnalysis
J() { julia -t auto --project=. "$@"; }
if [[ ${1:-} != figures ]]; then
J $D/generateRegularScheduleParticleHistories.jl > $L/2_generateRegular.log 2>&1
for c in LowLight150lux HighLight10000lux; do
  J $D/regularSchedule_MLE_particleSummary.jl $c > $L/2_summary_$c.log 2>&1
done
J $D/selectEntrainedParticles.jl > $L/2_selectEntrained.log 2>&1
fi
[[ ${1:-} == core ]] && { echo "exp2 core done"; exit 0; }
for c in LowLight150lux HighLight10000lux; do
  J $D/computeCrossCorrelations.jl $c > $L/2_crosscorr_$c.log 2>&1
done
# SHAP twice: each condition's beeswarm x axis is the union with the other condition's saved SHAP range (S5 = S6)
for r in 1 2; do
  for c in LowLight150lux HighLight10000lux; do
    J $D/computeSHAP.jl $c > $L/2_shap_$c.log 2>&1
  done
done
J $D/plotCorrelationsAndSHAPCombined.jl > $L/2_fig2.log 2>&1
for c in LowLight150lux HighLight10000lux; do
  J $D/compareWithNHANES.jl $c > $L/2_nhanes_$c.log 2>&1
done
echo "exp2 done"
