#!/bin/bash
# Experiment 4 (jet-lag light interventions, Fig 5) on Stanford Sherlock.
#
# rescuingJetLag_unified.jl runs as three Slurm stages (see RESCUE_STAGE in that script):
#   1. baseline   one job: 70-day post-baseline states + no-intervention adaptation, both conditions
#   2. policies   job array: each task runs POLICIES_PER_TASK of the 300 policies × 23 shifts for one condition
#   3. aggregate  one job: merges the per-policy files into systematic_rescue_analysis_steadyState.jld2
# Every simulation uses SIMULATION_SEED with per-particle RNGs, so the split doesn't change the results.
#
# Cost: ~5 min per policy on 8 Sherlock CPUs with a 50-day post-shift window (sh02 nodes ~330 s, sh03/sh04 ~240 s);
# 150 lux runs a 100-day window (INTERVENTION_TARGET_DAYS), so its tasks take about twice as long. Tasks whose
# per-scenario event files already exist (same window and seed) only recompute adaptation times.
#
# On a Sherlock login node (code and inputs copied there with sync_sherlock.sh):
#   bash exp4_sherlock.sh setup     # once: Data -> $SCRATCH symlink, input check, Pkg.instantiate
#   bash exp4_sherlock.sh test      # baseline job + the first policy task; read its log for the time per policy
#   bash exp4_sherlock.sh submit    # everything still missing; rerun after timeouts or failures
#   bash exp4_sherlock.sh status
# Settings below can be overridden from the environment, e.g. PARTITION=normal,owners bash exp4_sherlock.sh submit
set -euo pipefail

REPO_DIR=${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
DATA_DIR=${DATA_DIR:-${SCRATCH:+$SCRATCH/CircadianPOMDP_Data}}   # repo Data/ is a symlink here; job I/O belongs on $SCRATCH
PARTITION=${PARTITION:-normal}          # "normal,owners" if your group owns nodes (owners jobs can be preempted)
CPUS=${CPUS:-8}                         # threads per job
MEM=${MEM:-16G}
POLICIES_PER_TASK=${POLICIES_PER_TASK:-10}
TASK_TIME=${TASK_TIME:-04:00:00}        # per policy task; up to ~2 h for 10 policies at 150 lux on sh02 nodes
MAX_PARALLEL=${MAX_PARALLEL:-30}        # array tasks running at once

# Must match the Julia side: get_intervention_policies() (300 policies), light_conditions_specs() names, and
# intervention_condition_dir(1000, …) = <RUN_PREFIX>_post<INTERVENTION_TARGET_DAYS[condition]>d, with RUN_PREFIX =
# Particles1000_<BASELINE_PST_DAYS>d_<JETLAG_ALIGNMENT_FILENAME_SUFFIX>_<INTERVENTION_WINDOW_TAG>_<TARGET_REFERENCE_TAG>_<CBTMIN_DEFINITION_TAG>
N_POLICIES=${N_POLICIES:-300}
CONDITIONS=(LowLight150lux HighLight10000lux)
RUN_PREFIX=${RUN_PREFIX:-Particles1000_70d_V4_1.341hEarlier_contigWindows_settledTarget_cbtStHilaire}
post_days() { case $1 in LowLight150lux) echo 100 ;; *) echo 50 ;; esac; }   # INTERVENTION_TARGET_DAYS
INPUTS=(   # relative to Data/
  JetLag/1_InitialParticles/initial_particles1000.jld2
  JetLag/2_RegularScheduleAnalysis/includedParticles_V4_1.341hEarlier.jld2
  JetLag/2_RegularScheduleAnalysis/LowLight150lux/particleHistories/particleHist_regular_70days_V4_1.341hEarlier_mean.jld2
  JetLag/2_RegularScheduleAnalysis/HighLight10000lux/particleHistories/particleHist_regular_70days_V4_1.341hEarlier_mean.jld2
)

SELF=$REPO_DIR/experiments/4_JetLagPolicy/sherlock/exp4_sherlock.sh
adapt_dir() { echo "$DATA_DIR/JetLag/4_JetLagIntervention/Adaptations/${RUN_PREFIX}_post$(post_days "$1")d/$1"; }
LOG_DIR=$DATA_DIR/JetLag/4_JetLagIntervention/sherlock_logs
N_CHUNKS=$(( (N_POLICIES + POLICIES_PER_TASK - 1) / POLICIES_PER_TASK ))
N_TASKS=$(( N_CHUNKS * ${#CONDITIONS[@]} ))

# ---- array task t -> condition CONDITIONS[t / N_CHUNKS], policies of chunk t % N_CHUNKS ----
task_condition() { echo "${CONDITIONS[$(( $1 / N_CHUNKS ))]}"; }
task_first() { echo $(( ($1 % N_CHUNKS) * POLICIES_PER_TASK + 1 )); }
task_last() { local l=$(( $(task_first "$1") + POLICIES_PER_TASK - 1 )); echo $(( l < N_POLICIES ? l : N_POLICIES )); }
task_done() {   # every policy file of task $1 exists
  local dir k
  dir=$(adapt_dir "$(task_condition "$1")")
  for (( k = $(task_first "$1"); k <= $(task_last "$1"); k++ )); do
    [[ -f $dir/policy_${k}_results_steadyState.jld2 ]] || return 1
  done
}
baseline_done() {
  local c
  for c in "${CONDITIONS[@]}"; do [[ -f $(adapt_dir "$c")/baseline_results_steadyState.jld2 ]] || return 1; done
}

# ---- environment checks ----
find_julia() {   # juliaup (or any julia on PATH) first, then the Sherlock module
  if ! command -v julia >/dev/null 2>&1 && type ml >/dev/null 2>&1; then
    set +u; ml julia >/dev/null 2>&1 || true; set -u   # Lmod isn't set -u safe
  fi
  JULIA_BIN=$(command -v julia || true)
  [[ -n $JULIA_BIN ]] || { echo "julia not found. Install the Manifest's version with juliaup:" >&2
                           echo "  curl -fsSL https://install.julialang.org | sh -s -- --yes --default-channel $(manifest_julia)" >&2
                           exit 1; }
  local have want
  have=$("$JULIA_BIN" --startup-file=no -e 'print(VERSION)')
  want=$(manifest_julia)
  if [[ ${have%.*} != "${want%.*}" ]]; then
    echo "$JULIA_BIN is Julia $have but Manifest.toml was resolved with $want. Install it with juliaup:" >&2
    echo "  curl -fsSL https://install.julialang.org | sh -s -- --yes --default-channel $want" >&2
    exit 1
  fi
  echo "Julia $have ($JULIA_BIN)"
}
manifest_julia() { sed -n 's/^julia_version = "\(.*\)"/\1/p' "$REPO_DIR/Manifest.toml"; }

check_data() {
  [[ -L $REPO_DIR/Data && $(readlink "$REPO_DIR/Data") == "$DATA_DIR" ]] ||
    { echo "$REPO_DIR/Data should be a symlink to $DATA_DIR; run: bash $SELF setup" >&2; exit 1; }
  local f missing=0
  for f in "${INPUTS[@]}"; do
    [[ -f $DATA_DIR/$f ]] || { echo "missing input: $DATA_DIR/$f" >&2; missing=1; }
  done
  (( missing == 0 )) || { echo "copy the inputs with sync_sherlock.sh push (on your laptop)" >&2; exit 1; }
}

# ---- subcommands ----
setup() {
  [[ -n $DATA_DIR ]] || { echo "\$SCRATCH is not set; set DATA_DIR" >&2; exit 1; }
  mkdir -p "$DATA_DIR" "$LOG_DIR"
  if [[ -L $REPO_DIR/Data ]]; then
    [[ $(readlink "$REPO_DIR/Data") == "$DATA_DIR" ]] ||
      { echo "$REPO_DIR/Data points to $(readlink "$REPO_DIR/Data"), not $DATA_DIR" >&2; exit 1; }
  elif [[ -e $REPO_DIR/Data ]]; then
    echo "$REPO_DIR/Data is a real directory; move its contents to $DATA_DIR and remove it, or set DATA_DIR" >&2
    exit 1
  else
    ln -s "$DATA_DIR" "$REPO_DIR/Data"
    echo "linked $REPO_DIR/Data -> $DATA_DIR"
  fi
  check_data
  find_julia
  # Download packages on the login node; compute nodes precompile in the baseline job
  JULIA_PKG_PRECOMPILE_AUTO=0 "$JULIA_BIN" --project="$REPO_DIR" -e 'using Pkg; Pkg.instantiate()'
  echo "setup done; next: bash $SELF test"
}

submit() {   # $1 = all | test
  local mode=$1
  check_data
  find_julia
  mkdir -p "$LOG_DIR"
  local common=(--parsable --partition="$PARTITION" --chdir="$REPO_DIR" --kill-on-invalid-dep=yes --requeue
                --export=ALL,REPO_DIR="$REPO_DIR",DATA_DIR="$DATA_DIR",JULIA_BIN="$JULIA_BIN",POLICIES_PER_TASK="$POLICIES_PER_TASK",N_POLICIES="$N_POLICIES",RUN_PREFIX="$RUN_PREFIX")
  local dep=() id

  if baseline_done; then
    echo "baseline: done"
  else
    id=$(sbatch "${common[@]}" --job-name=exp4-baseline --cpus-per-task="$CPUS" --mem="$MEM" --time=02:00:00 \
                --output="$LOG_DIR/baseline_%j.log" "$SELF" _job baseline)
    dep=(--dependency=afterok:"$id")
    echo "baseline: job $id"
  fi

  local todo=() t
  for (( t = 0; t < N_TASKS; t++ )); do task_done "$t" || todo+=("$t"); done
  [[ $mode == test ]] && todo=(${todo[@]+"${todo[@]:0:1}"})   # empty when every task is done (set -u safe)
  if (( ${#todo[@]} > 0 )); then
    local spec
    spec=$(IFS=,; echo "${todo[*]}")
    id=$(sbatch "${common[@]}" ${dep[@]+"${dep[@]}"} --job-name=exp4-policies --cpus-per-task="$CPUS" --mem="$MEM" \
                --time="$TASK_TIME" --array="$spec%$MAX_PARALLEL" --output="$LOG_DIR/policies_%A_%a.log" \
                "$SELF" _job policies)
    dep=(--dependency=afterok:"$id")
    echo "policies: array job $id, ${#todo[@]} of $N_TASKS tasks ($POLICIES_PER_TASK policies each, $CPUS CPUs, $TASK_TIME)"
  else
    echo "policies: all $N_TASKS tasks done"
  fi

  if [[ $mode == test ]]; then
    echo "test: when the policy task finishes, check the '⏱' lines in $LOG_DIR/policies_*.log, set TASK_TIME, then run: bash $SELF submit"
    return
  fi
  id=$(sbatch "${common[@]}" ${dep[@]+"${dep[@]}"} --job-name=exp4-aggregate --cpus-per-task=1 --mem=16G --time=01:00:00 \
              --output="$LOG_DIR/aggregate_%j.log" "$SELF" _job aggregate)
  echo "aggregate: job $id"
  echo "watch with: bash $SELF status"
}

status() {
  squeue -u "$USER" -n exp4-baseline,exp4-policies,exp4-aggregate -o "%.20i %.15j %.9T %.10M %.10l %R" || true
  local c n t todo=0
  for c in "${CONDITIONS[@]}"; do
    d=$(adapt_dir "$c")
    n=$(find "$d" -maxdepth 1 -name 'policy_*_results_steadyState.jld2' 2>/dev/null | wc -l)
    printf '%-18s baseline %-4s policies %3d / %d   final %s\n' "$c" \
      "$([[ -f $d/baseline_results_steadyState.jld2 ]] && echo yes || echo no)" "$n" "$N_POLICIES" \
      "$([[ -f $d/systematic_rescue_analysis_steadyState.jld2 ]] && echo yes || echo no)"
  done
  for (( t = 0; t < N_TASKS; t++ )); do task_done "$t" || todo=$(( todo + 1 )); done
  echo "array tasks with missing policies: $todo of $N_TASKS"
  echo "logs: $LOG_DIR"
}

run_job() {   # inside Slurm; $1 = baseline | policies | aggregate
  local stage=$1
  cd "$REPO_DIR"
  export RESCUE_STAGE=$stage JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} GKSwstype=100
  if [[ $stage == policies ]]; then
    local t=$SLURM_ARRAY_TASK_ID
    export RESCUE_CONDITION RESCUE_POLICIES
    RESCUE_CONDITION=$(task_condition "$t")
    RESCUE_POLICIES="$(task_first "$t"):$(task_last "$t")"
  fi
  echo "[$(date '+%F %T')] $stage on $(hostname) ($(nproc) CPUs) ${RESCUE_CONDITION:-} ${RESCUE_POLICIES:-}"
  [[ $stage == baseline ]] && "$JULIA_BIN" --project=. -e 'using Pkg; Pkg.precompile()'
  "$JULIA_BIN" --project=. -t "$JULIA_NUM_THREADS" experiments/4_JetLagPolicy/rescuingJetLag_unified.jl
  echo "[$(date '+%F %T')] $stage finished"
}

case ${1:-} in
  setup) setup ;;
  test) submit test ;;
  submit) submit all ;;
  status) status ;;
  inputs) printf '%s\n' "${INPUTS[@]}" ;;   # used by sync_sherlock.sh
  _job) run_job "$2" ;;
  *) sed -n '2,20p' "$0"; exit 1 ;;
esac
