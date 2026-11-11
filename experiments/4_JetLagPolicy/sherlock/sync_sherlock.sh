#!/bin/bash
# Copy the code and Exp 4 inputs to Sherlock, or fetch the results. Run on your laptop.
#   SUNETID=<you> bash experiments/4_JetLagPolicy/sherlock/sync_sherlock.sh push         # code + inputs
#   SUNETID=<you> bash experiments/4_JetLagPolicy/sherlock/sync_sherlock.sh push-inputs  # Manifest.toml + inputs only
#                                                    (code from a git clone of the repo in $HOME/CircadianPOMDP)
#   SUNETID=<you> bash experiments/4_JetLagPolicy/sherlock/sync_sherlock.sh pull         # results + logs (small)
#   SUNETID=<you> bash experiments/4_JetLagPolicy/sherlock/sync_sherlock.sh pull-events  # per-scenario CBTmin events (~7 GB)
# Code goes to $HOME/CircadianPOMDP through the login node. Data goes to $SCRATCH/CircadianPOMDP_Data through the
# data-transfer node (dtn.sherlock.stanford.edu), where relative paths start in $SCRATCH.
# Each new ssh connection asks for Duo; an ssh ControlMaster for these hosts avoids repeated prompts.
# The plot scripts only need `pull`.
set -euo pipefail
: "${SUNETID:?set SUNETID to your Stanford username}"
REPO_DIR=$(cd "$(dirname "$0")/../../.." && pwd)
LOGIN=$SUNETID@login.sherlock.stanford.edu
DTN=$SUNETID@dtn.sherlock.stanford.edu
REMOTE_CODE=${REMOTE_CODE:-CircadianPOMDP}         # relative to $HOME on the login node
REMOTE_DATA=${REMOTE_DATA:-CircadianPOMDP_Data}    # relative to $SCRATCH on the DTN
RUN_PREFIX=${RUN_PREFIX:-Particles1000_70d_V4_1.341hEarlier_contigWindows_settledTarget_cbtStHilaire}   # + _post<days>d per condition
CODE_PATHS=(Project.toml src experiments/jetlag_reference.jl experiments/revision_baseline.jl experiments/4_JetLagPolicy)
VERSION_FILE=experiments/4_JetLagPolicy/sherlock/code_version.txt   # gitignored
INTERVENTION=JetLag/4_JetLagIntervention

cd "$REPO_DIR"
case ${1:-} in
  push)
    # Tracked and new (not ignored) files, the gitignored Manifest.toml (exact package versions), and a record of
    # exactly which code is running
    { git rev-parse HEAD; git status --porcelain -- "${CODE_PATHS[@]}"; shasum -a 256 Manifest.toml; } > "$VERSION_FILE"
    { git ls-files -co --exclude-standard -- "${CODE_PATHS[@]}"; echo Manifest.toml; echo "$VERSION_FILE"; } |
      rsync -av --files-from=- ./ "$LOGIN:$REMOTE_CODE/"
    bash experiments/4_JetLagPolicy/sherlock/exp4_sherlock.sh inputs |
      rsync -av --files-from=- Data/ "$DTN:$REMOTE_DATA/"
    ;;
  push-inputs)
    rsync -av Manifest.toml "$LOGIN:$REMOTE_CODE/"   # gitignored, but pins the package versions used locally
    bash experiments/4_JetLagPolicy/sherlock/exp4_sherlock.sh inputs |
      rsync -av --files-from=- Data/ "$DTN:$REMOTE_DATA/"
    ;;
  pull)
    mkdir -p "Data/$INTERVENTION/Adaptations"
    for days in 100 50; do   # INTERVENTION_TARGET_DAYS: 150 lux, 10,000 lux
      rsync -av "$DTN:$REMOTE_DATA/$INTERVENTION/Adaptations/${RUN_PREFIX}_post${days}d" "Data/$INTERVENTION/Adaptations/"
    done
    rsync -av "$DTN:$REMOTE_DATA/$INTERVENTION/sherlock_logs" "Data/$INTERVENTION/"
    ;;
  pull-events)
    rsync -av "$DTN:$REMOTE_DATA/$INTERVENTION/particleHistories_cbtStHilaire" "Data/$INTERVENTION/"
    ;;
  *) sed -n '2,12p' "$0"; exit 1 ;;
esac
