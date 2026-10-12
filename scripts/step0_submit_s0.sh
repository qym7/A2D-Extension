#!/usr/bin/env bash
# submit_s0.sh [smoke|full]
# Idea-1 S0 baseline gate: reproduce A2D Table 2 (MATH-500, Dream-v0 family) with the verified
# lockstep pipeline, to confirm env + checkpoint identity before any new measurement is trusted.
#
#   Table 2 (A2D):  schedule\tokens | Parent | A2D      Expected cells, n=200:
#                   Parent          |  38.4  | 42.8       free/parent      -> 38.4
#                   A2D             |  42.2  | 45.0       content(P,A2D)   -> 42.8
#                                                         schedule(P,A2D)  -> 42.2
#                                                         free/A2D         -> 45.0
# Checkpoint identity verified from merge_manifest.json on 2026-09-27:
#   dream-ins-04a_D-06a_A = D_b + 0.4*tau_D + 0.6*tau_A  (the paper's selected lambda=0.6)
set -euo pipefail

STAGE=${1:-smoke}
CSUB=/mnt/lts4/scratch/home/yqin/rcp_cloned/mlo/csub.py
USER_YAML=/mnt/lts4/scratch/home/yqin/rcp_cloned/mlo/user.yaml
IMG=registry.rcp.epfl.ch/lts4-mmonteir/mlo
RUN=/mnt/lts4/scratch/home/yqin/DllmAR/new_ideas/idea1_selective_distill/commands/run_lockstep_m500.sh

PARENT=Dream-org/Dream-v0-Instruct-7B
A2D=/mnt/lts4/scratch/home/yqin/DllmAR/dream_study/step2_instruct/outputs/checkpoints/dream-ins-04a_D-06a_A

submit () {  # submit <jobname> <driver> <other> <mode> <tag> <n>
  local job=$1 drv=$2 oth=$3 mode=$4 tag=$5 n=$6
  echo "=== submitting $job"
  python "$CSUB" -u "$USER_YAML" -n "$job" -g 1 --node_type b300 -i "$IMG" --train \
    --command="set -euo pipefail; bash $RUN '$drv' '$oth' '$mode' '$tag' '$n'"
}

if [ "$STAGE" = smoke ]; then
  # Plumbing only. n=20 does NOT test the Table-2 values; it tests that the job runs end to end.
  submit yqin-i1-s0-smk-m500-free-b300 "$PARENT" none free s0-smk-free 20
else
  submit yqin-i1-s0-m500-free-b300    "$PARENT" none     free     s0-free-parent 200
  submit yqin-i1-s0-m500-content-b300 "$PARENT" "$A2D"   content  s0-content     200
  submit yqin-i1-s0-m500-sched-b300   "$PARENT" "$A2D"   schedule s0-schedule    200
  submit yqin-i1-s0-m500-a2d-b300     "$A2D"    none     free     s0-free-a2d    200
fi
