#!/usr/bin/env bash
# run_lockstep_m500.sh <driver> <other|none> <mode> <tag> [n] [extra...]
# Idea-1 runner. Calls the UNMODIFIED, protocol-verified lockstep_port.py from dream_study
# (line-by-line copy of official Dream `_sample`, checked against diffusion_generate(T=0)).
# Only the output directory differs from dream_study/step2_instruct/scripts/jobwrap/lockstep_port.sh:
# idea-1 results go under new_ideas/, never into the A2D project's outputs.
# Protocol (MATH-500 branch, pinned to that jobwrap): steps 512, top-p 0.9, conf_temp 0.1, eps 1e-3,
# token = argmax (deterministic), prompts = our Dream-v0-Instruct MATH-500 lm_eval samples jsonl.
set -euo pipefail

DRV=${1:?driver}; OTH=${2:?other|none}; MODE=${3:?mode}; TAG=${4:?tag}; N=${5:-200}
shift 5 || true
EXTRA=("$@")

DS=/mnt/lts4/scratch/home/yqin/DllmAR/dream_study
IDEA=/mnt/lts4/scratch/home/yqin/DllmAR/new_ideas/idea1_selective_distill
SCRIPT=$DS/step2_instruct/scripts/lockstep_port.py

source /mnt/lts4/scratch/home/yqin/conda/etc/profile.d/conda.sh
conda activate mdlm-math
export HF_HOME=$DS/hf_cache PYTHONNOUSERSITE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH=/mnt/lts4/scratch/home/yqin/conda/envs/mdlm-math/lib/python3.9/site-packages/nvidia/nvjitlink/lib:${LD_LIBRARY_PATH:-}

PR=$(ls "$DS"/step2_instruct/outputs/official_eval/mathins_dreamins_m500/Dream-org__Dream-v0-Instruct-7B/samples_math_500_*.jsonl | head -1)
[ -s "$PR" ] || { echo "[FATAL] prompts jsonl not found"; exit 2; }

OUT=$IDEA/outputs/step0/lockstep/math500/$TAG
mkdir -p "$OUT"

OARG=()
[ "$OTH" != none ] && OARG=(--other "$OTH")

echo "[run] driver=$DRV other=$OTH mode=$MODE n=$N out=$OUT"
echo "[run] prompts=$PR"
python "$SCRIPT" --benchmark math500 --prompts "$PR" \
  --driver "$DRV" "${OARG[@]+"${OARG[@]}"}" --mode "$MODE" \
  --n "$N" --steps 512 --top-p 0.9 --conf-temp 0.1 --eps 1e-3 \
  --out-dir "$OUT" "${EXTRA[@]+"${EXTRA[@]}"}" 2>&1 | tee -a "$OUT/run.log"
echo "[run] done -> $OUT"
