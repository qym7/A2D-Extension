# STEP3 pipeline — frontier over-distillation, end to end (Snakemake)

What it does: distil the decode-time gate (student keeps its token where confident, an *over-merged* teacher supplies the token where
it is not) into **one** Dream-Instruct checkpoint that decodes at 1× with the standard sampler. Plan, loss and reading rules:
`../technical_details/step3_plan_frontier_kd.md`; the evidence that the loss positions are the right ones: `../technical_details/step3_prelim_signal_check.md`.

```
prompts.jsonl (17k, label-free)                     code/step3/build_prompts.py          login node, memory-capped
   └─> rollouts/s{0..3}/rollouts.jsonl               modules/run_step3_rollout.sh            4 GPU jobs (student decodes once, official sampler)
         └─> rollouts/all.jsonl
teachers  outputs/checkpoints/add-a10-b{10,15}       modules/run_build_addition.sh           A(β) = D_b + τ_D + β·τ_A, built once
   └─> train  outputs/checkpoints/step3-{arm}/merged  modules/run_step3_train.sh              1 GPU job per arm (LoRA, teacher frozen alongside)
         ├─> MATH-500 last 300, lockstep @1024, 3 shards  modules/run_gated_shard5.sh (host=merged, donor none, free)
         │       └─> outputs/stepN/step3-{arm}/math500/free-last300-merged/summary.json     code/step3/merge_math_shards.py
         └─> guards (only arms in config `guard_arms`)   modules/run_official_eval.sh → dream_study official ifeval.sh / coding.sh / math_instruct.sh
technical_details/step3_results_table.md             code/step3/results_table.py
```

## Run
```bash
cd /mnt/lts4/scratch/home/yqin/DllmAR/new_ideas/idea1_selective_distill/code/step3/pipeline
export PYTHONPATH=/mnt/lts4/scratch/home/yqin/DllmAR/new_ideas/modules/pydeps     # snakemake 7.32 lives here (isolated; no conda env was touched)
SM=/mnt/lts4/scratch/home/yqin/conda/envs/mdlm-math-b300/bin/python\ -m\ snakemake
$SM -n first_wave                 # dry run: shows the jobs that would be submitted
$SM -j 6 --keep-going first_wave  # M + C2, MATH only  (≈ 75 GPU·h at H200·0.51, ~4 days wall)
$SM -j 6 --keep-going all         # every arm in config.yaml + guards for `guard_arms`
```
Snakemake only orchestrates on the login node. Each GPU rule calls `modules/submit_wait.sh`, which submits through `modules/submit.sh`
(the project's quoting-safe csub wrapper, H200 × 0.51), polls `runai list` every 2 min, and succeeds only if the job *Succeeded* and the
expected output exists. Every submission is appended to `../../tmp.md` (the experiment queue). Re-running after a failure continues from the
last finished output; nothing is resubmitted if its output exists. `-j` bounds the number of concurrent RunAI jobs.

## Arms (config.yaml)
| arm | corruption | loss positions | target | question |
|---|---|---|---|---|
| M | suffix cut | frontier (first 4 masks) where student p < 0.8 | KL to A(1.5) | the method |
| C2 | suffix cut | all frontier | CE on the student's own rollout token | is the teacher needed (self-SFT) |
| C1 | suffix cut | all frontier | KL to A(1.5) | is the θ-selection needed |
| C3 | suffix cut | frontier, p < 0.8 | KL to A(1.0) (not over) | is "over" needed |
| C4 | random mask | masked, p < 0.8 | KL to A(1.5) | is the suffix corruption needed |

## Before the first real run (gates from new_ideas/CLAUDE.md)
1. `python -m py_compile code/step3/*.py`, `bash -n modules/run_step3_*.sh modules/submit_wait.sh modules/run_official_eval.sh`, `snakemake -n`.
2. Rollout smoke: 50 prompts, batch 4 — confirms batched left-padded decoding and gives s/prompt for the shard count.
3. Train smoke: `--max-steps 20 --n-canvases 200` on arm M — loss decreasing, `sel_rate` ≈ 0.5 (prelim: ~55 % of frontier states have a
   low-confidence position), merged checkpoint decodes 20 MATH problems non-degenerately.
4. Only then `first_wave`. Guards for an arm are enabled by adding it to `guard_arms` after its MATH number is read (plan §4 budget rule).

## Status (2026-10-15)
Code written, **nothing smoked yet**: `train_frontier_kd.py` is a draft (single-GPU, LoRA via peft, teacher frozen alongside, micro-batch 1 state);
`run_official_eval.sh` score extraction for coding/gsm8k reads the official JSONs by pattern and may need one adjustment on the first run.
