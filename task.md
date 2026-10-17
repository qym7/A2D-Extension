# Idea 1 — TASK (current gate and next actions; updated 2026-10-15)

**Current gate:** 1k test done on the full last 300 (2026-10-16): M(mixpool-1k) 50.3 / C2 36.3 / Ins 36.7, and the generalisation table is in (`step3_distillation.md` §Generalisation: IFEval rand100 M 64 / Ins 62 / M(.4,.6) 65 / A(2.0) 51; HumanEval+ 60.4 / 57.3 / 61.0 / 56.7). Gate row guards in (IFEval 70 / HumanEval+ 59.8). **Rule 2026-10-17: all training variants are tuned on the validation set only** (math = MATH test minus MATH-500 rand100, IFEval val100, MBPP+ val100; `modules/submit_step3_recipe.sh`); running tonight: ep4 / half500 / θ 0.6 / θ 0.4 + Ins, M references. AR-teacher baseline (Qwen2.5-7B-Instruct, prefix mask, same data/budget) in: MATH 46.7 / IFEval 52 / HE+ 57.3 — the diffusion over-teacher wins on both axes. The tulu3-1k (Tulu-3-only) attempt was INVALID (wrong pool; `trash/n1k_invalid_20261016/`) — pipeline fixed; relaunch and the recipe search (TODO 0) are Amel's.

## Where things stand
- STEP0–2 closed: gate established and explained (`step1_methods.md`); decode-time multi-model work closed (`step2_verdict.md`).
- STEP3 code written, nothing run on GPU yet: `code/step3/` + `code/step3/pipeline/` (Snakemake, dry-run OK: first wave = 16 jobs).
- Prompt set built: `outputs/step3/prompts.jsonl` (17 k).

## Next actions, in order (full TODO with equations: `step3_distillation.md` "TODO", @Amel)
1. DONE rollout smokes (m46 50 prompts, Ins 200 prompts): 19–22 s/prompt regardless of batch → batch 1; commit_conf <0.8 ≈ 24 %.
2. DONE training smoke (arm M, Ins ⇐ A(2.0), 200 canvases, 20 steps): micro-batch 4 (69.7 GB), 0.23 s/frame ≈ 6.5 GPU·h per arm, kd 0.61 → 0.3, merged checkpoint decodes.
3. `snakemake first_wave` (M + M06 + C2, MATH last 300 only): Ins rollout 17k ≈ 92 GPU·h (8k ≈ 43) + 3 × 6.5 training + 3 × 7 eval ≈ 130 (or 80) GPU·h.
4. Read by the rules in `technical_details/step3_plan_frontier_kd.md` §4; enable guards (`guard_arms`) for arms that clear the MATH bar; then C1/C3/C4.

## Done criteria for STEP3
1× student ≥ Ins + 6 on MATH last 300, > C2 + 3, no loss on GSM8K / HumanEval+ / IFEval. Kill: C2 matches, or a guard regresses after one retry.

Verification commands: `python -m py_compile code/step*/*.py`; `bash -n ../modules/run_step3_*.sh`; `cd code/step3/pipeline && PYTHONPATH=../../../../modules/pydeps python -m snakemake -n first_wave`.
