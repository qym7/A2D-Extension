# Idea 1 — TASK (current gate and next actions; updated 2026-10-15)

**Current gate:** 1k test done on the full last 300 (2026-10-16): M(mixpool-1k) 50.3 / C2 36.3 / Ins 36.7 (`step3_distillation.md`). The tulu3-1k (Tulu-3-only) attempt was INVALID (wrong pool, 119 canvases; `trash/n1k_invalid_20261016/`) — pipeline fixed, relaunch needs user go (15 jobs ≈ 20 GPU·h). Also pending go: guards (IFEval-150 + HumanEval+) on Ins / M(.4,.6) / A(2.0) / M(mixpool-1k).

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
