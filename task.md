# Idea 1 — TASK (current gate and next actions; updated 2026-10-15)

**Current gate:** STEP3 smoke jobs — not yet submitted (user approval needed for every job).

## Where things stand
- STEP0–2 closed: gate established and explained (`step1_methods.md`); decode-time multi-model work closed (`step2_verdict.md`).
- STEP3 code written, nothing run on GPU yet: `code/step3/` + `code/step3/pipeline/` (Snakemake, dry-run OK: first wave = 16 jobs).
- Prompt set built: `outputs/step3/prompts.jsonl` (17 k).

## Next actions, in order
1. Rollout smoke: 50 prompts, batch 4, H200·0.51 (`modules/run_step3_rollout.sh`) — checks batched left-padded decoding, gives s/prompt.
2. Training smoke: arm M, `--max-steps 20 --n-canvases 200` — loss decreasing, `sel_rate` ≈ 0.5, merged checkpoint decodes 20 MATH problems.
3. `snakemake first_wave` (M + C2, MATH last 300 only; ≈ 75 GPU·h).
4. Read by the rules in `technical_details/step3_plan_frontier_kd.md` §4; enable guards (`guard_arms`) for arms that clear the MATH bar; then C1/C3/C4.

## Done criteria for STEP3
1× student ≥ Ins + 6 on MATH last 300, > C2 + 3, no loss on GSM8K / HumanEval+ / IFEval. Kill: C2 matches, or a guard regresses after one retry.

Verification commands: `python -m py_compile code/step*/*.py`; `bash -n ../modules/run_step3_*.sh`; `cd code/step3/pipeline && PYTHONPATH=../../../../modules/pydeps python -m snakemake -n first_wave`.
