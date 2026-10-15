# Idea 1 — TASK

**Current gate:** `S0 baseline gate` — not yet submitted.
Status legend: `todo` / `running` / `passed` / `FAILED` / `killed`.

## Why this idea is cheap: the pipeline already exists and is protocol-verified

`../../dream_study/step2_instruct/scripts/lockstep_port.py` (230 lines) already runs **two
Dream-architecture models in lockstep on one shared canvas**, with:
- a decode loop that is a line-by-line copy of official Dream `_sample` (entropy remasking, linear
  schedule, top-p applied to logits *before* the confidence, bf16 confidence top-k), verified token
  for token against `model.diffusion_generate(..., temperature=0)` via `--check-official K`;
- `--mode content` = driver's positions, **other's tokens** where the argmax differs;
  `--mode schedule` = other's positions, driver's tokens;
- MATH-500 and HumanEval prompt loading + official scoring;
- a job wrapper with the pinned protocol: `scripts/jobwrap/lockstep_port.sh`.

A2D Table 2 was produced with this script, which gives us **known confirmation values** (MATH-500,
Dream-v0 family): parent 38.4 · parent-schedule+A2D-tokens 42.8 · A2D-schedule+parent-tokens 42.2 ·
A2D both 45.0. Any modified copy must reproduce these before it is trusted.

E1 needs exactly one new thing: **gate the port on the driver's confidence** (`--mode
content_gated --theta θ`), plus a substitution counter. ~10 lines.

## Assets confirmed present (2026-09-27)

| Role | Path / id |
|---|---|
| host A (diffusion parent `D_p`) | `Dream-org/Dream-v0-Instruct-7B` (shared HF store) |
| host B (A2D merge, λ=0.6, IFEval-selected) | `../../dream_study/step2_instruct/outputs/checkpoints/dream-ins-04a_D-06a_A` |
| donor at selected α=0.9 | `../../dream_study/step2_instruct/outputs/checkpoints/dream-ins-00aD-09aA` |
| donor past-peak α∈{1.2,1.5} | **to build** with `../../dream_study/step2_instruct/scripts/build_task_vector_merge.py` |
| MATH-500 prompts (chat-templated, lm_eval samples) | `../../dream_study/step2_instruct/outputs/official_eval/mathins_dreamins_m500/Dream-org__Dream-v0-Instruct-7B/samples_math_500_2026-09-01T13-03-23.836318.jsonl` |
| HumanEval prompts (all 164, EvalPlus dllm provider) | `../../dream_study/step2_instruct/outputs/arness/humaneval_prompts_diffucoder_all.jsonl` |
| env | `mdlm-math` for MATH-500 (minerva scorer needs antlr4==4.11), `mdlm-prime` for HumanEval |

## Protocol checklist (GPU-cost preflight gate — must be complete before any `csub.py`)

| Field | Value | Source |
|---|---|---|
| model role | driver = host, other = donor; both Dream-v0 arch, bf16 | A2D §4.1 / `lockstep_port.py` |
| dataset + split | MATH-500, our lm_eval samples jsonl, **first n=200** | A2D §2 footnote 1 ("first 200 samples for the analysis") |
| prompt / template | Dream-v0-Instruct chat template, already baked into the samples jsonl | `lockstep_port.sh` |
| backend / decoder | official Dream `_sample`, entropy remasking, linear schedule | `lockstep_port.py` docstring |
| steps / max_new | 512 / 512 | `lockstep_port.sh` math500 branch |
| temperature | token = **argmax** (deterministic); confidence/order computed at `conf_temp=0.1` | `lockstep_port.py` |
| top-p | 0.9 (applied to logits before confidence) | `lockstep_port.sh` |
| eps | 1e-3 (Dream `generation_config`) | `lockstep_port.sh` |
| sanitizer / scorer | lm_eval diffllm text post-processing + fork's `minerva_math process_results` | `lockstep_port.py` |
| expected baseline | free/parent = **38.4**, content-port = **42.8**, schedule-port = **42.2**, A2D = **45.0** (MATH-500) | A2D Table 2 |
| hardware | B300 (free until 2026-09-30); job names end `-b300` | workspace CLAUDE.md |
| memory | 2 × 7B bf16 resident ≈ 28 GB + activations → 1 full GPU | — |

## Steps

### S0 — baseline gate (`todo`)
Copy `lockstep_port.py` to `code/step1/lockstep_gated.py` **unmodified**, run the four
Table-2 cells at n=200 on B300, and confirm each within the noise band of the published value.
Do not touch the gate code until S0 passes. *(Note: Table 2's cells are the parent↔A2D-merge pair,
so S0 also confirms the checkpoints on disk are the paper's.)*
Verification: `runai logs`, then the four scores in `results/s0_table2_repro.md`.
**If S0 fails, stop and diagnose — it is an environment/checkpoint problem, not a research result.**

### S1 — build past-peak donors (`todo`)
`build_task_vector_merge.py --anchor Dream-org/Dream-v0-Base-7B --term Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-7B <α>`
for α ∈ {1.2, 1.5}. Run `check_ckpt_lineage.py` on each and keep the JSON in `details/`.
Sanity: eval each donor *standalone* (`--mode free`) on MATH-500 n=200 to locate where it has
collapsed. H1 requires the donors to be visibly worse standalone than α=0.9 — if they are not,
they are not "past peak" and S1 must extend α.

### S2 — gated content-port sweep (`todo`, the decisive run)
Add `--mode content_gated --theta θ` + substitution counter to the copy. Grid:
`α ∈ {0.9, 1.2, 1.5} × θ ∈ {0, 0.3, 0.5, 0.7, 0.9, 1.0}`, host = Dream-v0-Instruct, MATH-500 n=200.
θ=0 must reproduce S0's free/parent number exactly (plumbing check inside the sweep).
Record for every cell: score, **substitution rate**, cap rate, position/token disagreement.
Then repeat the single best cell with host = `dream-ins-04a_D-06a_A`.

### S3 — decision (`todo`)
Apply SPEC.md's three-part rule. Write `results/README.md` either way. If it passes, the held-out
check is IFEval (same family, A2D's own anchor) at the frozen `(α, θ)` — **choose the benchmark
before looking at S2**.

### S4 — E2 distillation (`blocked` on S3 + the novelty check in ANALYSIS.md)

## Done criteria

- `results/s0_table2_repro.md` reproduces A2D Table 2 within noise (B300-labelled).
- `results/s2_grid.md` holds the full (α, θ) grid **with substitution rates**.
- `results/README.md` states one of: `method result accepted` / `killed, reason`.
- `commands/` holds only the submitters/runners that produced accepted numbers.

## Verification commands

```bash
python -m py_compile code/step1/lockstep_gated.py
bash -n commands/*.sh
# θ=0 must equal free mode; assert in the sweep aggregator
```
