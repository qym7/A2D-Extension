# STEP2 — Multi-model and logit-space combinations, and what they say about quick distillation (2026-10-15)

STEP1 established the gate (student decodes, the over teacher supplies the token at the student's low-confidence steps) and its mechanism.
STEP2 asks the next two questions and answers them with the same protocol (MATH-500 **last 300**, lockstep @1024, H200·0.51; all numbers
`technical_details/step2_*.md`, the full tables of the 10-12 state in `technical_details/step2_verdict_20261012_archive.md`):

1. **Is there more to get from several models at decode time** — in token space, in probability/logit space, across teacher lines, with AR?
2. **What does that imply for a *quick* distillation** — which positions, which target, which teacher, which data — so that STEP3 trains one 1× model.

## 1. Reference points (last 300)
| system | acc | finish | cost |
|---|---:|---:|---:|
| Ins (student) | 36.7 | 76 % | 1× |
| A(2.0) over addition (teacher) | 46.7 | 94 % | 1× |
| M(0,1) over merge (teacher, other line) | 44.0 | 74 % | 1× |
| gate Ins ⇐ A(1.5) @0.8 / Ins ⇐ A(2.0) @0.8 | 48.3 / 49.0 | 95 % / 92 % | 1.14× (lazy teacher call) |
| gate M(.4,.6) ⇐ A(2.0) @0.8 (best number, two lines changed at once) | 51.0 | 89 % | 1.14× |
| AR Qwen2.5-7B-Instruct, strict / lenient | 55.3 / 70.7 | 78 % | 1× AR |

## 2. Everything we combined (`technical_details/step2_plan_ladder_softmix.md`, `step2_cascade_two_lines.md`, `step1_methods.md` §1–2)
| combination | space | rule at a low-confidence step | last 300 | reading |
|---|---|---|---:|---|
| teacher dose β (Ins ⇐ A(β) @0.8) | weights of the teacher | one teacher, β = 1.0 / 1.5 / 2.0 / 2.5 / 3.0 | 44.3 / 48.3 / 49.0 / 47.0 / collapsed | teacher must be *past* the peak; plateau from 1.5; rate 2 grows 1.75 → 6.6 % with no gain → **rule: smallest β past the peak** |
| weight-space midpoint (0.7,1.3) alone | weights | no gate, 1× | 44.3 | the mixture is not a point in weight space |
| soft mix, Ins + A(2.0) | probability | argmax of (p_S + p_T)/2 | 46.3 | lowers the dose (rate 2 1.3 % vs 5 %), lands between student and gate; θ=1.0 ≡ θ=0.8 → gate is only a compute saver here; **hard replacement stays** |
| teacher mean, A(1.5)/A(2.0)/A(2.5) | probability | argmax of the teachers' mean | not run (killed by prior) | the three singles are a plateau whose correct sets overlap like two θ of one arm (Jaccard 0.63–0.67); majority vote of their *answers* = 48.3 = single |
| most-confident of {A(2.0), M(0,1)} | probability (confidence) | the more confident teacher's token | 48.7 | = single teacher; winners M(0,1) 61 % / A(2.0) 39 %, agree 67 % |
| Ins ⇐ M(0,1) @0.8 | token, other line | one teacher | 45.7 | inherits M(0,1)'s finish problem (76 %); the Ins student gains only from the addition line |
| abstention cascade (precision-ordered, unfinished = wrong) | answer level | M(0,1) → Ins ⇐ A(2.0) | 53.0 (1.5×); 55–57 at 2.5× | the only decode-time gain from two models; needs whole sequences, not a token-level signal |
| AR + diffusion | answer level | AR first, diffusion on AR's unboxed | 66.7–68.3 | < AR lenient 70.7; AR-wrong & diffusion-right is 23 real problems; no combination here beats AR |

**One sentence:** every token-level or probability-level way of adding a second teacher equals one over teacher; the gate's gain is a single
clean signal — *which positions* (the student's low-confidence steps) and *which token* (the over teacher's) — and that is exactly what a
distillation loss needs.

## 3. What this fixes for the distillation signal
| design choice | decided | because |
|---|---|---|
| teacher | **one** over teacher on the student's own line, A(1.5) (`add-a10-b15`); A(2.0) equivalent | no multi-teacher gain; β rule; other line hurts the Ins student |
| target | the teacher's **full distribution** (forward KL), not its argmax | hard replacement is what works at decode time, but at training time the teacher's off-argmax mass is free information the gate never used; the hard-token version is control C2's shape |
| positions | **only** where the student's raw p < 0.8 | random positions at the gate's dose = student (STEP1); these positions are 10 % of steps, 3–5 % of tokens — the loss is sparse by design |
| training state | **prefix visible, suffix masked** (a cut of the student's own rollout), loss on the first K ≤ 4 masks | `technical_details/step3_prelim_signal_check.md`: random masking finds 11–19 % of the gate's positions, the suffix frontier 96–98 % at 0.87 precision; the three A(β) agree there 77 % and the agreed token is the gate's 90 % |
| data | **prompts only, no labels, no AR** | the teacher supplies the target; AR cannot (no mask semantics, and the user wants a diffusion model); Tulu-3 (Dream-Instruct's own SFT mixture, incl. its math subsets) + MATH train + code + chat |
| adaptation | **short**: LoRA r64, 17 k prompts, 2 epochs | the signal is a few % of tokens; the goal is to fold +12 points into 1×, not to re-train |
| controls | self-SFT on the same states (C2), all-frontier KD (C1), non-over teacher A(1.0) (C3), random-mask corruption (C4) | each removes one of the four decisions above |

## 4. Verdict → STEP3
- Decode-time multi-model work is **closed**: the ceiling is one teacher (≈ 49), the cascade buys +4–6 at 1.5–2.5×, AR stays 20 points above under lenient scoring.
- The gate is worth keeping as a **training signal**: positions and targets are reproducible at training time without labels, from one extra forward of a frozen teacher.
- STEP3 = the quick distillation in §3, run as `pipeline/` (Snakemake: prompts → rollouts → LoRA arms → MATH last 300 → official guards).
  Success: the 1× student ≥ Ins + 6 on MATH last 300 and > self-SFT + 3, with no loss on GSM8K / HumanEval+ / IFEval. Kill: self-SFT matches
  (teacher contributed nothing) or any guard regresses after one lr/epoch retry (then the recipe is math-only).
  Plan and reading rules: `technical_details/step3_plan_frontier_kd.md`.
