# STEP2 — Can several models do better than one teacher? What does that tell us for distillation? (2026-10-15)

Protocol as in STEP1: MATH-500 last 300, lockstep @1024, H200·0.51. Details: `technical_details/step2_*.md`; old full tables: `technical_details/archive/step2_verdict_20261012.md`.

## Takeaways
1. **One over teacher is the ceiling.** Every way of combining models at decode time — soft mixing, teacher averaging, picking the more confident teacher, voting, a second teacher line — lands at the single-teacher number (≈ 48–49). Only an answer-level cascade gains (+4–6) and it costs 1.5–2.5×.
2. **The gain is one simple signal**: *where* = the student's low-confidence steps (≈ 10 % of steps), *what* = the over teacher's token. That signal can be reproduced at training time without labels (STEP3 prelim).
3. **So distil, don't combine.** STEP3 trains one 1× model with: one over teacher A(1.5), KL to its distribution, only at the student's low-confidence frontier of a prefix-visible state, label-free prompts, short LoRA adaptation.

## 1. Reference points
| system | acc | cost |
|---|---:|---:|
| Ins (student) | 36.7 | 1× |
| A(2.0) over addition (teacher) | 46.7 | 1× |
| gate Ins ⇐ A(1.5) @0.8 / ⇐ A(2.0) @0.8 | 48.3 / 49.0 | 1.14× |
| AR Qwen2.5-7B-Instruct strict / lenient | 55.3 / 70.7 | — |

## 2. What we combined
| combination | acc | reading |
|---|---:|---|
| teacher dose: Ins ⇐ A(β) @0.8, β = 1.0 / 1.5 / 2.0 / 2.5 / 3.0 | 44.3 / 48.3 / 49.0 / 47.0 / collapsed | teacher must be past the peak; plateau from 1.5 → **use the smallest β past the peak** |
| weight-space midpoint (0.7,1.3), no gate | 44.3 | the mixture is not a point in weight space |
| soft mix: argmax of (p_student + p_teacher)/2 | 46.3 | fewer tokens change; worse than hard replacement |
| mean of A(1.5)/A(2.0)/A(2.5) | not run | the three singles make the same errors (Jaccard 0.63–0.67); their majority vote = 48.3 = single |
| more-confident of {A(2.0), M(0,1)} | 48.7 | = single teacher |
| other line: Ins ⇐ M(0,1) @0.8 | 45.7 | inherits M(0,1)'s unfinished answers |
| abstention cascade M(0,1) → Ins ⇐ A(2.0) | 53.0 at 1.5×; 55–57 at 2.5× | the only decode-time gain; needs whole answers, not a token signal |
| AR first, diffusion on AR's unboxed answers | 66.7–68.3 | below AR lenient 70.7; nothing here beats AR |

## 3. Design decisions for STEP3 (each one has a control)
| decision | choice | control that removes it |
|---|---|---|
| teacher | one over teacher, A(1.5) | C3: A(1.0), not over |
| target | KL to the teacher's full distribution | C2: CE on the student's own token (self-SFT) |
| positions | only where the student's p < 0.8 | C1: every frontier position |
| training state | prefix visible, suffix masked; loss on the first 4 masks (prelim: reproduces 96 % of the gate's positions; random masking 11–19 %) | C4: random masking |
| data | 17 k prompts, no labels (Tulu-3 incl. math, MATH train, code, chat); LoRA r64, 2 epochs | — |

Success: 1× student ≥ Ins + 6 on MATH last 300 and > C2 + 3, no loss on GSM8K / HumanEval+ / IFEval.
Kill: C2 matches (teacher adds nothing), or a guard regresses after one retry.
Plan: `technical_details/step3_plan_frontier_kd.md`. Pipeline: `code/step3/pipeline/`.
