# STEP1 — AR-ness of the gated decode (what we can and cannot say, 2026-10-13)

**Question.** Does the gate (or the teacher) change *how left-to-right* the decode is? Is "student keeps the order" really a different order from the teacher's?

## 1. What the saved runs allow (zero GPU)
The lockstep decoder commits one position per step but does not log the commit sequence; `gate_stats.json` only keeps
(a) the number of steps at which the teacher chose the position (`n_steps_sched_ported`, schedule/both modes) and
(b) the canvas positions of substituted tokens (`sub_positions`, first 20k). So the DiffuCoder Global/Local AR-ness@k of
these arms cannot be recomputed post hoc. From (a) and (b):

| arm (last 300, @0.8) | steps where the teacher picked the position | substituted-token position, share per canvas eighth (0–7) |
|---|---:|---|
| Ins ⇐ A(2.0) content | 0 % | 11 17 15 15 12 10 7 13 |
| Ins ⇐ A(2.0) position + content | 14.5 % | 12 17 15 14 13 10 7 13 |
| Ins ⇐ A(2.0) position only | 5.9 %* | — |
| Ins ⇐ A(2.0) uncond. random 13.5 % | 0 % | 8 14 11 10 7 8 11 **31** |
| M(.4,.6) ⇐ M(0,1) content | 0 % | 9 12 13 14 14 13 11 15 |
| M(.4,.6) ⇐ M(0,1) position + content | 21.6 % | 8 14 10 14 13 13 11 17 |
| M(.4,.6) ⇐ M(0,1) position only | 29.4 % | — |
| M(.4,.6) ⇐ M(0,1) uncond. random 23.9 % | 0 % | 9 10 12 15 14 13 11 17 |
| M(.4,.6) ⇐ A(2.0) content | 0 % | 10 14 12 12 12 11 9 **20** |

\* in position-only mode the student's trajectory drifts, so fewer steps fall below θ than in the content gate.

Readings. The gate's substitutions are spread over the whole answer, slightly front-loaded (eighths 1–3 = the setup of the solution);
the unconditional random control on Ins puts 31 % of its substitutions in the last eighth — i.e. into the post-EOS padding that the
student fills late, where a token change is worthless. That alone explains part of why "same dose, random positions" does nothing:
a confidence-chosen position is almost never padding.

## 2. What A2D already measured (`../dream_study/step2_instruct/outputs/official_eval/arness/arness_posthoc_summary.json`, 20 prompts, native decode)
| model, MATH chat mode | Spearman(order, position) | adjacency rate |
|---|---:|---:|
| Dream-Base (raw) | 0.94 | 0.66 |
| Dream-Instruct | 1.00 | 0.94 |
| best merge (τ_D/τ_A convex) | 1.00 | 0.93 |

Dream-Instruct and the merges are already essentially left-to-right on MATH in chat mode, and our lockstep setting (one token per step,
confidence temperature 0.1) is even greedier. So the student's and the teacher's orders are both ≈ AR and nearly identical; handing the
position to the teacher mostly re-picks the same next position. That is consistent with §2 of `step1_methods.md`: position-only ≈ student,
position + content ≈ content. The gate is best described as *near-AR diffusion decoding where the teacher supplies the token at the
student's uncertain steps*, not as a change of order.

## 3. What a direct measurement would need
Add a per-problem commit-order dump to `scripts/step1_lockstep_gated.py` (one list of 1024 positions per item, written next to
`texts.jsonl`), then rerun student alone, teacher alone, content gate, position + content on ~50 problems each (AR-ness is a stable
per-run statistic: A2D used 20 prompts). Cost ≈ 4 arms × 50 problems × 2× ≈ 6 GPU-h on H200·0.51. Metrics: DiffuCoder Global/Local
AR-ness@{1,2,4,8} via `arness_metrics.py`, plus Spearman and adjacency for continuity with A2D. Not submitted — awaiting a decision.
