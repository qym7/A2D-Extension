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

## 3. Direct measurement (2026-10-13, ids 200–249, `scripts/step1_arness_from_records.py`, H200·0.51, `--record` runs)
Order = sequence of committed canvas positions. "full" = whole 1024-token canvas incl. post-EOS padding (DiffuCoder App. A.2); "response" = positions
before the first EOS only. Texts of every recorded run equal the earlier non-record run of the same arm on all 50 ids (argmax decode is deterministic).

| arm | n | acc | Global@1 (full) | Local@1 (full) | Local@8 (full) | Global@1 (response) | Local@1 (response) | Spearman | adjacency | response len | texts = earlier run |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Ins alone | 50 | 36 | 0.988 | 0.970 | 0.910 | 0.956 | 0.887 | 1.000 | 0.977 | 305 | 50/50 |
| A(2.0) alone | 50 | 42 | 0.941 | 0.897 | 0.711 | 0.934 | 0.881 | 1.000 | 0.922 | 295 | 50/50 |
| Ins ⇐ A(2.0) content @0.8 | 50 | 46 | 0.967 | 0.934 | 0.809 | 0.961 | 0.903 | 1.000 | 0.949 | 509 | 50/50 |
| Ins ⇐ A(2.0) position+content @0.8 | 50 | 48 | 0.957 | 0.932 | 0.806 | 0.947 | 0.911 | 1.000 | 0.948 | 535 | 50/50 |
| M(.4,.6) alone | 50 | 40 | 0.871 | 0.855 | 0.620 | 0.913 | 0.816 | 0.999 | 0.889 | 260 | 50/50 |
| M(0,1) alone | 50 | 44 | 0.934 | 0.892 | 0.699 | 0.930 | 0.866 | 1.000 | 0.918 | 300 | 50/50 |
| M(.4,.6) ⇐ M(0,1) content @0.8 | 50 | 40 | 0.883 | 0.880 | 0.672 | 0.899 | 0.850 | 0.998 | 0.908 | 338 | 50/50 |
| M(.4,.6) ⇐ M(0,1) position+content @0.8 | 50 | 46 | 0.924 | 0.886 | 0.686 | 0.935 | 0.866 | 1.000 | 0.913 | 257 | 50/50 |

Readings.
- **Everything is near-AR** (Global@1 0.87–0.99, Spearman ≈ 1.00): the lockstep decoder at confidence temperature 0.1 commits almost strictly left to right for every model. Ins is the most AR (0.988), the merges the least (M(.4,.6) 0.871).
- **Position + content vs content only: a small shift toward the teacher's order, nothing more.** Ins ⇐ A(2.0): 0.957 vs 0.967 (teacher 0.941); M(.4,.6) ⇐ M(0,1): 0.924 vs 0.883 (teacher 0.934). On the response only the Local@1 numbers are within 0.02. The accuracy differences in `step1_methods.md` §2 (51.3 vs 49.0; 44.0 vs 47.7) are not accompanied by a change in *how* left-to-right the decode is — consistent with "the teacher's order carries nothing".
- **The content gate keeps the student's order**: Ins ⇐ A(2.0) sits between Ins and A(2.0) (0.967) only because the teacher's tokens make the answers longer (response 305 → 509 tokens) and long answers have more non-adjacent commits; M(.4,.6) ⇐ M(0,1) is at the student (0.883 vs 0.871).
- More AR is not better: the least AR model (M(.4,.6)) has the highest precision on finished problems; the most AR (Ins) the lowest accuracy. AR-ness is a property of the decoder temperature here, not a lever.
