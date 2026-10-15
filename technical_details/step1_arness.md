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

## 3. Direct measurement (2026-10-13, ids 200–249, `code/step1/arness_from_records.py`, H200·0.51, `--record` runs)
Definitions checked against DiffuCoder §4.1 / App. A.2 (arXiv:2506.20639v2): **Global AR-ness@k** = share of decoding steps whose unmasked
position is among the k earliest remaining masks; **Local AR-ness@k** = share of steps where the new token and the k previously predicted
tokens form a strictly consecutive span. The paper averages over every step of the fixed-length canvas (512 there, 1024 here), i.e. including
the positions after the end of the answer ("full"). Because Dream fills the post-EOS region with high-confidence EOS tokens in a
non-left-to-right order, the full-canvas number also measures *how much padding there is*; we therefore add the same metrics restricted to
the positions before the first `<|im_end|>` ("response"; truncated answers = whole canvas). Order = sequence of committed canvas positions
(one per step; the last step's 3 leftovers flattened in increasing position order, a 0.3 % tie rate). Texts of every recorded run equal the
earlier non-record run of the same arm on all 50 ids (argmax decode is deterministic).

| arm | n | acc | Global@1 (full) | Local@1 (full) | Local@8 (full) | Global@1 (response) | Local@1 (response) | Spearman | adjacency | response len | texts = earlier run |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Ins alone | 50 | 36 | 0.988 | 0.970 | 0.910 | 0.960 | 0.899 | 1.000 | 0.977 | 421 | 50/50 |
| A(2.0) alone | 50 | 42 | 0.941 | 0.897 | 0.711 | 0.941 | 0.897 | 1.000 | 0.922 | 1021 | 50/50 |
| Ins ⇐ A(2.0) content @0.8 | 50 | 46 | 0.967 | 0.934 | 0.809 | 0.958 | 0.912 | 1.000 | 0.949 | 812 | 50/50 |
| Ins ⇐ A(2.0) position+content @0.8 | 50 | 48 | 0.957 | 0.932 | 0.806 | 0.948 | 0.914 | 1.000 | 0.948 | 847 | 50/50 |
| M(.4,.6) alone | 50 | 40 | 0.871 | 0.855 | 0.620 | 0.957 | 0.907 | 0.999 | 0.889 | 460 | 50/50 |
| M(0,1) alone | 50 | 44 | 0.934 | 0.892 | 0.699 | 0.953 | 0.911 | 1.000 | 0.918 | 602 | 50/50 |
| M(.4,.6) ⇐ M(0,1) content @0.8 | 50 | 40 | 0.883 | 0.880 | 0.672 | 0.950 | 0.909 | 0.998 | 0.908 | 577 | 50/50 |
| M(.4,.6) ⇐ M(0,1) position+content @0.8 | 50 | 46 | 0.924 | 0.886 | 0.686 | 0.941 | 0.908 | 1.000 | 0.913 | 593 | 50/50 |

Readings (corrected 2026-10-13 after fixing the response-end detection — an earlier version read the end of the answer off the last tokens
and under-counted truncated answers).
- **On the answer itself every arm is equally and almost fully AR**: Global@1 0.94–0.96, Local@1 0.90–0.91, Spearman ≈ 1.00. The lockstep
  decoder at confidence temperature 0.1 commits left to right for every model; nothing here distinguishes student, teacher or gate.
- **The full-canvas spread (0.87–0.99) is a response-length effect, not a model property**: Ins (mean answer 421 tokens, 600 tokens of
  padding) scores 0.988 and M(.4,.6) (460 tokens) 0.871 only because the padding region is committed out of order; A(2.0), which runs to the
  end of the canvas on most problems (mean 1021), has full = response.
- **Position + content vs content only**: response Global@1 0.948 vs 0.958 (Ins pair) and 0.941 vs 0.950 (merge pair) — within 0.01. The §2
  accuracy differences (51.3 vs 49.0; 44.0 vs 47.7) are not order effects. The "merge becomes more AR with position + content" reading of the
  full-canvas column (0.883 → 0.924) was the padding region being committed in a different order, not the answer: on the response the two
  are equal (0.950 vs 0.941).
- Consequence for A2D's earlier numbers (`arness_posthoc_summary.json`, native decode, 20 prompts): those are also full-canvas and so carry the
  same length confound; the Spearman/adjacency there are consistent with the response-only picture here (everything ≈ AR in chat mode).
