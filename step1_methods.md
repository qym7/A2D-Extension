# STEP1 — The method and the evidence that the mechanism is what we think it is

Protocol: MATH-500, Dream-v0 family, lockstep decoder @1024 tokens / 1024 steps (one token committed per step), top_p 0.9, conf_temp 0.1.
Subsets: first 200 = ids 0–199 (screening), last 300 = ids 200–499 (harder; the default since 2026-10-12), all 500.
Notation in `technical_details/step0_naming.md`; generated numbers in `technical_details/step1_main_table.md`; sharded runs merged by `scripts/step1_merge_shards.py`.

**The method (content gate).** Two models decode the same canvas in lockstep. At every step the *student* picks the position to
write (its own confidence ranking) and proposes a token with raw probability p. If p < θ the *teacher's* token for that position is
written instead; otherwise the student's. The teacher never changes the order. (Implementation note: the teacher currently runs every
step; since p is known before the teacher is needed, it could be called only on the ~15% low-confidence steps, i.e. ≈1.15× compute.)

Earlier dead ends, one line each: item-level routing by sequence confidence = random (offline, `technical_details/step0_figs/confidence_rerun_sim.md`);
the 512-token gate was truncation-bound (`technical_details/step1_gate512_e1_e3_e4.md`); answer-level voting works (59.6 at 3 decodes) but is not
online and the AR reference under lenient scoring is 71 (`technical_details/step1_theta_vote.md`).

## 1. All student ⇐ teacher pairs (big table)

| System | HW | First 200 | Last 300 | All 500 | Rate 1: handed to teacher | Rate 2: token changed | Finish rate (last 300) | Acc on finished (last 300) | Last 300: vs student / vs teacher (win/loss, p) | All 500: vs student / vs teacher (win/loss, p) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| **AR reference (different decoder: lm-eval greedy, 1024 tokens)** | | | | | | | | | | |
| AR Qwen2.5-7B-Instruct, strict (\boxed only) | H200 | 57.0 | 55.3 | 56.0 | | | 78% | 70.9 | | |
| AR Qwen2.5-7B-Instruct, lenient (final-answer extraction) | H200 | | 70.7 | 71.0 | | | (no-\boxed answers are complete, not truncated) | | | |
| **Single diffusion models (1×)** | | | | | | | | | | |
| Ins | B300 | 44.5 | 36.7 | 39.8 |  |  | 76% | 48.5 |  |  |
| M(.4,.6) best merge | B300 | 53.5 | 42.0 | 46.6 |  |  | 81% | 51.9 |  |  |
| M(0,1) over merge | B300 | 48.0 | 44.0 | 45.6 |  |  | 74% | 59.2 |  |  |
| A(2.0) over addition | H200 | 58.5 | 46.7 | 51.4 |  |  | 94% | 49.5 |  |  |
| **Ins ⇐ A(2.0)** | | | | | | | | | | |
| @0.0 = student alone | B300 | 44.5 | 36.7 | 39.8 |  |  | 76% | 48.5 |  |  |
| @0.2 | H200 | 44.5 |  |  | 0.1%‡ | 0.06%‡ |  |  |  |  |
| @0.4 | H200 | 53.0 |  |  | 1.1%‡ | 0.68%‡ |  |  |  |  |
| @0.6 | H200 | 56.5 | <u>49.3</u> | <u>52.2</u> | 5.3% | 2.55% | 92% | 53.8 | 57/19, 0.00 / 41/33, 0.42 | 96/34, 0.00 / 59/55, 0.78 |
| @0.8 | H200 | <u>59.0</u> | <u>49.0</u> | <u>53.0</u> | 13.5% | 4.96% | 93% | 52.9 | 62/25, 0.00 / 40/33, 0.48 | 106/40, 0.00 / 60/52, 0.51 |
| @1.0 ≈ teacher alone (reference) | H200 | 58.5 | 46.7 | 51.4 |  |  | 94% | 49.5 |  |  |
| **M(.4,.6) ⇐ A(2.0)** | | | | | | | | | | |
| @0.0 = student alone | B300 | 53.5 | 42.0 | 46.6 |  |  | 81% | 51.9 |  |  |
| @0.4 | H200 |  | 43.7 |  | 2.8% | 1.76% | 83% | 52.6 | 31/26, 0.60 / 36/45, 0.37 |  |
| @0.6 | H200 | 58.0 | <u>49.7</u> | <u>53.0</u> | 7.4% | 3.49% | 89% | 56.0 | 48/25, 0.01 / 38/29, 0.33 | 75/43, 0.00 / 60/52, 0.51 |
| @0.8 | H200 | <u>59.0</u> | <u>51.0</u> | <u>54.2</u> | 14.9% | 5.06% | 89% | 57.3 | 49/22, 0.00 / 39/26, 0.14 | 79/41, 0.00 / 60/46, 0.21 |
| @1.0 (measured: teacher content, student order) | H200 |  | 46.3 |  | 99.5% | 6.57% | 95% | 48.8 | 48/35, 0.19 / 29/30, 1.00 |  |
| random positions, 26% dose | H200 |  | 46.3 |  | 0.0% | 2.52% | 84% | 54.9 | 38/25, 0.13 / 38/39, 1.00 |  |
| random positions, 85% dose | H200 |  | <u>50.0</u> |  | 0.0% | 5.77% | 93% | 53.6 | 52/28, 0.01 / 41/31, 0.29 |  |
| @1.0 ≈ teacher alone (reference) | H200 | 58.5 | 46.7 | 51.4 |  |  | 94% | 49.5 |  |  |
| **M(.4,.6) ⇐ M(0,1)** | | | | | | | | | | |
| @0.0 = student alone | B300 | 53.5 | 42.0 | 46.6 |  |  | 81% | 51.9 |  |  |
| @0.3 † | B300 | <u>54.0</u> | 43.3 | <u>47.6</u> | — | 0.92%§ | 78% | 55.3 | 21/17, 0.63 / 29/31, 0.90 | 33/28, 0.61 / 58/48, 0.38 |
| @0.6 | H200 | 52.0 | <u>45.3</u> | <u>48.0</u> | 14.8% | 2.26% | 77% | 58.9 | 36/26, 0.25 / 33/29, 0.70 | 53/46, 0.55 / 54/42, 0.26 |
| @0.7 † | B300 | <u>54.0</u> | <u>48.0</u> | <u>50.4</u> | — | 2.57%§ | 77% | 62.1 | 46/28, 0.05 / 37/25, 0.16 | 66/47, 0.09 / 63/39, 0.02 |
| @0.8 | H200 | 53.5 | <u>47.7</u> | <u>50.0</u> | 23.9% | 2.73% | 77% | 61.9 | 46/29, 0.06 / 36/25, 0.20 | 68/51, 0.14 / 61/39, 0.04 |
| @0.9 | H200 |  | <u>47.3</u> |  | 29.0% | 2.78% | 77% | 61.7 | 45/29, 0.08 / 33/23, 0.23 |  |
| @1.0 ≈ teacher alone (reference) | B300 | 48.0 | 44.0 | 45.6 |  |  | 74% | 59.2 |  |  |
| **Ins ⇐ M(.4,.6)** | | | | | | | | | | |
| @0.0 = student alone | B300 | 44.5 | 36.7 | 39.8 |  |  | 76% | 48.5 |  |  |
| @0.2 | H200 | 43.0 |  |  | 0.1%‡ | 0.04%‡ |  |  |  |  |
| @0.4 | H200 | 46.5 |  |  | 0.8%‡ | 0.29%‡ |  |  |  |  |
| @0.6 | H200 | 53.5 | <u>44.7</u> | <u>48.2</u> | 3.2% | 0.80% | 80% | 55.8 | 47/23, 0.01 / 35/27, 0.37 | 78/36, 0.00 / 55/47, 0.49 |
| @0.8 | H200 | 52.0 | 41.7 | 45.8 | 6.3% | 0.98% | 79% | 52.7 | 39/24, 0.08 / 29/30, 1.00 | 73/43, 0.01 / 48/52, 0.76 |
| @1.0 ≈ teacher alone (reference) | B300 | 53.5 | 42.0 | 46.6 |  |  | 81% | 51.9 |  |  |

Rate 1 = share of committed tokens at which the student's raw p < θ (handed to the teacher); Rate 2 = share of committed tokens where the teacher's token was written and differs from the student's. Computed on the last 300 unless marked ‡ (first 200 only) or § (9/30 run: all-500 summary, rate 1 not recorded). Random controls have rate 1 = 0 by construction.
<u>Underlined</u> = better than both its student and its teacher on that subset (point estimate; see the p columns).
@0.0 is exactly the student alone (verified token-identical). @1.0 is the teacher alone only as a reference: the gate at θ=1.0 writes the teacher's token at every position but keeps the student's decode order, and the one measured @1.0 run (M(.4,.6) ⇐ A(2.0)) lands on the teacher's accuracy with different texts.
Finish = output contains \boxed; unfinished outputs are always wrong for our systems (true truncation).

## 2. Mechanism ablations (last 300; the two pairs we care about)

The claim to test: *the gain comes from the teacher's token content at the student's low-confidence steps, with the student keeping
the denoising order.* Three ablations isolate it, all at θ = 0.8, same steps triggered:

| ablation | what changes at a low-confidence step | tests |
|---|---|---|
| content gate (reference) | teacher's token, student's position | — |
| **position + content** (`both_gated`) | teacher picks the position **and** the token | does the teacher's order add anything on top of its content? |
| **position only** (`schedule_gated`) | teacher picks the position, student writes the token | is the order alone worth anything? |
| **unconditional random** (`content_random_uncond`, dose = gate's rate 1) | a random step takes the teacher's token, no confidence, no disagreement check | does confidence know *where* to ask? |

| pair | arm (all gated arms: 2× decode, H200·0.51, last 300) | last 300 acc | finish rate | acc on finished | vs content gate @0.8 (win/loss, p) |
|---|---|---:|---:|---:|---|
| Ins ⇐ A(2.0) | **student alone**: Ins (1×, B300) | 36.7 | 76% | 48.5 | 25/62, 0.00 |
| | **teacher alone**: A(2.0) over addition (1×, H200) | 46.7 | 94% | 49.5 | 33/40, 0.48 |
| | content gate @0.8 (reference; student's order, teacher's token where student p < 0.8) | 49.0 | 93% | 52.9 | — |
| | position + content @0.8 | <u>51.3</u> | 94% | 54.8 | 32/25, 0.43 (vs teacher 35/21, 0.08) |
| | position only @0.8 | 38.7 | 76% | 51.1 | 24/55, 0.00 (vs student 36/30, 0.54) |
| | unconditional random 13.5% (= rate 1 of @0.8) | 38.0 | 82% | 46.3 | 25/58, 0.00 (vs student 33/29, 0.70) |
| | unconditional random 5.3% (= rate 1 of @0.6; vs @0.6 = 49.3) | 38.3 | 81% | 47.5 | 28/60, 0.00 (vs student 23/18, 0.53) |
| M(.4,.6) ⇐ M(0,1) | **student alone**: M(.4,.6) best merge (1×, B300) | 42.0 | 81% | 51.9 | 29/46, 0.06 |
| | **teacher alone**: M(0,1) over merge (1×, B300) | 44.0 | 74% | 59.2 | 25/36, 0.20 |
| | content gate @0.8 (reference; H200 rerun — θ sweep 0.6/0.8/0.9 = 45.3/47.7/47.3, all at finish 77%, precision 59–62) | 47.7 | 77% | 61.9 | — |
| | position + content @0.8 | 44.0 | 72% | 61.4 | 30/41, 0.24 (vs teacher 15/15, 1.00 — identical to M(0,1) alone) |
| | position only @0.8 | 43.0 | 77% | 56.1 | 31/45, 0.14 (vs student 31/28, 0.79) |
| | unconditional random 23.9% (= rate 1 of @0.8) | 41.3 | 80% | 51.9 | 25/44, 0.03 (vs student 25/27, 0.89) |

Outcome (all cells filled 2026-10-13; every pre-registered reading rule below is met). Position-only sits at the student on both pairs
(38.7 vs 36.7, p=0.54; 43.0 vs 42.0, p=0.79): the teacher's *order* carries nothing. The M(.4,.6) ⇐ M(0,1) unconditional random at
23.9% is also the student (41.3 vs 42.0) and loses to the gate 25/44 (p=0.03). On both pairs, letting the teacher also pick the position does not beat the content gate (Ins ⇐ A(2.0): 51.3 vs 49.0,
p=0.43; M(.4,.6) ⇐ M(0,1): 44.0 vs 47.7, and 44.0 is exactly the teacher alone, 15/15 ties — handing over the order hands over the
finish rate too, 77% → 72%). The unconditional random control at the gate's own dose sits at the student (38.0 / 38.3 vs Ins 36.7)
and loses to the gate 25/58 and 28/60 (p < 0.001): the confidence signal knows *where* to ask; the dose alone is worthless.

Reading rules, fixed in advance: position+content ≈ content → order adds nothing; position-only ≈ student → order alone is nothing;
content gate ≫ unconditional random → confidence knows where to ask (the earlier "random among disagreements" control did not test
this: it was handed the disagreement positions for free and matched the token-changed count instead, see `tmp.md` 2026-10-12).

**AR-ness of the decode order** (`technical_details/step1_arness.md` §3; ids 200–249 with the commit order recorded, texts identical to the main runs).
Global@1 / Local@1 = DiffuCoder AR-ness over the full 1024-token canvas (share of commits that take the leftmost remaining mask / the position right after the previous one).

| arm | Global@1 | Local@1 | Spearman | response len (tokens) |
|---|---:|---:|---:|---:|
| Ins alone | 0.988 | 0.970 | 1.000 | 305 |
| A(2.0) alone | 0.941 | 0.897 | 1.000 | 295 |
| Ins ⇐ A(2.0) content @0.8 | 0.967 | 0.934 | 1.000 | 509 |
| Ins ⇐ A(2.0) position + content @0.8 | 0.957 | 0.932 | 1.000 | 535 |
| M(.4,.6) alone | 0.871 | 0.855 | 0.999 | 260 |
| M(0,1) alone | 0.934 | 0.892 | 1.000 | 300 |
| M(.4,.6) ⇐ M(0,1) content @0.8 | 0.883 | 0.880 | 0.998 | 338 |
| M(.4,.6) ⇐ M(0,1) position + content @0.8 | 0.924 | 0.886 | 1.000 | 257 |

Every arm is near left-to-right. The merges are the *least* AR (M(.4,.6) 0.871, the model with the highest precision), Ins the most (0.988).
The content gate keeps the student's order (M(.4,.6) ⇐ M(0,1) 0.883 ≈ 0.871); position + content moves it toward the teacher by ≤ 0.04
(0.924 on the merge pair, where it also collapses to the teacher's accuracy). The §2 accuracy differences are therefore not order effects.

Supporting decomposition (student M(.4,.6), teacher A(2.0), `technical_details/step1_dose_curve_mechanism.md`): the teacher's tokens raise the finish rate
(81% → 89–95%), the student's remaining tokens keep the precision on finished problems (51.9 → 57.3; 100% teacher content collapses
to the teacher's 49.5); the weight midpoint (0.7,1.3) gets the finish rate but not the precision; A(3.0) is past the dose optimum.
