# STEP1 — The method and the evidence that the mechanism is what we think it is

Protocol: MATH-500, Dream-v0 family, lockstep decoder @1024 tokens / 1024 steps (one token committed per step), top_p 0.9, conf_temp 0.1.
Subsets: first 200 = ids 0–199 (screening), last 300 = ids 200–499 (harder; the default since 2026-10-12), all 500.
Notation in `technical_details/naming.md`; generated numbers in `technical_details/main_table.md`; sharded runs merged by `details/draft_scripts/merge_shards.py`.

**The method (content gate).** Two models decode the same canvas in lockstep. At every step the *student* picks the position to
write (its own confidence ranking) and proposes a token with raw probability p. If p < θ the *teacher's* token for that position is
written instead; otherwise the student's. The teacher never changes the order. (Implementation note: the teacher currently runs every
step; since p is known before the teacher is needed, it could be called only on the ~15% low-confidence steps, i.e. ≈1.15× compute.)

Earlier dead ends, one line each: item-level routing by sequence confidence = random (offline, `technical_details/figs/confidence_rerun_sim.md`);
the 512-token gate was truncation-bound (`technical_details/e1_e3_e4_gate_summary.md`); answer-level voting works (59.6 at 3 decodes) but is not
online and the AR reference under lenient scoring is 71 (`technical_details/theta_vote.md`).

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
| @0.6 | H200 | 52.0 |  |  | 14.6%‡ | 2.28%‡ |  |  |  |  |
| @0.7 † | B300 | <u>54.0</u> | <u>48.0</u> | <u>50.4</u> | — | 2.57%§ | 77% | 62.1 | 46/28, 0.05 / 37/25, 0.16 | 66/47, 0.09 / 63/39, 0.02 |
| @0.8 | H200 | 53.5 |  |  | 23.6%‡ | 2.65%‡ |  |  |  |  |
| @0.6 / @0.8 / @0.9, last 300 (running) | | | | | | | | | | |
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

| pair | arm | last 300 acc | finish rate | acc on finished | vs content gate (win/loss, p) |
|---|---|---:|---:|---:|---|
| Ins ⇐ A(2.0) | content gate @0.8 | 49.0 | 93% | 52.9 | — |
| | position + content @0.8 | | | | |
| | position only @0.8 | | | | |
| | unconditional random 13.5% (= rate 1 of @0.8) | | | | |
| | unconditional random 5.3% (= rate 1 of @0.6; vs @0.6 = 49.3) | | | | |
| M(.4,.6) ⇐ M(0,1) | content gate @0.8 (H200 rerun) | | | | |
| | position + content @0.8 | | | | |
| | position only @0.8 | | | | |
| | unconditional random (dose from the @0.8 rerun) | not yet submitted | | | |

Reading rules, fixed in advance: position+content ≈ content → order adds nothing; position-only ≈ student → order alone is nothing;
content gate ≫ unconditional random → confidence knows where to ask (the earlier "random among disagreements" control did not test
this: it was handed the disagreement positions for free and matched the token-changed count instead, see `tmp.md` 2026-10-12).

Supporting decomposition (student M(.4,.6), teacher A(2.0), `technical_details/dose_curve_mechanism.md`): the teacher's tokens raise the finish rate
(81% → 89–95%), the student's remaining tokens keep the precision on finished problems (51.9 → 57.3; 100% teacher content collapses
to the teacher's 49.5); the weight midpoint (0.7,1.3) gets the finish rate but not the precision; A(3.0) is past the dose optimum.
