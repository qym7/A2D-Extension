# STEP3 preliminary study — is the training-time "low-confidence position" the decode-time one? (plan, 2026-10-14)

User's concern (2026-10-14): at decode time the gate evaluates confidence at **one** position per step (the student's most confident
mask, 97–99 % the leftmost one, with a complete left prefix). A masked-diffusion training loss corrupts the whole response at once and
scores every masked position, so "p < θ" there is evaluated under a different state (many masks, no prefix). If the two sets of
positions differ, a selective distillation loss trained on "p < θ under random masking" is not the gate that gave +12 at inference.
Second question (user): at those positions, do several teachers agree (consensus targets), and does the agreed token equal what the gate wrote?

## Design (one GPU job, zero new decoding)
Data: the 50 recorded problems of Ins ⇐ A(2.0) @0.8 (ids 200–249, `outputs/step1/arness50/math500/ins-to-a20-t08/records`): final canvas,
commit order, and the student's raw p at every step. Decode-time sets per problem: L_dec = positions committed at a step with p < 0.8;
S_dec ⊂ L_dec = positions where the teacher's token differed (actually substituted).

Corruptions of the same final canvas:
- **suffix S(t)**: the exact decode-time state at step t (positions committed before t visible, the rest masked). 8 steps per problem,
  half drawn from the low-confidence steps, half uniform. Includes a determinism check: the student's p at the leftmost mask must
  reproduce the recorded p_driver_raw.
- **random R(r)**: standard SFT corruption, each response position masked i.i.d. with r ∈ {0.1, 0.3, 0.5, 0.7, 0.9} (response = prompt end
  to the first `<|im_end|>`/`<|endoftext|>`).

Models: student Ins; teachers A(1.5), A(2.0), A(2.5), M(0,1) — loaded one at a time. At every masked position: raw argmax + its p.

## Metrics (pooled over problems, per corruption)
| name | meaning |
|---|---|
| low_share | share of masked positions with student p < 0.8 = how much of the loss a "p < θ" rule would select |
| dec_low_recall / dec_sub_recall | of the decode-time L_dec / S_dec positions that are masked here, how many are low-confidence again |
| low_precision_vs_dec_low | of the training-time low-confidence positions, how many were low-confidence at decode time |
| agree_A / agree_A20_M01 / agree_all | at training-time low-confidence positions: the three A(β) agree / A(2.0) = M(0,1) / all four agree |
| agreeA_eq_final / agreeA_eq_student | when the A(β) agree: their token = the token the gate finally wrote / = the student's own argmax |
| student_eq_final, A20_eq_final | at low-confidence positions: student's / A(2.0)'s argmax = the final token |
| conf_q10_25_50_75 | student confidence quantiles at masked positions (does θ need to depend on r?) |

## Reading rules (fixed before the run)
- **Suffix reproduces the gate** (sanity): leftmost_det ≈ 8/8 per problem, dec_low_recall ≈ 1 at the leftmost mask.
- **Random masking is a different signal** if, at r ≥ 0.5, low_share > 0.5 or low_precision_vs_dec_low < 0.3: the "p < θ" rule under
  random masking selects mostly positions the gate never touched. Then the training recipe must use suffix (prefix-visible) corruption
  or a ratio-dependent threshold (quantile) — decided by conf_q.
- **Consensus targets are worth having** if agree_A ≥ 0.6 and agreeA_eq_final ≥ 0.9 at low-confidence positions: the three teachers
  give one clean label that is what the gate would have written. If agree_A < 0.4, the consensus set is too small and a single teacher
  (A(1.5) or A(2.0)) is the target.
- **student_eq_final** is a measure of how much of the low-confidence set the gate actually changes (should be ≈ 1 − rate 2 / rate 1 ≈ 0.63 at the suffix states).

Script: `code/step3/signal_check.py`; runner `modules/run_step3_signal.sh`; output `outputs/step3/step3_signal_check/signal_check.json`.
Cost: 50 × 13 corrupted inputs × 5 models = 3 250 forwards ≈ 15 min on H200·0.51 plus 5 model loads.

## Results (2026-10-14, 50 problems ids 200–249, θ = 0.8; `outputs/step3/step3_signal_check_v2/signal_check.json`)
Suffix = exact decode-time states (8 per problem, half of them sampled at low-confidence steps, so low_share there is by design);
"first K" = the K leftmost masks of that state. Random = i.i.d. masking of the response at ratio r.

| corruption | positions | low share | gate-low recalled | gate-sub recalled | low ∩ gate-low / low | A(β) agree | A20=M01 | all 4 agree | agreed = gate wrote | agreed = student | student = final | A20 = final | student p quantiles 10/25/50/75 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| suffix, first 1 (frontier) | 400 | 0.55 | **0.96** | **0.98** | **0.87** | **0.77** | 0.70 | 0.59 | **0.90** | 0.64 | 0.63 | 0.84 | .37 / .53 / .77 / 1.0 |
| suffix, first 4 | 1 599 | 0.57 | 0.97 | 0.97 | 0.55 | 0.70 | 0.64 | 0.52 | 0.70 | 0.68 | 0.50 | 0.60 | .24 / .42 / .71 / 1.0 |
| suffix, first 16 | 6 366 | 0.69 | 0.98 | 0.98 | 0.33 | 0.63 | 0.53 | 0.41 | 0.39 | 0.54 | 0.31 | 0.32 | .09 / .16 / .40 / .94 |
| suffix, all masks | 229 381 | 0.63 | 0.88 | 0.87 | 0.16 | 0.50 | 0.35 | 0.24 | 0.09 | 0.28 | 0.12 | 0.08 | .07 / .11 / .41 / .99 |
| random r = 0.1 | 4 044 | 0.02 | **0.11** | 0.15 | 0.92 | 0.86 | 0.68 | 0.64 | 0.75 | 0.68 | 0.59 | 0.69 | 1.0 / 1.0 / 1.0 / 1.0 |
| random r = 0.3 | 12 288 | 0.03 | 0.15 | 0.22 | 0.82 | 0.83 | 0.76 | 0.69 | 0.70 | 0.70 | 0.57 | 0.66 | .99 / 1.0 / 1.0 / 1.0 |
| random r = 0.5 | 20 206 | 0.05 | 0.19 | 0.24 | 0.71 | 0.79 | 0.68 | 0.60 | 0.67 | 0.70 | 0.60 | 0.61 | .95 / 1.0 / 1.0 / 1.0 |
| random r = 0.7 | 28 441 | 0.11 | 0.34 | 0.40 | 0.57 | 0.75 | 0.68 | 0.58 | 0.60 | 0.70 | 0.53 | 0.53 | .69 / .99 / 1.0 / 1.0 |
| random r = 0.9 | 36 586 | 0.52 | 0.77 | 0.80 | 0.26 | 0.64 | 0.54 | 0.41 | 0.38 | 0.61 | 0.36 | 0.32 | .12 / .26 / .74 / .99 |

Determinism: the student's p at the frontier reproduced the recorded p_driver_raw within 0.02 in 252/359 states (bf16 batch-shape noise; the
rest differ by more but the low/high decision flips rarely — gate-low recall 0.96).

### Reading
1. **Random masking is a different signal.** With the right context visible the student is confident almost everywhere (median p = 1.0 up
   to r = 0.7): at r ≤ 0.5 only 11–19 % of the positions the gate handed over are low-confidence again. The positions it does select are
   gate positions (precision 0.7–0.9) but it finds one in ten. At r = 0.9 recall is 0.77 only because everything is masked (precision 0.26).
   A "p < θ under random masking" loss therefore trains on a tenth of the gate's positions plus, at high r, mostly positions the gate never touched.
2. **The suffix state at the frontier is the gate.** At the leftmost mask of a decode-time state, 96 % of the gate's low-confidence
   positions and 98 % of its substitutions are low-confidence again, and 87 % of the low-confidence positions are gate positions.
   Four masks in, precision is already 0.55; sixteen in, 0.33: positions behind the frontier are uncertain for a different reason
   (missing left context) and the gate never scores them. **Training corruption = prefix visible, suffix masked; loss = the frontier
   (K ≤ 4) only.**
3. **Consensus targets are clean at the frontier.** The three A(β) agree on 77 % of frontier low-confidence positions, and when they agree
   the token is the one the gate actually wrote 90 % of the time (A(2.0) alone 84 %; the student's own argmax 63 %). All four teachers
   (incl. M(0,1)) agree on 59 %. So "three A(β) agree" is a usable target rule: ≈ 3/4 coverage, 90 % fidelity to the gate.
4. **θ needs no ratio dependence** if the suffix corruption is used (the frontier p distribution matches decoding by construction);
   under random masking it would (median p collapses from 1.0 to 0.74 between r = 0.7 and 0.9).

### Consequence for the STEP3 recipe
Suffix corruption at a random cut (optionally several cuts per sequence), student forward, loss on the first K ≤ 4 masks where the student's
raw p < 0.8, target = teacher distribution (single A(1.5)/A(2.0), or the mean of the three A(β) restricted to positions where they agree),
no loss elsewhere (or a small standard masked-CE term on the student's own rollout to keep the diffusion objective). Teachers see the same
corrupted input. No labels are needed, so the prompt set is not restricted to math.
