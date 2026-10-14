# STEP3 preliminary study — is the training-time "low-confidence position" the decode-time one? (plan, 2026-10-14)

User's concern (2026-10-14): at decode time the gate evaluates confidence at **one** position per step (the student's most confident
mask, 97–99 % the leftmost one, with a complete left prefix). A masked-diffusion training loss corrupts the whole response at once and
scores every masked position, so "p < θ" there is evaluated under a different state (many masks, no prefix). If the two sets of
positions differ, a selective distillation loss trained on "p < θ under random masking" is not the gate that gave +12 at inference.
Second question (user): at those positions, do several teachers agree (consensus targets), and does the agreed token equal what the gate wrote?

## Design (one GPU job, zero new decoding)
Data: the 50 recorded problems of Ins ⇐ A(2.0) @0.8 (ids 200–249, `details/logs/arness50/math500/ins-to-a20-t08/records`): final canvas,
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

Script: `scripts/step3_signal_check.py`; runner `modules/run_step3_signal.sh`; output `details/logs/step3_signal_check/signal_check.json`.
Cost: 50 × 13 corrupted inputs × 5 models = 3 250 forwards ≈ 15 min on H200·0.51 plus 5 model loads.
