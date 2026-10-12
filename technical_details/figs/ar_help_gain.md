# AR-help-best β vs accuracy-best β: how much more AR-help, at what cost (2026-10-09, zero GPU)

![](ar_help_gain.png)

Script `../../details/draft_scripts/ar_help_gain.py`, input `ar_help_vs_acc_peak.json`, output `ar_help_gain.json`.
Rules: AR-help-best β chosen **only on the 0.1 grid** (0.05/0.15/0.25… not eligible); accuracy-best β = reported best on
all points; AR-help ties → smallest tied β; **"still rising ↗" = AR-help-best is the largest 0.1-grid β that was run,
so the AR-help peak is not confirmed.** Dream-Coder MBPP merge dropped (only 2 points on the 0.1 grid).

## Summary
- 15 of 22 lines get more AR-help by moving β up: median **+7.7 pp** of the AR-help pool (range +2.4 to +30.3),
  at a median accuracy cost of **−0.6** (range 0 to −6.5).
- Large gains with ~zero accuracy cost: DiffuCoder-SFT HE addition (+30.3 pp, acc ±0), DiffuCoder-RL merge (+25.6 pp, ±0),
  DreamReasoner 8k merge (+28.6 pp, −0.8), DreamReasoner 4k merge (+23.8 pp, −1.4).
- Costly: Nemotron addition lines (−4.2, −6.5 acc; AR-harm +18) and Nemotron-8B HumanEval merge (−5.5).
- **9 lines still rising** (peak not confirmed): DiffuCoder-SFT HE merge, DiffuCoder-RL addition, DreamReasoner 8k addition,
  DreamReasoner 4k merge and addition, Nemotron-8B IFEval addition, Nemotron-8B MATH-500 / HumanEval / GSM8K merge.
- Same β for both: Dream-v0 IFEval addition (1.1), Nemotron-8B IFEval merge (0.4), Nemotron-8B MBPP merge (grading ⚠).

## Caveats
Help pools are small in code/reasoning (14–46 items): +1 item = 2–7 pp. Sampled decoding (code T0.1–0.2, Qwen3 avg@4)
adds item noise. Grids were designed around accuracy.

## Full table
| series | mode | acc-best β (acc, AR-help) | AR-help-best β (acc, AR-help) | ΔAR-help | Δacc | ΔAR-harm | peak? |
|---|---|---|---|---:|---:|---:|---|
| Dream-v0 · IFEval | merge | 0.6 (66.5, 68/127) | 0.7 (66.0, 71/127) | +3 (+2.4 pp) | -0.5 | +3 | peak found (tie [0.7, 0.8]) |
| Dream-v0 · IFEval | addition | 1.1 (72.1, 85/127) | 1.1 (72.1, 85/127) | +0 (+0.0 pp) | +0.0 | +0 | peak found |
| Dream-Coder · HumanEval | merge | 0.25 (85.4, 9/18) | 0.2 (84.8, 9/18) | +0 (+0.0 pp) | -0.6 | -1 | peak found (tie [0.2, 0.3, 0.4]) |
| Dream-Coder · HumanEval | addition | 0.4 (83.5, 10/18) | 0.5 (82.9, 11/18) | +1 (+5.6 pp) | -0.6 | +1 | peak found |
| DiffuCoder-SFT · HumanEval | merge | 0.2 (79.3, 14/33) | 0.4 (78.7, 16/33) | +2 (+6.1 pp) | -0.6 | +1 | **still rising ↗** |
| DiffuCoder-SFT · HumanEval | addition | 0.15 (73.8, 7/33) | 0.6 (73.8, 17/33) | +10 (+30.3 pp) | +0.0 | +0 | peak found |
| DiffuCoder-SFT · MBPP | merge | 0.05 (79.4, 12/46) | 0.2 (78.6, 15/46) | +3 (+6.5 pp) | -0.8 | +4 | peak found |
| DiffuCoder-SFT · MBPP | addition | 0.4 (78.3, 15/46) | 0.2 (77.8, 15/46) | +0 (+0.0 pp) | -0.5 | +1 | peak found (tie [0.2, 0.4]) |
| DiffuCoder-RL · MBPP+ | merge | 0.3 (69.0, 8/39) | 0.5 (69.0, 18/39) | +10 (+25.6 pp) | +0.0 | +3 | peak found |
| DiffuCoder-RL · MBPP+ | addition | 0.4 (66.9, 1/39) | 0.8 (66.9, 4/39) | +3 (+7.7 pp) | +0.0 | +1 | **still rising ↗** |
| DreamReasoner · MATH-500 @8k | merge | 0.2 (94.8, 8/14) | 0.3 (94.0, 12/14) | +4 (+28.6 pp) | -0.8 | +6 | peak found (tie [0.3, 0.4]) |
| DreamReasoner · MATH-500 @8k | addition | 0.5 (94.4, 9/14) | 0.7 (94.2, 10/14) | +1 (+7.1 pp) | -0.2 | +0 | **still rising ↗** |
| DreamReasoner · MATH-500 @4k | merge | 0.2 (91.6, 14/21) | 0.4 (90.2, 19/21) | +5 (+23.8 pp) | -1.4 | +4 | **still rising ↗** |
| DreamReasoner · MATH-500 @4k | addition | 0.3 (91.4, 13/21) | 0.4 (90.2, 15/21) | +2 (+9.5 pp) | -1.2 | +2 | **still rising ↗** |
| Nemotron-3B · ifeval | merge | 0.3 (66.9, 31/82) | 0.5 (65.6, 36/82) | +5 (+6.1 pp) | -1.3 | -1 | peak found |
| Nemotron-3B · ifeval | addition | 0.4 (65.2, 29/82) | 0.8 (61.0, 36/82) | +7 (+8.5 pp) | -4.2 | +18 | peak found |
| Nemotron-8B · ifeval | merge | 0.4 (73.0, 51/88) | 0.4 (73.0, 51/88) | +0 (+0.0 pp) | +0.0 | +0 | peak found |
| Nemotron-8B · ifeval | addition | 0.5 (69.3, 39/88) | 0.9 (62.8, 43/88) | +4 (+4.5 pp) | -6.5 | +18 | **still rising ↗** |
| Nemotron-8B · math-500 | merge | 0.4 (89.0, 15/28) | 0.4 (89.0, 15/28) | +0 (+0.0 pp) | +0.0 | +0 | **still rising ↗** |
| Nemotron-8B · human-eval | merge | 0.3 (76.8, 14/28) | 0.4 (71.3, 18/28) | +4 (+14.3 pp) | -5.5 | +0 | **still rising ↗** |
| Nemotron-8B · mbpp | merge | 0.2 (68.8, 53/88) | 0.2 (68.8, 53/88) | +0 (+0.0 pp) | +0.0 | +0 | peak found ⚠ grading |
| Nemotron-8B · gsm8k | merge | 0.4 (94.5, 28/42) | 0.4 (94.5, 28/42) | +0 (+0.0 pp) | +0.0 | +0 | **still rising ↗** |

## AR-help at the accuracy-best point vs at the AR-help-best point (absolute)

AR-help = items the parent got wrong and the AR model got right that this point gets right (parent itself = 0 by definition).

| line | acc-best β → AR-help | AR-help-best β → AR-help | Δ |
|---|---|---|---|
| Dream-v0 · IFEval · merge | 0.6 → 68/127 (53.5%) | 0.7 → 71/127 (55.9%) | +3 (+2.4 pp) |
| Dream-v0 · IFEval · addition | 1.1 → 85/127 (66.9%) | 1.1 → 85/127 (66.9%) | +0 (+0.0 pp) |
| Dream-Coder · HumanEval · merge | 0.25 → 9/18 (50.0%) | 0.2 → 9/18 (50.0%) | +0 (+0.0 pp) |
| Dream-Coder · HumanEval · addition | 0.4 → 10/18 (55.6%) | 0.5 → 11/18 (61.1%) | +1 (+5.6 pp) |
| DiffuCoder-SFT · HumanEval · merge | 0.2 → 14/33 (42.4%) | 0.4 ↗ → 16/33 (48.5%) | +2 (+6.1 pp) |
| DiffuCoder-SFT · HumanEval · addition | 0.15 → 7/33 (21.2%) | 0.6 → 17/33 (51.5%) | +10 (+30.3 pp) |
| DiffuCoder-SFT · MBPP · merge | 0.05 → 12/46 (26.1%) | 0.2 → 15/46 (32.6%) | +3 (+6.5 pp) |
| DiffuCoder-SFT · MBPP · addition | 0.4 → 15/46 (32.6%) | 0.2 → 15/46 (32.6%) | +0 (+0.0 pp) |
| DiffuCoder-RL · MBPP+ · merge | 0.3 → 8/39 (20.5%) | 0.5 → 18/39 (46.2%) | +10 (+25.6 pp) |
| DiffuCoder-RL · MBPP+ · addition | 0.4 → 1/39 (2.6%) | 0.8 ↗ → 4/39 (10.3%) | +3 (+7.7 pp) |
| DreamReasoner · MATH-500 @8k · merge | 0.2 → 8/14 (57.1%) | 0.3 → 12/14 (85.7%) | +4 (+28.6 pp) |
| DreamReasoner · MATH-500 @8k · addition | 0.5 → 9/14 (64.3%) | 0.7 ↗ → 10/14 (71.4%) | +1 (+7.1 pp) |
| DreamReasoner · MATH-500 @4k · merge | 0.2 → 14/21 (66.7%) | 0.4 ↗ → 19/21 (90.5%) | +5 (+23.8 pp) |
| DreamReasoner · MATH-500 @4k · addition | 0.3 → 13/21 (61.9%) | 0.4 ↗ → 15/21 (71.4%) | +2 (+9.5 pp) |
| Nemotron-3B · ifeval · merge | 0.3 → 31/82 (37.8%) | 0.5 → 36/82 (43.9%) | +5 (+6.1 pp) |
| Nemotron-3B · ifeval · addition | 0.4 → 29/82 (35.4%) | 0.8 → 36/82 (43.9%) | +7 (+8.5 pp) |
| Nemotron-8B · ifeval · merge | 0.4 → 51/88 (58.0%) | 0.4 → 51/88 (58.0%) | +0 (+0.0 pp) |
| Nemotron-8B · ifeval · addition | 0.5 → 39/88 (44.3%) | 0.9 ↗ → 43/88 (48.9%) | +4 (+4.5 pp) |
| Nemotron-8B · math-500 · merge | 0.4 → 15/28 (53.6%) | 0.4 ↗ → 15/28 (53.6%) | +0 (+0.0 pp) |
| Nemotron-8B · human-eval · merge | 0.3 → 14/28 (50.0%) | 0.4 ↗ → 18/28 (64.3%) | +4 (+14.3 pp) |
| Nemotron-8B · mbpp · merge | 0.2 → 53/88 (60.2%) | 0.2 → 53/88 (60.2%) | +0 (+0.0 pp) |
| Nemotron-8B · gsm8k · merge | 0.4 → 28/42 (66.7%) | 0.4 ↗ → 28/42 (66.7%) | +0 (+0.0 pp) |

## Best merge (accuracy-best convex point) vs best AR-help point over both modes (0.1 grid)

| series | best merge β → AR-help, acc | best AR-help point → AR-help, acc | ΔAR-help | Δacc |
|---|---|---|---|---|
| Dream-v0 · IFEval | 0.6 → 68/127 (53.5%), 66.5 | addition 1.1 → 85/127 (66.9%), 72.1 | +17 (+13.4 pp) | +5.6 |
| Dream-Coder · HumanEval | 0.25 → 9/18 (50.0%), 85.4 | addition 0.5 → 11/18 (61.1%), 82.9 | +2 (+11.1 pp) | −2.5 |
| DiffuCoder-SFT · HumanEval | 0.2 → 14/33 (42.4%), 79.3 | addition 0.6 → 17/33 (51.5%), 73.8 | +3 (+9.1 pp) | −5.5 |
| DiffuCoder-SFT · MBPP | 0.05 → 12/46 (26.1%), 79.4 | merge 0.2 → 15/46 (32.6%), 78.6 | +3 (+6.5 pp) | −0.8 |
| DiffuCoder-RL · MBPP+ | 0.3 → 8/39 (20.5%), 69.0 | merge 0.5 → 18/39 (46.2%), 69.0 | +10 (+25.6 pp) | ±0 |
| DreamReasoner · MATH-500 @8k | 0.2 → 8/14 (57.1%), 94.8 | merge 0.3 → 12/14 (85.7%), 94.0 | +4 (+28.6 pp) | −0.8 |
| DreamReasoner · MATH-500 @4k | 0.2 → 14/21 (66.7%), 91.6 | merge 0.4 ↗ → 19/21 (90.5%), 90.2 | +5 (+23.8 pp) | −1.4 |
| Nemotron-3B · IFEval | 0.3 → 31/82 (37.8%), 66.9 | merge 0.5 (= addition 0.8) → 36/82 (43.9%), 65.6 | +5 (+6.1 pp) | −1.3 |
| Nemotron-8B · IFEval / MATH-500 / MBPP / GSM8K | best merge is already the AR-help best | | +0 | 0 |
| Nemotron-8B · HumanEval | 0.3 → 14/28 (50.0%), 76.8 | merge 0.4 ↗ → 18/28 (64.3%), 71.3 | +4 (+14.3 pp) | −5.5 |

## Parent AR-help = 0 and the noise floor
- The diffusion parent's AR-help is 0 **by definition** (pool = items the parent got wrong); verified on every series (0/pool).
- That 0 is not a noise floor. Same-protocol parent re-run available for Dream-Coder HumanEval only
  (`coding_full/Dream-Coder-v0-Instruct-7B`, B300, T0.1, 512 tok; vs the Hopper parent): acc 82.9 vs 81.7,
  **AR-help 2/18 (11.1 pp)**, AR-harm 0/4. So on small sampled code pools, ~+11 pp is reachable by re-running the parent.

## What the extra AR-help is worth in accuracy (upper bound)
If every extra AR-help item were added with zero damage elsewhere:
Dream-v0 IFEval +3.1 · Dream-Coder HE +1.2 · DiffuCoder-SFT HE +1.8 / MBPP +0.8 · DiffuCoder-RL MBPP+ +2.6 ·
DreamReasoner MATH-500 +0.8 (8k) / +1.0 (4k) · Nemotron-3B IFEval +0.9 · Nemotron-8B HE +2.4, others 0 (acc points).
AR-help still **missing at the best merge** (pool items it gets wrong), in acc points of the full benchmark:
Dream-v0 IFEval 10.9 · Dream-Coder HE 5.5 · DiffuCoder-SFT HE 11.6 / MBPP 9.0 · DiffuCoder-RL 8.2 · DreamReasoner 1.2 / 1.4 ·
Nemotron-3B IFEval 9.4 · Nemotron-8B IFEval 6.8 / MATH 2.6 / HE 8.5 / MBPP 9.3 / GSM8K 1.1.
→ Turning β recovers only a small slice (≤ ~3 pts) of the 5–12 pts of AR-help left on the table; most of it is not
reachable by any single global β. That headroom would need per-item choice (routing/cascade), not a bigger β.
