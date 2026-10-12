# Dream-v0: where is AR-help highest? (2026-10-09, zero GPU)

![](ar_help_max_dreamv0.png)

Script `../../details/draft_scripts/ar_help_max_dreamv0.py`; per-point table `ar_help_max_dreamv0.json`.
Every same-protocol point on disk: IFEval 42 points (lm-eval, 1280 tok/steps, T0.1, top_p 0.9, entropy, n=541, prompt-strict
per item; `-b300ref`, `-h200`, steps256, subsets excluded), MATH-500 7 points (plain, 512 tok/steps, n=500).
AR-help inherited = Dream-Instruct wrong, Qwen2.5-7B-Instruct right, merge right.

## Result

| | max AR-help point | AR-help | AR-harm | acc | max-acc point |
|---|---|---:|---:|---:|---|
| IFEval | **(τ_D 1.0, τ_A 1.1)** | 85/127 (67%) | 11/52 | 72.1 | same point |
| MATH-500 | **(0.9, 1.1)** | 71/140 (51%) | 14/52 | 51.0 | same point |

- **AR-help maximum = accuracy maximum on both benchmarks.** Across points (parent excluded), Spearman(AR-help, acc) =
  **0.86** on IFEval (41 points) and 1.00 on MATH-500 (6 points); Spearman(AR-harm, acc) = −0.66 / −0.70.
  AR-help spans 28–85 on IFEval, AR-harm only 10–27: accuracy differences between points are mostly help, not harm.
- The max sits on the high-τ_D row, **off the convex line**; the convex line tops out at 71 (τ_A 0.7–0.8).
  τ_D = 0 rows never exceed 55 help even at τ_A 1.0: keeping τ_D is what lets the AR help be used.
- Even the best point uses only 2/3 of the IFEval help pool and 1/2 of the MATH-500 pool.

## Caveats
- IFEval accuracy here is prompt-strict (paper's 4-avg peak is (0.9,1.1) = 79.6; on prompt-strict (1.0,1.1) leads by 1.3).
- The (1.0,1.1) / (0.9,1.1) points were found by searching IFEval → IFEval panel is selection-biased; MATH-500 only 7 points.
- Grid is not uniform (dense near τ_A≈1, τ_D 0.65–1.0), so "max" is max over what was run.
