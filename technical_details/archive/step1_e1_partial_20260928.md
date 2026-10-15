# Idea 1 E1 — partial results snapshot, 2026-09-28

Resume-safe accuracies computed from each arm's `texts.jsonl` (deduplicated by problem id),
NOT from `run.log`: the runner skips already-finished problems on resume, so its in-run
counter resets and is not comparable across preempted jobs.

Host = A2D merge `dream-ins-04a_D-06a_A` (n=500 reference value 45.0). Donor = `D_b + a*tau_A`.
Gate: take the donor's token where the host's RAW p(argmax) < theta.

| arm | alpha | theta | problems done | accuracy % |
|---|---|---|---:|---:|
| a09-t00 | 0.9 | 0 | 219 | 49.3 |
| a09-t01 | 0.9 | 0.1 | 60 | 53.3 |
| a09-t02 | 0.9 | 0.2 | 60 | 55.0 |
| a09-t03 | 0.9 | 0.3 | 220 | 49.1 |
| a09-t04 | 0.9 | 0.4 | 61 | 50.8 |
| a09-t05 | 0.9 | 0.5 | 206 | 48.5 |
| a09-t07 | 0.9 | 0.7 | 223 | 45.7 |
| a09-t09 | 0.9 | 0.9 | 217 | 43.3 |
| a09-t10 | 0.9 | 1.0 | 221 | 40.7 |
| a12-t01 | 1.2 | 0.1 | 16 | 43.8 |
| a12-t03 | 1.2 | 0.3 | 160 | 48.8 |
| a12-t05 | 1.2 | 0.5 | 199 | 47.7 |
| a12-t07 | 1.2 | 0.7 | 177 | 44.1 |
| a12-t09 | 1.2 | 0.9 | 190 | 39.5 |
| a12-t10 | 1.2 | 1.0 | 191 | 37.2 |
| a15-t03 | 1.5 | 0.3 | 186 | 46.8 |
| a15-t05 | 1.5 | 0.5 | 147 | 48.3 |
| a15-t07 | 1.5 | 0.7 | 147 | 38.8 |
| a15-t09 | 1.5 | 0.9 | 147 | 38.1 |
| a15-t10 | 1.5 | 1.0 | 143 | 37.1 |

## Reading, stated with its caveat

At matched progress the accuracy declines **monotonically** with theta for all three donors, with
no interior optimum, and the best cells are tied with theta=0 (the host alone):
alpha=0.9 gives 49.8 / 49.5 / 48.8 / 45.7 / 43.5 / 40.6 at theta = 0 / 0.3 / 0.5 / 0.7 / 0.9 / 1.0.

**The theta=0.1 / 0.2 / 0.4 cells showing 53-55% are a progress artifact, not a signal**: they have
only ~60 problems done and the early MATH-500 problems are easier (theta=0 also read 53.3% at 60).
They must reach comparable n before the (0, 0.5) region can be ruled out.

If the shape holds to n=500 this falsifies H1: the donor's tokens at positions where the host is
unconfident are not better than the host's own, so the vector's usable content and its damage are
**not** separable. That is consistent with the prior already in the A2D records: the two models
disagree at only 3.4% of committed positions, and where they disagree the consulted model is *less*
sure (sharpened p 0.89 vs 0.996 where they agree).

## Arms stopped early, deliberately

The theta=0.9 and theta=1.0 arms at all three alphas were stopped at n≈150-220 while B300 was
saturated. Their conclusion (6-12 points below theta=0) is far outside noise at that n, and the
capacity was better spent on the theta in (0, 0.5) arms, which is where the decision actually lies.
Their numbers above are final-as-measured, at the n shown.
