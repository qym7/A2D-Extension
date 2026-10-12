# Idea 1 — the confidence gate is dead, and now we know why

MATH-500, n=500, deterministic argmax instrument (`lockstep_gated.py` wrapping A2D's
`lockstep_port.py`). 30 completed arms, all B300. Host = A2D's best merge
`dream-ins-04a_D-06a_A` unless stated.

## 1. The internal baseline

`a09-t00` has `substitution_rate` exactly 0.00 and scores **47.2**. Every number below is measured
against that, not against the published 45.0 — see the open item at the bottom.

## 2. Nothing beats substituting nothing

Best of all 30 arms: `e4-mrg08-t10m2` = **46.8**, at a substitution rate of **0.01%** (about 20
positions out of 226k committed). The optimum of the whole gate family is "don't substitute".

## 3. Accuracy is a function of the DOSE, not of which positions were chosen

Across the 22 old-rule arms (donors: pure-AR transfer at 0.9/1.2/1.5, and merges at lambda 0.8/1.0):

    acc = 46.54 - 2.18 * (substitution %)      R^2 = 0.83, n = 22

## 4. The gate's premise is false, measured directly

New diagnostic accumulated inside every E4 run: at committed positions where the host is unsure
**and** the two models' argmax differs, how often is the donor's raw confidence higher than the
host's?

| donor | donor more confident |
|---|---|
| `merge_l08` (lambda 0.8) | **19.5 - 21.1%** |
| `merge_l10` (lambda 1.0) | **26.5 - 29.6%** |

So in 70-80% of the candidate positions the donor is *less* sure than the host it would overwrite.
There is no reservoir of positions where the donor knows better.

## 5. Making the gate smarter makes it no better, and sometimes worse

E4 rule: substitute only when `host_raw_p < theta` **and** `donor_raw_p > host_raw_p + margin`.
Compared at matched substitution rate against the old rule (same donor):

| donor | sub% | old rule | new rule |
|---|---|---|---|
| `merge_l08` | ~0.2-0.3% | 44.6 | 44.8 (no change) |
| `merge_l10` | ~0.8% | 43.8 | **41.0 - 41.8 (worse)** |

For the pure-AR donor, selecting exactly the positions where it is *more confident* does **more**
damage than selecting positions where the host merely hesitates. The donor's confident-but-different
predictions are its confident errors. Confidence is not just uninformative here, it is
anti-informative.

## 6. Mirror image: a WEAK host DOES gain, and the clean part of it is quotable

Host = `Dream-v0-Instruct-7B` (the original un-merged instruct model, published 38.4 — the weakest
model in the set). Donor = one of the merges. Two borrowing amounts were run.

| donor | donor alone (free) | borrow EVERYTHING (full content port) | borrow a little (gate theta=0.3) | disagreement |
|---|---|---|---|---|
| `merge_l04` (lambda 0.4) | 42.8 | **44.6** | 39.4 | 1.2% |
| `dream-ins-04a_D-06a_A` (0.6) | 45.0 | 42.8 *(A2D's published number)* | 40.6 | 3.0% |
| `merge_l08` (0.8) | 41.0 | 42.8 | 40.4 | 4.7% |
| `merge_l10` (1.0) | 39.4 | 40.2 | **42.2** | 38.2% |

**The full-port column is free of the anchor problem** — every cell in it, including A2D's own 42.8,
comes from the same `run()` mode=content path on the same host, so they are directly comparable to
each other. Two clean readings:

- **Swapping A2D's donor for a lambda=0.4 merge moves the content port from 42.8 to 44.6 (+1.8), but
  the difference is NOT significant: 44 problems A-only vs 53 B-only, exact two-sided McNemar
  p = 0.42 (n=500 paired). It is noise. Do not quote it** (an earlier version of this file called it
  quotable, before the test was run). Note the lambda=0.4 donor is *weaker on its own* (42.8 vs 45.0), so this is
  not "a better partner"; it is a *closer* partner (1.2% disagreement vs 3.0%).
- **How much to borrow scales inversely with how different the donor is.** For the closest donor,
  borrow everything (44.6 > 39.4). For the most distant one (38.2% disagreement), borrowing
  everything hurts (40.2) and borrowing 0.45% is much better (42.2). This is exactly the concern the
  user raised when asking for this ablation: an over-merged donor may simply be too far away.

What still needs the anchor: any claim of the form "beats the parent's 38.4", because the gate arms
sit against a `free`-mode number and the strong host's `free` anchor came out +2.2 high. One cheap
job (parent, theta=0, n=500) fixes that.

## 7. Verdict

H1 ("AR-transferred behaviour is localized, so distil/substitute only at uncertain tokens") is
falsified, now with a mechanism rather than just a negative: the signal a diffusion LM has at decode
time (its own confidence, and the donor's) does not identify positions where the donor's content is
an improvement. Any gate built on confidence can only trade accuracy for dose.

## Open item (affects the A2D submission, not just this idea)

The theta=0 anchor is 47.2 but the host's published free-mode score is 45.0, on a deterministic
instrument, i.e. 11 problems. Cause is a code-path difference: 45.0 comes from `mode=free` (official
`diffusion_generate`), the gate runs the hand-written `run()` loop, and no A2D arm ever ran `run()`
with zero ports. A2D's published 2x2 mixes the two paths (diagonal 38.4/45.0 from `free`,
off-diagonal 42.8/42.2 from `run()`). Decisive job, ~2.5 h on 1 GPU:
`lockstep_port.py --mode content --driver <merge> --other <the same merge>` at n=500; it must return
45.0.
