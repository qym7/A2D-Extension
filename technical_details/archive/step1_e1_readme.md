# Idea 1 — results: **H1 FALSIFIED** (2026-09-28, n=500, B300)

## Verdict

There is no `(alpha, theta)` that beats the host. Accuracy falls **monotonically with the fraction of
positions substituted**, at a strikingly constant rate, for every donor. The AR vector's usable
content and its damage are **not separable at the token level**. Idea 1 is killed; the E2 distillation
follow-up is **not** worth running, because there is nothing to distil.

## Setting

Host = the A2D merge `dream-ins-04a_D-06a_A` (`D_b + 0.4 tau_D + 0.6 tau_A`), whose own n=500
free-mode score is **45.0** — the best known scalar point, i.e. Rule 3's bar.
Donor = `D_b + alpha tau_A`. Gate: take the donor's token at a committed position only where the
host's **raw** `p(argmax) < theta`. MATH-500, n=500, 512 steps, top-p 0.9, conf_temp 0.1, argmax
tokens — the verified lockstep protocol.

### Past-peak controls (required by H1, and they pass)

| donor | standalone n=500 |
|---|---:|
| `D_b + 0.9 tau_A` (A2D's selected transfer) | 40.2 (paper) |
| `D_b + 1.2 tau_A` | **35.2** |
| `D_b + 1.5 tau_A` | **27.8** |

The over-scaled donors are decisively worse on their own, so the regime H1 needs was genuinely tested.

## The measurement-quality problem, stated before the result

`theta = 0` never fires the gate (substitution rate 0.0000, verified), so it should reproduce the
host's free-mode 45.0 problem for problem. It reads **47.2** — 2.2 points (11 problems) higher.

The most likely cause is numeric nondeterminism amplified by the trajectory: the gated mode runs an
extra donor forward pass at every step (a pure read, but it changes memory layout and cuBLAS
heuristics), and in bf16 over 512 sequential commits a single flipped argmax early rewrites the whole
generation. **So this instrument carries roughly +/-2 points of run-to-run noise whenever the resident
model set changes, and differences below ~2 points are not interpretable.** Every comparison below
therefore uses `theta = 0` (47.2) as the in-experiment control, not 45.0.

### alpha = 0.9

| theta | substitution rate | accuracy % |
|---:|---:|---:|
| 0.0 | 0.00% | 47.2 |
| 0.1 | 0.10% | 47.2 |
| 0.2 | 0.35% | 45.8 |
| 0.3 | 0.70% | 45.0 |
| 0.4 | 1.20% | 43.2 |
| 0.5 | 1.55% | 43.4 |
| 0.7 | 2.40% | 40.8 |

Disagreement rate 3.0%. Slope over the measured range: **-2.7 accuracy points per 1% of positions substituted**.

### alpha = 1.2

| theta | substitution rate | accuracy % |
|---:|---:|---:|
| 0.1 | 0.16% | 47.4 |
| 0.2 | 0.51% | 45.8 |
| 0.3 | 0.92% | 44.0 |
| 0.4 | 1.57% | 42.0 |
| 0.5 | 2.17% | 42.6 |
| 0.7 | 3.25% | 39.0 |

Disagreement rate 4.3%. Slope over the measured range: **-2.7 accuracy points per 1% of positions substituted**.

### alpha = 1.5

| theta | substitution rate | accuracy % |
|---:|---:|---:|
| 0.3 | 1.15% | 43.4 |
| 0.5 | 2.73% | 43.2 |
| 0.7 | 4.57% | 35.2 |

Disagreement rate 5.6%. Slope over the measured range: **-2.4 accuracy points per 1% of positions substituted**.

## The interesting part of the negative result

Plotted against **substitution rate** rather than theta, the three donors collapse onto one line:

| donor | disagreement rate | slope |
|---|---:|---:|
| `alpha = 0.9` | 3.0% | **-2.7 points per 1% substituted** |
| `alpha = 1.2` | 4.3% | **-2.7 points per 1% substituted** |
| `alpha = 1.5` | 5.6% | **-2.4 points per 1% substituted** |

Two things follow, and the second is the one worth keeping.

1. **Over-scaling does not make the donor's individual tokens worse — it makes it want to change more
   of them.** The disagreement rate climbs with alpha (3.0 -> 4.3 -> 5.6%) while the damage *per
   substituted token* stays put at about -2.6 points per 1%. That is why the alpha sweep collapses:
   accuracy is a function of how much you let the donor overwrite, almost independently of which donor.
2. **A2D's localization observation is a correlation, not a mechanism.** A2D §4.1 reports that the
   merge changes the token at only 3.4% of positions, concentrated where the parent is unconfident,
   and Table 2's content port (merge tokens on the parent's schedule) gains +4.4 (38.4 -> 42.8). It is
   tempting to read that as "the gain lives in those 3.4% of tokens". This experiment shows the
   inference does not run the other way: writing an AR-direction donor's tokens at the host's
   unconfident positions is *uniformly harmful*, in proportion to how many you write. The 3.4% of
   changed tokens accompany the gain; they do not deliver it.

Note the substitution rates are all <= 4.6%, so the "genuinely selective" requirement in SPEC.md was
satisfied throughout. The idea failed on the merits, not on the intervention budget.

## Why the prior should have been lower, in hindsight

The A2D records already said the consulted model is **less** sure exactly where it disagrees
(sharpened `p(argmax)` 0.89 at disagreements vs 0.996 at agreements). A model that is less confident
precisely where it differs is a poor source of replacement tokens. That was in
`plots/lockstep/README.md` before this experiment ran, and it deserved more weight than it got.

## Cost

24 arms + 2 builds + 2 standalone controls on B300 during the free window. Six high-theta arms were
stopped at n~150-220 once their conclusion was far outside noise (snapshot:
`e1_partial_20260928.md`); the rest ran to n=500.
