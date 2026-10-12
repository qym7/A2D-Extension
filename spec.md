# Idea 1 — Harvest over-scaled AR vectors at uncertain positions only

## The observation we are standing on

A2D §4.1 measured that the merged model is a **very local** edit of the diffusion parent: it picks
a different position at only **5.3% of denoising steps** and commits a different token at only
**3.4% of positions**, and those changes **concentrate where the parent has low confidence**.

A2D's own coefficient sweeps (paper App., `tab_hparam_sweep.tex`) show a peak-then-collapse shape:

| Family / anchor | sweep |
|---|---|
| Qwen2.5-Coder → Dream-Coder, HumanEval, transfer `D_b+ατ_A` | 63.4 (α=0) · 71.3 (0.4) · 70.7 (0.5) · **74.4 (0.6)** · 67.1 (0.7) · 62.2 (0.8) |
| Ministral-3-8B-Reasoning → Nemotron-LD-8B, MATH-500, merge | 87.2 (λ=0) · 90.8 (0.1) · **92.6 (0.2)** · 90.4 (0.3) · 90.2 (0.4) · 39.4 (0.6) · 11.4 (0.8) |
| Ministral-3-3B-Instruct → Nemotron-LD-3B, IFEval, merge | 68.0 · 71.5 (0.2) · **72.9 (0.3)** · 71.9 (0.4) · 72.5 (0.5) · 69.1 (0.6) · 51.2 (0.8) |
| Qwen2.5-7B → Dream-v0, IFEval, merge | 63.0 · 65.0 · 67.6 · 69.8 (0.4) · 70.3 · **72.8 (0.6)** · 72.6 · 71.5 (0.8) · 56.7 (1.0) |

## Hypothesis

**H1 (separability).** Past the peak coefficient α\*, the over-scaled model's *global* generation
degrades (termination, formatting, fluency — A2D's collapse rows) while its *local* preference at
positions where the host is uncertain does **not** degrade at the same rate. If so, the AR vector's
usable content and its damage are separable, and a token-level policy that consults the over-scaled
model **only** at uncertain positions can harvest content that the scalar sweep has to throw away.

This predicts something the scalar family cannot do: a model that is simultaneously as fluent as
α\* and as knowledgeable as α ≫ α\*.

## Why this is not killed by the existing negative results

`../PRIOR_NEGATIVE_RESULTS.md` establishes that **the transfer effect is sequence-level, not
local-token**: a teacher-forced masked-diffusion loss does not even recognise the known-good raw
merge. So idea 1 **must not** be probed with a teacher-forced per-token objective, and its first
experiment is not a training run at all — it is a substitution *inside real decoding*, scored by
the official scorer on the completed generation. That is the one probe shape the prior work did not
try, and it is the probe shape the sequence-level finding demands.

Second, the intervention is native to the architecture: in a dLLM the "uncertain positions" are
exactly the ones the entropy/confidence schedule commits **last**, so the selection signal is
already computed by the decoder. In AR-land the same shape requires an extra mechanism.

## Non-goals

- No teacher-forced local-token objective as a primary signal. (Presumed dead; see the map.)
- No new benchmark, prompt, or scorer. A2D anchors and native official protocols only.
- Not a decoding-efficiency paper. The E1 policy costs 2 forward passes; that is a cost we pay for
  the measurement, and the reason E2 (distillation) exists at all.
- Not another α/λ sweep. If the winning configuration is "always substitute", that *is* a sweep and
  the idea is dead.

## E1 — selective substitution during decoding (the decisive test, inference-only)

- **host** = best known checkpoint for the family (A2D merge at its selected λ; and separately the
  diffusion parent `D_p`, as a second host).
- **donor** = `D_b + α τ_A` at `α ∈ {α*, 1.5α*, 2α*}` — deliberately **past** the peak.
- At each denoising step the host proposes (position, token) with confidence `c`. If `c < θ`, take
  the **donor's** choice at that step instead; otherwise keep the host's. Both models see the same
  canvas, so this is one shared trajectory, two scorers.
- `θ ∈ {0, 0.3, 0.5, 0.7, 0.9, 1}`; `θ=0` reproduces host exactly (plumbing check), `θ=1` is a
  donor-driven decode.
- Log, per run: substitution rate (fraction of committed positions taken from the donor), the
  position/token disagreement rates (to compare against A2D's 5.3% / 3.4%), cap rate, and the
  official score.
- Variant to run only if substitution is too coarse: logit mixing
  `softmax((1−w)·z_host + w·z_donor)` restricted to the top-q% highest-entropy positions.

### Pre-registered decision rule

Idea 1 **survives** only if **all** of:
1. some `(α > α*, θ)` beats the best scalar point on the anchor by more than the family's own
   seed/pool noise band, **and**
2. the winning configuration's substitution rate is **≤ ~10%** of committed positions — i.e. it is
   genuinely selective (Rule 2: a near-100% rate is just norm reduction / a sweep in disguise),
   **and**
3. the gain survives on one non-anchor benchmark of the same family at the *same* `(α, θ)`, chosen
   before looking (A2D's own generalisation discipline).

Otherwise: **kill, write the negative result, do not proceed to E2.** In particular, if the
accuracy-vs-α curve at fixed θ tracks the plain scalar sweep, H1 is false — the damage and the
content are not separable.

## E2 — distil the policy into one weight set (only if E1 passes)

Sample trajectories from the E1 policy; fine-tune the host on those trajectories with the loss
restricted to the **substituted positions and the states in which they were substituted** — "只学
几个，不学一整个 traj". The point of E2 is (a) collapse 2 forward passes into 1, (b) test whether the
localized behaviour is *learnable* as a weight edit, which is the mechanism→method step the
contribution gate wants. E2 is a training run and needs its own protocol gate.

## Kill criterion (restated, single sentence)

If no `(α>α*, θ)` with substitution rate ≤10% beats the best scalar point on the anchor and holds on
one held-out benchmark, idea 1 is dead and the folder gets a negative-result README.

## Open novelty risk — must be closed before E2

"Consult a second model where the first is uncertain" exists in AR-land (contrastive decoding,
DExperts, proxy-tuning, Co-LLM / learning-to-decode-collaboratively, speculative ensembles).
Before any E2 compute, `ANALYSIS.md` must contain a literature check stating what is new here
beyond "the same trick, but the two models are a base and its own over-scaled task-vector edit, and
the uncertainty signal is the diffusion schedule itself".
