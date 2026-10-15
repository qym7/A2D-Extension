> **Status 2026-10-15:** superseded by `technical_details/` (one file per result, with sources and deviations). Kept as the original source/deviation log of STEP0.

# Idea 1 — ANALYSIS

## Sources

- A2D §4.1 (localization: 5.3% of steps change position, 3.4% of positions change token,
  concentrated at low parent confidence) and Table 2 (the schedule×tokens 2×2: 38.4 / 42.8 / 42.2 /
  45.0 on MATH-500). Analysis subsample = first 200 (A2D §2 footnote 1).
- A2D App. `tab_hparam_sweep.tex` for the peak-then-collapse coefficient curves quoted in SPEC.md.
- `../PRIOR_NEGATIVE_RESULTS.md` — read this first.

## The one finding that shapes this idea

From the collaborator's `COMPUTATION_AWARE_MERGING_RESULTS.md`: the teacher-forced local-token
objective **does not recognise** the known-good raw merge — it calls the AR vector neutral or
slightly harmful while real GSM8K accuracy rises substantially, because a corrupted gold canvas
leaks so much future solution context that Dream-Base already predicts 98.4% of sampled targets.
His conclusion: *the main transfer effect is sequence-level, not local-token.*

Two consequences, both load-bearing:

1. **The probe must be generation.** Hence E1 substitutes tokens *inside real decoding* and scores
   the finished output. No teacher-forced loss anywhere in the decisive experiment.
2. **The distillation half of the original idea (E2) is at real risk.** "Distil the merged model at
   uncertain tokens" is a local-token objective by construction. If the effect is sequence-level,
   E2 may fail even when E1 succeeds. This is why E1 is designed to be valuable *on its own* — a
   training-free decoding policy that beats the scalar frontier is a result whether or not it can
   be compressed into one weight set.

## Why the "very large AR vector + distill" framing was weakened, and what replaced it

The user's original framing was: add a very large AR vector, then distil the diffusion model onto
those tokens, starting from the best merge. Two problems:
- the "large vector" regime is where A2D's sweeps collapse (Nemotron-LD-8B MATH-500 λ=0.8 → 11.4),
  so a distillation *teacher* there is mostly noise;
- distillation makes the method training-based, giving up A2D's training-free selling point, and
  the user themself called this "有点没意思".

The replacement keeps the sharp part — that the usable content and the damage may be **separable**
— and tests it with zero training. The large vector is used as a *consultant at uncertain
positions*, not as a teacher over trajectories. If separability holds, the distillation question
becomes interesting again (and better posed); if it does not, we learned that cheaply.

## Open questions

- **Novelty (blocking for E2).** "Consult a second model where the first is uncertain" exists in
  AR-land: contrastive decoding, DExperts, proxy-tuning, Co-LLM (learning to decode
  collaboratively), speculative-style ensembles. Not yet searched. The claim to defend is narrower:
  *the two models are a checkpoint and its own over-scaled task-vector edit, and the uncertainty
  signal is the diffusion commit schedule itself, which the decoder already computes.* Do the
  search before spending E2 compute.
- Which confidence to gate on: the committed token's softmax probability `p` (what
  `dream_step_divergence.py` logs) or the neg-entropy used for the position ranking. They are
  different axes; `lockstep_port.py` computes both (`pd` and `ned`). Start with `p`; report both.
- Should the donor also be allowed to change the **position** (schedule-port) under the same gate?
  A2D found schedule and tokens contribute nearly equally (42.2 vs 42.8), so probably yes — but add
  it as a second, separate arm, not as a confound in the first grid.
- `LTR_PROBE_5.md` (collaborator) measured the LTR index at ~0.9994 for both parent and merge, i.e.
  Dream already decodes essentially left-to-right on these prompts. If so, "uncertain positions"
  may correlate with *late* positions and the intervention could reduce to a suffix edit. Check the
  positional distribution of substitutions in S2 before claiming anything about order.

## Failed attempts

_(none yet)_
