# Retry of idea 1 with an intact donor (2026-10-09, proposal; zero GPU so far)

## Original idea (SPEC.md, H1)
Past the best coefficient the over-scaled model degrades globally, but its local preferences where the host is
uncertain stay useful → consult the over-scaled model **only at uncertain positions** during decoding
(host = best merge, donor = over-scaled, confidence-gated substitution, θ sweep).

## Why the first test may not have tested it
- E1/E3 donors were `D_b + α·τ_A` / convex λ ≥ 1 → **τ_D = 0** (MAINLINE B, "λ 的含义" retraction). The donor had lost the
  diffusion post-training, so "donor damage" was mostly *missing τ_D*, not AR over-scaling.
- Most losses were truncation at 512 tokens (MAINLINE A6); at 1024 the dose curve reversed (θ=0.7: +3.8, p=0.09).
- New facts (figs/ar_help_gain.md, help_harm_all_families.md): with τ_D kept, a larger β keeps adding AR-help, and the
  damage it adds is mostly *losing items both parents solve* (host perturbation), which is what gating should protect.

## Headroom with today's points (oracle = host right OR donor right; host = accuracy-best, donor = AR-help-best)

| line | host | donor | donor-only right / host-only right | oracle |
|---|---:|---:|---|---:|
| DiffuCoder-SFT HE addition | 73.8 | 73.8 | 13 / 13 | 81.7 (+7.9) |
| Nemotron-3B IFEval merge | 66.9 | 65.6 | 43 / 50 | 74.9 (+7.9) |
| Nemotron-8B IFEval addition | 69.3 | 62.8 | 36 / 71 | 76.0 (+6.7) |
| Nemotron-8B HumanEval merge | 76.8 | 71.3 | 11 / 20 | 83.5 (+6.7) |
| DreamReasoner M500 8k addition | 94.4 | 94.2 | 12 / 13 | 96.8 (+2.4) |
| Dream-Coder HE addition | 83.5 | 82.9 | 5 / 6 | 86.6 (+3.0) |

Caveat: flips run both ways in near-equal numbers, and a same-protocol parent re-run already flips items
(Dream-Coder HE: AR-help 2/18 from re-running the parent). Part of every oracle gap is sampling noise; only a
**router that beats the host** counts.

## Proposed order (cheap first)
1. Zero GPU: on saved outputs, test label-free item-level routers host→donor (math: no `\boxed`/hit cap; code: no
   parsable function; IFEval: none obvious). Reports router acc vs host and vs a random router at the same switch rate.
2. GPU, Dream-v0 only (lockstep pipeline exists, `dream_study/step2_instruct/scripts/lockstep_port.py`): token-level
   gate with the **intact donor** `D_p + β·τ_A`, β ∈ {1.4, 1.6} (past the IFEval peak 1.1), host = (1.0,1.1) or convex 0.6,
   θ ∈ {0, 0.5, 0.9}, MATH-500 @1024, matched random-position control. Two models loaded → 0.5 H200.
   Kill if gate ≤ random control at matched switch rate (same criterion E1 failed).

## Online variants (need GPU; item-level rerun already ruled out offline, figs/confidence_rerun_sim.md)
- (a) per-step substitution (original E1): host step with confidence < θ → donor commits that step.
- (b) **hand-off** (new): the first time host confidence < θ, the donor takes over and finishes denoising from the
  current partial canvas. Same number of forward passes as one decode (compute-neutral), but both models loaded (0.5 H200).
- Minimal test: Dream-v0, MATH-500 @1024, n=200, host (1.0,1.1), donor intact (1.0,1.6); arms θ∈{θ_a, θ_b} for (a) and (b),
  plus a random-trigger control at matched switch rate, plus donor alone (≈6 arms). Kill: no arm beats its random control.
