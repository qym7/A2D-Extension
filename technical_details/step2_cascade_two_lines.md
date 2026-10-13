# STEP2 — Are the two clean pairs complementary? A zero-GPU cascade check (2026-10-13)

Question (user): can Ins ⇐ A(2.0) (finish-rate line) and M(.4,.6) ⇐ M(0,1) (precision line) be combined?
Everything below is recomputed from saved outputs (`details/logs/.../t08*`), MATH-500, @1024 lockstep, θ = 0.8 on both.

## 1. Where each one wins (last 300)
| | count | of which |
|---|---:|---|
| only Ins ⇐ A(2.0) right | 41 | M(.4,.6) ⇐ M(0,1) **unfinished** on 22 |
| only M(.4,.6) ⇐ M(0,1) right | 37 | Ins ⇐ A(2.0) finished-but-wrong on 32 (unfinished on only 5) |
| both right / either right | 35.3 % / 61.3 % | |

The two failure modes are different *and* only one is observable at decode time: the precision line fails by not finishing
(`\boxed` absent), the finish line fails by finishing with the wrong answer. So the order of a cascade is forced: precision line first,
finish line only when the first one did not finish. (The 2026-10-09 note `step2_note_cascade_idea.md` proposed exactly this
"precision-first" shape on the 512-token arms.)

## 2. Cascade = M(.4,.6) ⇐ M(0,1) @0.8, and if no `\boxed`, Ins ⇐ A(2.0) @0.8 on the same prompt
| subset | Ins ⇐ A(2.0) | M(.4,.6) ⇐ M(0,1) | M(.4,.6) ⇐ A(2.0) (best single) | **cascade** | vs Ins ⇐ A(2.0) | vs best single | fallback rate → compute |
|---|---:|---:|---:|---:|---|---|---|
| first 200 | 59.0 | 53.5 | 59.0 | **61.0** | 14/10, 0.54 | 23/19, 0.64 | 20 % → 2.40× |
| last 300 | 49.0 | 47.7 | 51.0 | **55.0** | 37/19, **0.02** | 33/21, 0.13 | 23 % → 2.46× |
| all 500 | 53.0 | 50.0 | 54.2 | **57.4** | 51/29, **0.02** | 56/40, 0.13 | 22 % → 2.44× |

AR reference: strict 56.0 (all 500) / 55.3 (last 300); lenient 71.0 / 70.7. The cascade is the first system here at or above the AR
*strict* number; it stays ≈ 14 points under the lenient one, so the "beat AR" verdict does not change.

Caveats. The rule has no free parameter and uses no gold (finish = `\boxed` present), but it was chosen after seeing the §1 split, so the
first-200 column is the only out-of-sample check (same direction, not significant at n = 200). The two arms share the whole prompt, so
the fallback cost is a full second 2× decode on 22 % of problems; a lazy teacher call would bring the whole thing to ≈ 1.4×.

## 3. What it means for the pipeline
The two lines are complementary in the specific sense that each repairs the other's *detectable* failure. For STEP3 the same
complementarity appears for free: rejection-sampled targets come from whichever system solved the training prompt, so the distilled
1× student sees both behaviours (finish within budget; precise tokens) without any decode-time cascade.

## 4. Within one line (Ins ⇐ A(β), different θ or β): not complementary in the usable sense (2026-10-13)
Last 300, Ins ⇐ A(2.0) @0.6 vs @0.8: only-@0.6 right 34, only-@0.8 right 33 — symmetric, and in those cases the other arm is
*finished-but-wrong* (unfinished on only 5/34 and 5/33), so nothing at decode time says which one to trust; cascade by finish = 51.0 / 50.7
(≈ +1.5, noise level). Same picture for A(1.1) vs A(2.0) and θ 0.6 vs 0.8 on the first 200 (cascades 55–60 around the 57.5–59 singles).
The oracle union is large (66.0 on the last 300) but it is decoding noise: two near-copies of one system flip different items.
Agreement is a confidence signal (same `\boxed` on 128/300 → 84 % acc; otherwise 23 %) but not a selector; routing "@0.6/@0.8 agree → take it,
else M(.4,.6) ⇐ M(0,1) with finish fallback" gives 56.0 at ≈ 4.9× — no better than the plain two-line cascade (55.0 at 2.5×).

Conclusion: complementarity comes from the **teacher's axis** (addition line → finish rate; merge line → precision), i.e. from a different
*detectable* failure mode, not from more θ or β points on one line. A chain Ins ⇐ A(β1) ⇐ A(β2) cannot add information either: a gate never
exceeds its teacher, A(β) peaks at 2.0 and A(3.0) is collapsed (M(.4,.6) ⇐ A(3.0) @0.8 = 42.3 = student). The system is expandable by adding
*teachers with a new detectable failure mode*, not by adding thresholds.

## 5. Overlap of the β arms on the Ins line (2026-10-13, last 300, Ins ⇐ A(β) @0.8)
| pair | both right / either right (Jaccard) | only first / only second |
|---|---|---|
| β1.5 vs β2.0 | 117 / 175 (0.67) | 28 / 30 |
| β1.5 vs β2.5 | 111 / 175 (0.63) | 34 / 30 |
| β2.0 vs β2.5 | 111 / 177 (0.63) | 36 / 30 |
| reference: β2.0 @0.6 vs @0.8 | 114 / 181 (0.63) | | 

All three right: 97; any right (oracle): 191 = 63.7 %; exactly one right: 46; exactly two: 48. **Majority vote of the three = 145 = 48.3 %**, i.e. no better than any single arm (48.3 / 49.0 / 47.0).
Reading: the sets are far from identical (a third of the union is arm-specific), but the only-right counts are symmetric and the overlap is the same as between two θ values of one arm — the differences look like decoding noise (a token flip early in a chain changes the whole answer), not like complementary knowledge. A vote cannot exploit it, and there is no detector for which arm is right on a given problem (same conclusion as §4 for θ). The 63.7 % oracle is the ceiling that only a per-problem detector could reach.

## 6. Cheaper cascades with Ins as the only student (2026-10-13, last 300; user: starting from the best merge is too expensive)
Ins ⇐ M(0,1) has not been run. What exists: Ins ⇐ M(.4,.6) (@0.6 = 44.7, @0.8 = 41.7) and M(0,1) alone (44.0, 77/300 unboxed).

| pair (vs Ins ⇐ A(2.0) @0.8 = 49.0) | both right / either (J) | only A-gate / only other |
|---|---|---|
| Ins ⇐ M(.4,.6) @0.6 | 106 / 175 (0.61) | 41 / 28 |
| Ins ⇐ M(.4,.6) @0.8 | 104 / 168 (0.62) | 43 / 21 |
| M(0,1) alone | 103 / 176 (0.59) | 44 / 29 |
| M(.4,.6) ⇐ M(0,1) @0.8 (reference, §1) | 106 / 184 (0.58) | 41 / 37 |

Cascades (first system's answer if it has `\boxed`, else the second's):
| first → fallback | acc | fallbacks | decode cost |
|---|---:|---:|---:|
| M(.4,.6) ⇐ M(0,1) @0.8 → Ins ⇐ A(2.0) @0.8 (§1) | 55.0 | 69 | 2 + 2·0.23 ≈ 2.5× |
| **M(0,1) alone → Ins ⇐ A(2.0) @0.8** | **53.0** | 77 | 1 + 2·0.26 ≈ **1.5×** |
| Ins ⇐ M(.4,.6) @0.6 → Ins ⇐ A(2.0) @0.8 | 51.3 | 60 | 2 + 2·0.20 ≈ 2.4× |
| Ins ⇐ A(2.0) @0.8 → M(0,1) | 51.0 | 22 | 2 + 0.07 ≈ 2.1× |
| Ins ⇐ A(2.0) @0.8 → Ins ⇐ M(.4,.6) @0.6 | 50.3 | 22 | ≈ 2.1× |

Reading: the order matters — the merge-line system must go first, because its failure mode (no `\boxed`, truncation) is the detectable one, and the addition-line gate finishes 93 % of the time, so it is the right fallback. The over merge *alone* as the first stage already gives +4 over the single gate at 1.5× total cost; its 77 truncations are exactly the problems handed to the gate. Whether Ins ⇐ M(0,1) @θ as the first stage does better than M(0,1) alone is untested (prior: no — Ins ⇐ M(.4,.6) is below M(.4,.6) ⇐ M(0,1) at every θ, the merge line is a weak teacher for Ins).
