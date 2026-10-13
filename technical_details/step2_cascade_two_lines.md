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
