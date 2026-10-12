# Precision-first cascade — a zero-GPU check (2026-10-09)

Motivation: "decouple (τ_D, τ_A)" is too naive to be the contribution. This note looks for a simple method that
follows from the mechanism already measured in `technical_details/mainline.md` (A5, A10): `official = finish rate × acc|finish`,
and merges trade these two factors against each other.

All numbers: MATH-500, n=500, official lm-eval, 512 tokens, plain prompt unless "concise", existing samples in
`dream_study/step2_instruct/outputs/official_eval/mathins_*`. Finish = response contains `\boxed` (proxy).
Paired tests are exact McNemar (only-A-right vs only-B-right).

## 1. Two facts from the existing arms

| arm | acc | finish | acc / finish |
|---|---:|---:|---:|
| parent (plain) | 38.4 | 80.4 | 47.8 |
| **parent (concise prompt)** | **47.0** | **93.0** | 50.5 |
| convex merge (0.4,0.6) plain | 45.0 | 68.8 | 65.4 |
| convex merge concise | 49.2 | 76.4 | 64.4 |
| decoupled (0.9,1.1) | 51.0 | 92.6 | 55.1 |

1. **A prompt alone closes most of the gap.** parent+concise = 47.0; decoupled vs parent-concise is 69 vs 49, p=0.08.
   Any weight-space claim on MATH-500 must beat the prompt baseline, not plain parent.
2. **The two merges sit on opposite corners.** Convex merge = precise but often does not finish; decoupled = finishes
   but is less precise. Neither dominates.

## 2. Cascade: run the precise model; only if it gives no answer, re-run with the high-finish model

The detector (no `\boxed` in budget) is free and needs no labels.

| cascade | acc | 2nd pass on | vs decoupled alone | vs parent-concise |
|---|---:|---:|---|---|
| convex → decoupled | 53.8 | 31.2% | 49 vs 35, p=0.16 | 67 vs 33, **p=0.0009** |
| convex-concise → decoupled | **54.4** | 23.6% | 51 vs 34, p=0.08 | 64 vs 27, **p=0.0001** |
| decoupled → parent-concise | 53.0 | 7.4% | 10 vs 0, p=0.002 | — |
| oracle (decoupled ∨ parent-concise) | 60.8 | — | — | — |

Order matters: precise-first beats finish-first (convex→dec 53.8 vs dec→convex 51.8).
Compute: ~1.24–1.31× one pass, vs. ~2.8× for doubling the budget to 1024 (which gave the merge 53.5–54.0, n=200).

## 3. Caveats (why this is a lead, not a result)

- Post-hoc: cascade pairs were chosen after seeing these arms; one benchmark; one family (Dream-v0).
- vs the decoupled single model it is not yet significant (p=0.08–0.16).
- Finish proxy is `\boxed` presence, not the scorer's `[invalidanswer]`.

## 4. What would make it a contribution (all cheap)

1. Zero-GPU: same simulation on every other family/benchmark with saved samples (DreamReasoner AIME — A11 says
   28/59 harms are empty preds; GSM8K; Dream-Coder with "tests pass/compiles" as detector?).
2. Compute-matched comparison against the 1024-budget single model on the same 200 problems.
3. Predictive rule: the first stage should be the arm with the highest acc/finish, the second the arm with the highest
   finish — both measurable on a small unlabeled dev set (finish) plus a few labels (acc/finish).
