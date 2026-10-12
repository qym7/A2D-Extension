# Online confidence-gated substitution, host = Dream-v0-Instruct (2026-10-10)

## Verdict (2026-10-10 13:15, θ=0.6 still running)
**Negative for idea 1 as a method.** Swapping a few donor tokens where Instruct is unsure lifts Instruct
(θ=0.4: 89 → 106/200, p=0.01–0.03), but no arm beats simply running a better single model: best gated = best merge alone
(106 vs 107) and below best addition alone (106 vs 114, n.s.), at 2× decoding compute (both models every step).
θ=0.6 (partial) does not change that: merge donor ties best merge alone, add1.1 donor falls below add1.1 alone.
Positive by-product: the gate does contain an over-scaled donor (add2.0 disagrees on 22–24% of positions, gated at 0.4
it equals add1.1-donor). Not run (would be needed for any positive claim): matched-rate random-position control; θ=0.8.

## θ=0.6 (partial; each arm on its own completed prefix, same problems for every row)
| donor | n done | θ=0.2 | θ=0.4 | θ=0.6 | Instruct alone | best merge alone | add1.1 alone | θ=0.6 vs Instruct | vs add1.1 alone |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| add1.1 | 86 | 46 | 45 | 43 | 39 | 48 | 51 | 13/9, p=0.52 | 4/12, p=0.08 |
| add2.0 | 100 | 49 | 47 | 52 | 47 | 54 | 58 | 17/12, p=0.46 | 4/10, p=0.18 |
| merge | 147 | 67 | 67 | 78 | 67 | 78 | 84 | 22/11, p=0.08 | 17/23, p=0.43 |
Final 200-problem θ=0.6 numbers: to be filled when jobs `yqin-i1-gi{11,20,m}-t06-h200f051` finish.


Setup: MATH-500 first 200, 1024 tokens / 1024 steps, same lockstep decoder for every row (top_p 0.9, conf-temp 0.1),
0.51 H200 per job. Gate (E1, `lockstep_gated.py --mode content_gated`): at each committed position, if the host's RAW
p(token) < θ, commit the donor's token. Donor forward runs every step (compute = 2 models always); "steps using donor"
= steps where ≥1 committed token was actually taken from the donor (`gate_per_item.jsonl`, new 2026-10-09).
Script: `details/draft_scripts/gated_ins_1024_summary.py`. Paired exact McNemar (wins / losses vs the reference).

## Reference lines (same decoder, alone)
| model | acc |
|---|---:|
| Instruct | 89/200 (44.5) |
| best merge (0.4,0.6) | 107/200 (53.5) |
| best addition (1.0,1.1) | 114/200 (57.0) |

## Gated arms
| arm | acc | vs Instruct | vs best merge | vs add1.1 alone | steps using donor / problem (of 1022; % problems ≥1) | tokens from donor / problem | token sub rate | disagree rate |
|---|---:|---|---|---|---|---:|---:|---:|
| Ins→add1.1 θ=0.2 | 95 | 8/2, p=0.11 | 18/30, p=0.11 | 17/36, p=0.01 | 0.5 (31%) | 0.5 | 0.05% | 1.4% |
| Ins→add1.1 θ=0.4 | **106** | 28/11, **p=0.01** | 21/22, p=1.0 | 19/27, p=0.30 | 3.9 (92%) | 3.9 | 0.38% | 1.4% |
| Ins→add2.0 θ=0.2 | 89 | 5/5, p=1.0 | 17/35, p=0.02 | 16/41, p<0.01 | 0.6 (36%) | 0.6 | 0.06% | 24% |
| Ins→add2.0 θ=0.4 | **106** | 35/18, **p=0.03** | 21/22, p=1.0 | 23/31, p=0.34 | 7.0 (96%) | 7.0 | 0.68% | 22% |
| Ins→merge θ=0.2 | 86 | 3/6, p=0.51 | 16/37, p=0.01 | 15/43, p<0.01 | 0.4 (32%) | 0.4 | 0.04% | 1.4% |
| Ins→merge θ=0.4 | 93 | 21/17, p=0.63 | 19/33, p=0.07 | 16/37, p=0.01 | 3.0 (88%) | 3.0 | 0.29% | 1.5% |

## Reading
1. θ=0.4 with an addition donor lifts Instruct by +17 problems (p=0.01 / 0.03) while taking only 4–7 tokens per problem
   from the donor (0.4–0.7% of tokens). It reaches best-merge level (106 vs 107, 21/22) but NOT add1.1 alone (106 vs 114, n.s.).
2. The over-scaled donor add2.0 disagrees with Instruct on 22–24% of positions (add1.1: 1.4%), yet gated at 0.4 it is as
   good as add1.1 — the gate keeps its damage out. Not yet known: add2.0 alone @1024.
3. θ=0.2 changes <1 token per problem and still flips 9–10 problems both ways: one substituted token re-routes the
   trajectory. This is the noise floor of any per-item comparison here.
4. Merge donor helps less (93, n.s.) at a similar number of substitutions.
5. Not yet run: matched-rate random-position control (the E1 kill criterion), so "the gate picks the right positions" is unproven.

Caveats: multiple θ/donor arms tested, no correction; several jobs were preempted and resumed (texts resume per problem, so
accuracies are unaffected; `gate_stats.json` of resumed arms covers only the last segment → use `gate_per_item.jsonl`).

## Breakdown (details/draft_scripts/gated_ins_1024_stats.py)
Pools vs Instruct alone: add1.1 alone fixes 39, breaks 14; both right 75, both wrong 72.
| arm | acc | AR-help (of 39) | AR-harm (of 14) | both-right lost | both-wrong won | problems using donor | donor steps / problem (median, p90, max) | flips vs Instruct: used / not used |
|---|---:|---:|---:|---:|---:|---:|---|---|
| Ins→add1.1 θ=0.2 | 95 | 5 | 0 | 2 | 3 | 62 | 0, 2, 5 | 10/62 vs 0/138 |
| Ins→add1.1 θ=0.4 | 106 | 19 | 4 | 7 | 9 | 184 | 3, 8, 21 | 39/184 vs 0/16 |
| Ins→add2.0 θ=0.2 | 89 | 3 | 0 | 5 | 2 | 72 | 0, 2, 5 | 10/72 vs 0/128 |
| Ins→add2.0 θ=0.4 | 106 | 20 | 6 | 12 | 15 | 192 | 6, 14, 28 | 53/192 vs 0/8 |
| Ins→merge θ=0.2 | 86 | 2 | 0 | 6 | 1 | 63 | 0, 1, 4 | 9/63 vs 0/137 |
| Ins→merge θ=0.4 | 93 | 15 | 4 | 13 | 6 | 177 | 2, 6, 15 | 38/177 vs 0/23 |
Problems with zero substitutions never differ from Instruct (0/138 etc.) → decoder deterministic; every flip is caused by the donor tokens.
θ=0.4 add donors recover ~half of add1.1's AR-help (19–20/39) with 4–6/14 harm, but also flip 7–12 problems both parents solve.
`gen_len` is 1024 for every row (canvas length incl. padding) — not a length measure.

## 2026-10-10 student/teacher 汇总（已跑完 200 题的 arm）
命名：student = 当 base 的模型（host），teacher = 低置信位置被换进来的模型（donor）。
替换率 1 = 用了 teacher 的位置（student p<θ）；替换率 2 = teacher 给的 token 与 student 不同。均按已提交 token 计。
add2.0 单独跑（200 题）= 117，对 add1.1 114 为 25/22, p=0.77（1024 下并不 over）。

| student → teacher | θ | gated /200 | student /200 | teacher | gated vs teacher（赢/输, p） | 替换率 1 | 替换率 2 |
|---|---|---:|---:|---:|---|---:|---:|
| Ins → add1.1 | 0.2 | 95 | 89 | 114/200 | 17/36, 0.01 | 0.08% | 0.05% |
| Ins → add1.1 | 0.4 | 106 | 89 | 114/200 | 19/27, 0.30 | 0.98% | 0.38% |
| Ins → add2.0 | 0.2 | 89 | 89 | 117/200 | 18/46, <0.01 | 0.08% | 0.06% |
| Ins → add2.0 | 0.4 | 106 | 89 | 117/200 | 17/28, 0.14 | 1.09% | 0.68% |
| Ins → add2.0 | 0.6 | 113 | 89 | 117/200 | 18/22, 0.64 | 5.22% | 2.53% |
| Ins → merge | 0.2 | 86 | 89 | 107/200 | 16/37, 0.01 | 0.08% | 0.04% |
| Ins → merge | 0.4 | 93 | 89 | 107/200 | 19/33, 0.07 | 0.82% | 0.29% |
| Ins → merge | 0.6 | 107 | 89 | 107/200 | 20/20, 1.00 | 2.87% | 0.72% |

没有一个 arm 超过它的 teacher 单独跑。
