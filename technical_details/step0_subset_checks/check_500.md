# 100-of-500 random subset vs full 500 (2026-10-12)

## Fixed subset (seed 0, ids in subset100_of500_ids.json)

| system | full 500 | subset 100 | rank full | rank sub |
|---|---:|---:|---:|---:|
| AR Qwen-Ins | 56.0 | 56.0 | 1 | 4 |
| M46<=A20@0.8 | 54.2 | 61.0 | 2 | 1 |
| Ins<=A20@0.8 | 53.0 | 51.0 | 3 | 7 |
| M46<=A20@0.6 | 53.0 | 59.0 | 4 | 2 |
| Ins<=A20@0.6 | 52.2 | 56.0 | 5 | 3 |
| A(2.0) | 51.4 | 55.0 | 6 | 5 |
| M46<=M01@0.7 | 50.4 | 50.0 | 7 | 8 |
| Ins<=M46@0.6 | 48.2 | 46.0 | 8 | 10 |
| M46<=M01@0.3 | 47.6 | 43.0 | 9 | 11 |
| M(.4,.6) | 46.6 | 42.0 | 10 | 12 |
| Ins<=M46@0.8 | 45.8 | 51.0 | 11 | 6 |
| M(0,1) | 45.6 | 47.0 | 12 | 9 |
| Ins | 39.8 | 34.0 | 13 | 13 |

## Over 2000 random 100-subsets: P(pairwise order same as full 500), by full-500 gap (pp)

| gap | pairs×draws | P(order kept) |
|---|---:|---:|
| <2 | 34000 | 56% |
| 2-4 | 32000 | 71% |
| 4-6 | 30000 | 86% |
| 6-10 | 40000 | 94% |
| >=10 | 18000 | 99% |

Note: a 100-subset has SE ≈ 5 pp on a single accuracy; only orderings with full-500 gaps ≳ 6 pp are reliably kept.
