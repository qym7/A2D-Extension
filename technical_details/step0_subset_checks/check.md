# Random 50-problem subset vs full 200 (MATH-500 first 200, @1024)

subset (seed 0): [0, 3, 10, 15, 18, 23, 24, 25, 35, 37, 52, 55, 62, 64, 66, 72, 77, 79, 80, 84, 85, 90, 91, 98, 102, 103, 107, 111, 113, 120, 122, 124, 126, 129, 130, 133, 136, 140, 141, 143, 149, 154, 156, 158, 163, 175, 185, 190, 194, 195]

| system | full 200 (%) | subset 50 (%) |
|---|---:|---:|
| add2.0 | 58.5 | 54 |
| add1.1 | 57.0 | 42 |
| Ins→add2.0 θ0.6 | 56.5 | 52 |
| merge | 53.5 | 42 |
| Ins→merge θ0.6 | 53.5 | 54 |
| Ins→add1.1 θ0.4 | 53.0 | 46 |
| Ins→add2.0 θ0.4 | 53.0 | 54 |
| Ins→add1.1 θ0.2 | 47.5 | 40 |
| Ins→merge θ0.4 | 46.5 | 44 |
| Ins | 44.5 | 36 |
| Ins→add2.0 θ0.2 | 44.5 | 38 |
| Ins→merge θ0.2 | 43.0 | 36 |

seed-0 subset: Spearman 0.71, pairwise order agreement 75%, top system kept: True
2000 random subsets: Spearman median 0.83 (5–95%: 0.48–0.95); pairwise agreement median 79% (5–95%: 60–90%); top kept 37%
