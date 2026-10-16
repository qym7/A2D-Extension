# STEP3 results — run `step3-mixpool-1k` (MATH-500, 300 shared problems of the last 300, lockstep @1024, every row decoded by ONE model except the two gates)

| row | acc | finish | acc\|finished | avg len (tok) | vs Ins | vs C2 | vs M(.4,.6) | vs A(2.0) | GSM8K | HumanEval+ | IFEval | HW |
|---|---:|---:|---:|---:|---|---|---|---|---:|---:|---:|---|
| Ins (student, untouched) | 36.7 | 76% | 48.5 | 375 | — | 37/36, p=1.00 | 26/42, p=0.07 | 32/62, p=0.00 | — | — | — | B300 |
| M(.4,.6) best merge | 42.0 | 81% | 51.9 | 448 | 42/26, p=0.07 | 45/28, p=0.06 | — | 35/49, p=0.16 | — | — | — | B300 |
| A(2.0) over addition (= the teacher) | 46.7 | 94% | 49.5 | 1005 | 62/32, p=0.00 | 58/27, p=0.00 | 49/35, p=0.16 | — | — | — | — | H200 |
| gate Ins ⇐ A(2.0) @0.8 (2×) | 49.0 | 93% | 52.9 | 795 | 62/25, p=0.00 | 64/26, p=0.00 | 51/30, p=0.03 | 40/33, p=0.48 | — | — | — | H200 |
| **M** = Ins ⇐ A(2.0) distilled, θ=0.8, λ_anchor=0.3, reverse KL, lr 0.0001, 2 ep, LoRA r64 (1×) | 50.3 | 92% | 54.7 | 639 | 63/22, p=0.00 | 59/17, p=0.00 | 51/26, p=0.01 | 46/35, p=0.27 | — | — | — | H200 |
| **C2** = Ins self-SFT (CE on its own rollout token, no teacher), lr 0.0001, 2 ep, LoRA r64 (1×) | 36.3 | 80% | 45.6 | 856 | 36/37, p=1.00 | — | 28/45, p=0.06 | 27/58, p=0.00 | — | — | — | H200 |

McNemar cells: wins-of-row / wins-of-column, exact two-sided p. Reference rows come from `outputs/step0` (single models) and `outputs/step1` (gates). Ins and M(.4,.6) were decoded on B300, everything else on H200·0.51 — one hardware column, no cross-pool claim without an H200 re-run of the B300 rows.
