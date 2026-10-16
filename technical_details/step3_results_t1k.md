# STEP3 results — run `step3-t1k` (MATH-500, 200 shared problems of the last 300, lockstep @1024, every row decoded by ONE model except the two gates)

| row | acc | finish | acc\|finished | avg len (tok) | vs Ins | vs C2 | vs M(.4,.6) | vs A(2.0) | GSM8K | HumanEval+ | IFEval | HW |
|---|---:|---:|---:|---:|---|---|---|---|---:|---:|---:|---|
| Ins (student, untouched) | 38.5 | 77% | 50.0 | 356 | — | 26/25, p=1.00 | 17/26, p=0.22 | 22/41, p=0.02 | — | — | — | B300 |
| M(.4,.6) best merge | 43.0 | 84% | 50.9 | 433 | 26/17, p=0.22 | 28/18, p=0.18 | — | 23/33, p=0.23 | — | — | — | B300 |
| A(2.0) over addition (= the teacher) | 48.0 | 94% | 50.8 | 1009 | 41/22, p=0.02 | 36/16, p=0.01 | 33/23, p=0.23 | — | — | — | — | H200 |
| gate Ins ⇐ A(2.0) @0.8 (2×) | 50.5 | 93% | 54.3 | 792 | 38/14, p=0.00 | 40/15, p=0.00 | 30/15, p=0.04 | 27/22, p=0.57 | — | — | — | H200 |
| **M** = Ins ⇐ A(2.0) distilled, θ=0.8, λ_anchor=0.3, reverse KL, lr 0.0001, 2 ep, LoRA r64 (1×) | 50.5 | 94% | 54.0 | 644 | 39/15, p=0.00 | 40/15, p=0.00 | 32/17, p=0.04 | 28/23, p=0.58 | — | — | — | H200 |
| **C2** = Ins self-SFT (CE on its own rollout token, no teacher), lr 0.0001, 2 ep, LoRA r64 (1×) | 38.0 | 81% | 46.9 | 857 | 25/26, p=1.00 | — | 18/28, p=0.18 | 16/36, p=0.01 | — | — | — | H200 |

McNemar cells: wins-of-row / wins-of-column, exact two-sided p. Reference rows come from `outputs/step0` (single models) and `outputs/step1` (gates). Ins and M(.4,.6) were decoded on B300, everything else on H200·0.51 — one hardware column, no cross-pool claim without an H200 re-run of the B300 rows.
