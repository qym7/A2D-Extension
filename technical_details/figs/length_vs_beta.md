# Output length vs addition β (2026-10-09, zero GPU)

Script `../../details/draft_scripts/length_vs_beta.py`; data `length_vs_beta.json`. Code lengths are sanitized-solution characters (a proxy: raw generations are not in these files). DreamReasoner @4k older runs did not save responses.

| series | unit | parent | best merge (β) | addition β → length (× parent) |
|---|---|---:|---|---|
| Dream-v0 · IFEval | chars | 1417 | 1798 (×1.27) β0.6 | 0.2: 1585 (×1.12) · 0.4: 1593 (×1.12) · 0.6: 1697 (×1.20) · 0.7: 1789 (×1.26) · 0.8: 1923 (×1.36) · 0.9: 1994 (×1.41) · 1.0: 2065 (×1.46) · 1.1: 2270 (×1.60) · 1.2: 2525 (×1.78) · 1.4: 2972 (×2.10) · 1.6: 3589 (×2.53) |
| Dream-v0 · MATH-500 | chars | 773 | 898 (×1.16) β0.6 | 1.1: 1505 (×1.95) |
| Dream-Coder · HumanEval | solution chars | 702 | 740 (×1.05) β0.25 | 0.2: 694 (×0.99) · 0.3: 679 (×0.97) · 0.4: 676 (×0.96) · 0.5: 673 (×0.96) · 0.6: 663 (×0.94) |
| DiffuCoder-SFT · HumanEval | solution chars | 609 | 585 (×0.96) β0.2 | 0.15: 615 (×1.01) · 0.2: 615 (×1.01) · 0.25: 610 (×1.00) · 0.35: 612 (×1.01) · 0.4: 602 (×0.99) · 0.5: 586 (×0.96) · 0.6: 545 (×0.89) · 0.7: 470 (×0.77) |
| DiffuCoder-SFT · MBPP | solution chars | 324 | 341 (×1.05) β0.05 | 0.15: 332 (×1.03) · 0.2: 333 (×1.03) · 0.25: 341 (×1.05) · 0.35: 336 (×1.04) · 0.4: 331 (×1.02) · 0.6: 323 (×1.00) |
| DiffuCoder-RL · MBPP+ | solution chars | 119 | 135 (×1.13) β0.3 | 0.4: 119 (×1.00) · 0.5: 119 (×1.00) · 0.6: 119 (×1.00) · 0.8: 121 (×1.02) |
| DreamReasoner · MATH-500 @8k | chars | 3134 | 3203 (×1.02) β0.2 | 0.2: 3460 (×1.10) · 0.3: 3600 (×1.15) · 0.4: 3726 (×1.19) · 0.5: 4016 (×1.28) · 0.6: 4484 (×1.43) · 0.7: 4855 (×1.55) |
| DreamReasoner · MATH-500 @4k | chars | n/a | n/a β0.2 | 0.1: 2782 · 0.2: 2929 · 0.3: 3091 · 0.4: 3262 |
| Nemotron-3B · ifeval | tokens | 643 | 830 (×1.29) β0.3 | 0.1: 597 (×0.93) · 0.2: 598 (×0.93) · 0.3: 517 (×0.80) · 0.4: 507 (×0.79) · 0.5: 474 (×0.74) · 0.6: 550 (×0.86) · 0.7: 489 (×0.76) · 0.8: 500 (×0.78) · 0.9: 466 (×0.72) |
| Nemotron-8B · ifeval | tokens | 1403 | 1414 (×1.01) β0.4 | 0.1: 1191 (×0.85) · 0.2: 1024 (×0.73) · 0.3: 1008 (×0.72) · 0.4: 1108 (×0.79) · 0.5: 1061 (×0.76) · 0.6: 974 (×0.69) · 0.7: 1016 (×0.72) · 0.8: 970 (×0.69) · 0.9: 1073 (×0.77) |

Reading: addition lengthens **free-form text from Qwen-lineage donors** monotonically in β (Dream-v0 IFEval ×1.12 → ×2.53;
Dream-v0 MATH-500 ×1.95 at β=1.1; DreamReasoner ×1.10 → ×1.55). Code solutions do not get longer (flat or shorter).
Nemotron (Ministral donor) gets **shorter** under addition (×0.7–0.9), while its convex merge gets longer on 3B (×1.29).
