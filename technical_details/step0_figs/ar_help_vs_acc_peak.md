# Does AR-help peak where accuracy peaks? All families (2026-10-09, zero GPU)

![](ar_help_vs_acc_peak.png)

Script `../../details/draft_scripts/ar_help_vs_acc_peak.py`; file map `sweeps_file_map.json` (builder
`../../details/draft_scripts/build_sweeps_file_map.py`); numbers `ar_help_vs_acc_peak.json`.
y = share of the AR-help pool (parent wrong, AR right) that the merged point gets right. ▼ = best-accuracy β on that line.

## Answer: AR-help peaks at a larger β than accuracy, almost everywhere

| series | mode | acc-best β | AR-help-best β | at grid edge? |
|---|---|---|---|---|
| Dream-v0 IFEval | merge | 0.6 | 0.7–0.8 | |
| Dream-v0 IFEval | addition | 1.1 | 1.1 (same) | |
| Dream-Coder HE | addition | 0.4 | 0.5 | |
| Dream-Coder MBPP | merge | 0.1–0.15 | 0.25 | edge |
| DiffuCoder-SFT HE | merge | 0.2 | 0.4 | edge |
| DiffuCoder-SFT MBPP | merge | 0.05 | 0.2 | |
| DreamReasoner M500 8k | merge / addition | 0.2 / 0.5 | 0.3–0.4 / 0.7 | edge / edge |
| DreamReasoner M500 4k | merge / addition | 0.2 / 0.3 | 0.4 / 0.4 | edge / edge |
| Nemotron-3B IFEval | merge / addition | 0.3 / 0.4 | 0.5 / 0.8 | |
| Nemotron-8B IFEval | merge / addition | 0.4 / 0.5 | 0.4 / 0.9 | — / edge |
| Nemotron-8B HE | merge | 0.3 | 0.4 | edge |

- 13 lines: AR-help-best β > acc-best β. Same β: Dream-v0 addition, DiffuCoder-RL merge, Nemotron-8B IFEval/MATH/GSM8K merge.
  Never smaller, except Nemotron-8B MBPP (whose per-item grading is unreliable: first-code-block regrade moves parent 62→86).
  Undecidable (ties): Dream-Coder HE merge, DiffuCoder-SFT HE/MBPP addition, DiffuCoder-RL addition (pool hit only 1–4 items).
- Typical ratio: help-best β ≈ 1.5–2× acc-best β.
- In 8 lines the help max is the largest β tried → the true AR-help peak is not found yet (your point).

Interpretation: past the accuracy optimum the merged model keeps picking up AR-help items, but loses more elsewhere
(AR-harm and items both parents solved). The accuracy optimum is a trade-off point, not where AR transfer saturates.

## Caveats
- Coding/reasoning help pools are small (14–46 items; DreamReasoner 14/21) → ±1–2 items move the argmax.
- Sampled decoding in coding (T0.1–0.2) and Qwen3 AR (avg@4, majority) add noise; grids are coarse and uneven.
- Grids were designed around accuracy, so "help-best at edge" is a search artefact, not a peak.
- Nemotron: per-item IFEval is prompt-strict; pool not recorded (docs say H200); AR donor is B300 sampled.
