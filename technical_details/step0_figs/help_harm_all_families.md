# Addition vs merge, AR-help / AR-harm, all families (2026-10-09, zero GPU)

![](help_harm_all_families.png)

Script `../../details/draft_scripts/help_harm_all_families.py` (file map inside); counts `help_harm_all_families.json`.
D = diffusion parent, A = AR parent (native AR protocol), M = merged model. Bars are % of items, numbers are counts.
"merge" = best convex merge in the paper (Nemotron: protected `D_p−0.3a_D+0.3a_A`); "addition" = best direct addition
`D_p+β·τ_A` (Nemotron: protected `D_p+0.4a_A`). All accuracies recompute to the documented values except noted.

## Reading

1. **Addition wins only on Dream-v0** (IFEval +30 net, MATH-500 +25 net vs merge), where the convex optimum needs large β
   and so crushes τ_D. Everywhere else merge ≥ addition — consistent with MAINLINE A2's prediction.
2. **Where addition loses, it loses through the purple bar, not the orange one**: items that *both* parents solve.
   DiffuCoder-SFT HumanEval 15 vs 4, Nemotron-3B HumanEval 19 vs 12, MBPP 25 vs 15. Inherited AR-harm is about the
   same for both modes in every cell (largest gap 4 items). So addition's failure mode is damaging the host, not
   importing the donor's mistakes.
3. **AR-help inherited is similar between modes** except Dream-v0 (addition +9 to +17) and DiffuCoder-RL (addition β=0.4
   is almost identical to the parent: 1 change in 378 — τ_A is tiny there).
4. Nemotron-3B IFEval is the only cell with a big AR-harm pool (113; the AR donor is worse than the host) and both modes
   inherit ~25 of it.

## Caveats

- Pools mixed inside families: Dream-Coder/DiffuCoder arms are Hopper (H100/H200 mixed, some unrecorded);
  DreamReasoner B300; Nemotron AR donor B300 sampled T0.7 seed 0. Coding at T0.1–0.2 is sampled → item-level noise.
- DreamReasoner AR (Qwen3) is avg@4 sampled; here A=1 if ≥2/4 correct (gives 83.8 vs documented avg 81.15).
- Nemotron IFEval uses prompt-strict per item (the documented 4-way average has no per-item form).
- β for addition / merge were chosen on the reported benchmarks in several cells → selection bias in the accuracies.
- Missing cells: DiffuCoder-RL HumanEval, DreamReasoner AIME, Nemotron other benchmarks / 8B (no addition-arm files).
