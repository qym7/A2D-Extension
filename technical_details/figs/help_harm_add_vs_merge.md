# Addition vs convex merge: AR-help / AR-harm relative to Dream-v0-Instruct (2026-10-09, zero GPU)

![](help_harm_add_vs_merge.png)

D = Dream-v0-Instruct correct, A = Qwen2.5-7B-Instruct (AR parent) correct, M = merged model correct.
Counts in `help_harm_add_vs_merge.json`; script `../../details/draft_scripts/plot_help_harm_add_vs_merge.py`.

| | AR-help inherited | new solves | AR-harm inherited | other losses | net |
|---|---:|---:|---:|---:|---:|
| IFEval, addition (1.0,1.1) | 85/127 | 37 | 11/52 | 27 | +84 |
| IFEval, convex (0.4,0.6) | 68/127 | 27 | 12/52 | 29 | +54 |
| MATH-500, addition (1.0,1.1) | 64/140 | 33 | 10/52 | 29 | +58 |
| MATH-500, convex (0.4,0.6) | 55/140 | 17 | 14/52 | 25 | +33 |

Reading: addition's extra gain comes entirely from the gain side (+9 to +17 AR-help, +10 to +16 new solves);
AR-harm inherited does not grow (11 vs 12, 10 vs 14).

Sources (all `dream_study/step2_instruct/outputs/official_eval/`), identical diffusion protocol within each benchmark:
- IFEval: `dream_instruct_full541`, `merge_0.4_0.6`, `ifeval_ins-10aD-11aA` (lm-eval, 1280 tok/steps, T0.1, top_p0.9, entropy;
  pool not recorded in dir name, the H100 run of 10aD-11aA, not `-b300ref`); AR = `ifeval_qwen_inst_ar` (AR-native).
- MATH-500: `mathins_dreamins_m500`, `mathins_m500-merge-plain`, `mathins_m500-dream-ins-10aD-11aA-plain` (512 tok, plain); AR = `mathins_m500-ar-plain`.

Caveats: β=1.1 was chosen on IFEval, so the IFEval panel is selection-biased (MATH-500 is not); no paired test on the
category counts yet; on MATH-500 many "losses" are truncation (unfinished at 512), see MAINLINE A5/A10.
