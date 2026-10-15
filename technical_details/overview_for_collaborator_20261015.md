# Idea 1 "over distillation" — where we are (2026-10-15), for the collaborator

Repo: `qym7/A2D-Extension` (this folder). Numbers: Dream-v0-7B family, MATH-500, **last 300 problems (ids 200–499)** unless marked "all 500",
H200·0.51 unless marked †(B300). Notation: Ins = Dream-Instruct; M(a,b) = D_b + a·τ_D + b·τ_A merge line (M(.4,.6) = A2D's best merge,
M(0,1) = "over merge"); A(β) = D_b + τ_D + β·τ_A addition line (A(2.0) = "over addition"). "S ⇐ T @θ" = student S decodes one token per
step (lockstep @1024, argmax); where its raw token probability is below θ, teacher T's token is written instead.

## STEP0 — setup and baselines (`step0_setup_baselines.md`)
- The phenomenon: moving along either line past the accuracy peak gives a model that is *more often right per finished answer* but finishes
  fewer answers (M(0,1): finish 74 %, precision 61 %) or, on the addition line, finishes more but is less precise. Single models (last 300):
  Ins 36.7, M(.4,.6) 42.0, M(0,1) 44.0, A(2.0) 46.7; AR Qwen2.5-7B-Instruct 55.3 strict / 70.7 lenient.
- Code: `code/step0/` (line sweeps, help/harm plots, motivation figure `fig_motivation*.py`, subset-size checks, DreamReasoner).

## STEP1 — the gate and its mechanism (`step1_methods.md`)
- Gate results: M(.4,.6) ⇐ A(2.0) @0.8 = **51.0** (all 500: 54.2); Ins ⇐ A(2.0) @0.8 = 49.0; Ins ⇐ A(1.5) @0.8 = 48.3; M(.4,.6) ⇐ M(0,1) @0.7† = 48.0.
  The gate reaches the teacher's level or a bit above, at 1.14× compute (teacher called lazily only at low-confidence steps, ≈ 10 % of steps,
  ≈ 3–5 % of tokens actually change).
- Mechanism (pre-registered controls, all met): random positions at the same dose = student (the confidence signal knows *where*); teacher's decode
  order alone = student; order + content ≤ content. So: "student keeps its order, teacher supplies content at the student's unsure steps".
  Teacher must be "finishes but imprecise" (A(2.0)) — A(3.0) is collapsed, M(0,1) cannot finish.
- Code: `code/step1/lockstep_gated.py`, `lockstep_gated_v2.py` (the decoder: modes content_gated / both / schedule / random controls /
  content_mean / content_maxconf), `merge_shards.py`, `main_table.py`, arness/EOS/vote analyses. Runners in `../modules/run_gated_shard*.sh`.

## STEP2 — multi-model / logit-space exploration → the distillation design (`step2_verdict.md`, `technical_details/step2_*.md`)
- β sweep on Ins ⇐ A(β) @0.8: 1.0 → 44.3, 1.5 → 48.3, 2.0 → 49.0, 2.5 → 47.0: the teacher must be *over* the peak; plateau from 1.5; rule = smallest β past the peak.
- **Every token-level way of using a second teacher equals a single teacher**: soft mixing (46.3), teacher mean (killed by prior), most-confident of
  {A(2.0), M(0,1)} (48.7), majority vote of three β arms (48.3 = single). Their correct sets overlap like two θ values of one arm (Jaccard 0.63–0.67).
- The only inference-time multi-model gain is an *abstention cascade* (precision-ordered, unfinished = wrong): 53–57 at 1.5–2.5×.
- AR vs diffusion: AR holds the ceiling (lenient 70.7); AR-wrong & diffusion-right is 23 real problems; an AR-first cascade (66.7–68.3) < AR alone.
  Conclusion: no inference-time combination in this family beats AR on MATH; the value of the line is to fold the gate's +12 (Ins → 48–49) into a 1× model.

## STEP3 — distillation training (in progress; `technical_details/step3_plan_frontier_kd.md`, `step3_prelim_signal_check.md`, `code/step3/pipeline/`)
- Prelim (50 problems): under *random masking* (standard dLLM SFT corruption) only 11–19 % of the gate's positions are low-confidence again
  (right context leaks). Under a *prefix-visible / suffix-masked* state, the leftmost mask reproduces 96–98 % of the gate's decisions with 0.87
  precision; three A(β) teachers agree there 77 % of the time and the agreed token is the gate's token 90 %.
- Recipe: label-free prompts (17 k: Tulu-3 8 k incl. its math subsets, MATH train 6 k, Magicoder 2 k, no_robots 1 k) → student decodes once
  (official sampler) → random suffix cut → student + frozen A(1.5) forward on the same state → forward KL at the first 4 masks where the student's
  p < 0.8 (+ 0.1·CE on its own token) → LoRA r64, 2 epochs → eval at 1× with the standard sampler. Controls: self-SFT (C2), no selection (C1),
  non-over teacher (C3), random-mask corruption (C4). Success: ≥ Ins + 6 on MATH last 300, > C2 + 3, no loss on GSM8K / HumanEval+ / IFEval.
- Neighbours: OPDLM (AR teacher logits on rollout prefixes), TRIMS (AR-guided trajectory masking), dOPSD (self-distillation). Ours: an over-merged
  *diffusion* teacher on the student's own line, no AR teacher, no labels, loss only on the student's low-confidence frontier.
- Status: prompt set built; rollout, trainer (draft), Snakemake pipeline written; **no training job has run yet**. First GPU step = 50-prompt
  rollout smoke, then a 20-step training smoke. Budget first wave (M + C2, MATH only) ≈ 75 GPU·h at H200·0.51.

## Code layout (reorganised 2026-10-15: code and outputs separated, both split by step)
```
readme.md  step0_setup_baselines.md  step1_methods.md  step2_verdict.md      ← the narrative, one file per step
technical_details/step{0,1,2,3}_*.md                                        ← every number with its run, plans with pre-registered reading rules
code/step0/  code/step1/  code/step2/  code/step3/                          ← analysis, the gate decoder (step1), votes (step2), trainer + data (step3)
code/step3/pipeline/{Snakefile,config.yaml,README.md}                       ← STEP3 end-to-end orchestration (RunAI jobs via ../modules/submit_wait.sh)
outputs/checkpoints/                                                        ← A(β), M(a,b) checkpoints, shared by every step (untracked)
outputs/step0/ … outputs/step3/                                             ← per-problem texts, records, rollouts, LoRA, per step (untracked)
../modules/                                                                 ← job runners shared across ideas (run_gated_shard5.sh, run_step3_*.sh, submit.sh)
../tmp.md                                                                   ← the experiment queue (every submit / result / delete)
```
