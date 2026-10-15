# STEP3 — recipe survey: official Dream fine-tuning code, and what the distillation literature says to watch (2026-10-15)

## 1. Official instruction-tuning code (what exists, what we reuse)
| model | code | trainer | corruption / loss | recipe |
|---|---|---|---|---|
| **Dream-7B** (Dream-Instruct) | `DreamLM/Dream`, `examples/run_sft_tulu3.sh` + `src/trainer/fsdp_sft_trainer.py` (verl-based FSDP; LoRA via `model.lora_rank`, peft) — local read-only copy: Ke Wang's `sft/DLM-AR/external/Dream` | `q_sample`: per-sequence t ~ U(0,1), each response token masked with prob t, prompt never masked; CE on masked tokens; **cart** time reweighting (context-adaptive, symmetric-geometric over distance to unmasked tokens, `cart_p` 0.1); trailing EOS treated as one token | Tulu-3 (+ SmolLM2 in the paper), base = Dream-v0-Base, **lr 2e-6**, max len 2048, micro-batch 8, 3 epochs, per-batch cutoff `random_with_input_pad` (truncate every response in a micro-batch to one random response length), grad-ckpt |
| **Dream-Coder-7B-Instruct** | `DreamLM/Dream-Coder`, `instruct/examples/training/run_sft.sh` (same trainer) | same (cart) | Ling-Coder-SFT 5 M, **lr 5e-6**, batch 576, micro 6, max len 4000, 7 epochs, per-batch cutoff; paper adds a decaying [PAD]-logit penalty at inference against early termination |
| **DreamReasoner-8B** | `DreamLM/DreamReasoner` | — | — | **no training code released** (inference only; block diffusion, block 32, curriculum in the paper 2606.19257) |

Ke Wang's completed runs with this trainer (Dream-Base on 100 k Tulu-3 / NuminaMath, 4 × B300, 3.4 h) confirm it works in our cluster with the
`.venv-sft` he built. Two things in it matter for us even though we do not use its loss: (i) the **logit shift** (token at p is predicted from
hidden p−1), which our decoder and prelim already use; (ii) the per-batch cutoff — the official recipe trains on **truncated** responses on purpose
(Ke's note: with it off, 37 % of NuminaMath responses were cut mid-solution and the model learned early EOS; with it on at micro-batch 4 the
first run produced 36 % empty outputs). Our suffix cut is a truncation of the same kind, so finish rate must be watched (plan §4).

Reuse decision: our trainer (`code/step3/train_frontier_kd.py`) is a 120-line single-GPU script (LoRA student + frozen teacher side by side),
not the verl trainer — the verl trainer needs 4 GPUs and FSDP for a loss we are replacing anyway; its LoRA path and logit convention are the
parts we copied. If STEP3 succeeds and we go to full fine-tuning, the verl trainer is the drop-in (its `q_sample` call is the one line to swap).

## 2. Closest distillation recipes for diffusion LMs (what to copy, what to differ on)
| paper | teacher → student | states | positions | divergence | notes |
|---|---|---|---|---|---|
| **TOPD** trace-based on-policy distillation, 2607.16872 | frozen *diffusion* teacher (TraDo-8B-Instruct) → SDAR-4B / LLaDA-8B | student's own denoising trajectory (dynamic decoding, τ 0.9, T 1.0) | **trace-aligned**: a decision (j, x_j) at step t gets loss only if the final response keeps x_j at j | **reverse KL** (student‖teacher), sampled-token score-function estimator; full vocab for SDAR, top-k 10 for LLaDA | lr 2e-7 constant, batch 16, 30 rollout rounds, 8.5 k MATH prompts, ~5 h; ablations: on-policy > off-policy > semi-AR SFT; trace-aligned > random mask (75.2 vs 74.3); **reverse 75.9 > JSD 75.3 > forward 72.7** |
| **TAD** temporal-aware trajectory self-distillation, 2605.09536 | teacher = same model conditioned on the reference answer, one token per step | teacher trajectory states | near positions (within δ of being decoded) → **CE** on the teacher token; distant positions → **KL** to the teacher distribution | mixed | LLaDA avg 46.2 → 51.6 |
| **T3D** 2602.12262 | full-step → few-step self-distillation | trajectory | — | trajectory matching + reverse-KL DDO | few-step focus, not ours |
| OPDLM 2606.06712, TRIMS 2604.00666, dOPSD 2607.04428 | AR teacher / AR-guided masking / later-step self | rollout prefixes | masked positions | forward KL / CE | already in `dlm-train-inference-mismatch-literature` |
| **Unmasking OPD** 2605.10889 (AR) | — | — | — | — | distillation gradients align with the ideal update on *incorrect* rollouts; on rollouts the student already gets right the teacher signal is noise; "no universally good configuration" |

Where our recipe sits: TOPD is the nearest neighbour (frozen diffusion teacher, on-policy states, token-level KL). Differences that are ours:
the teacher is an **over-merged model on the student's own line** (no stronger model needed; TOPD needed an RL-trained teacher of ≥ the
student's size), and the loss is restricted to the **student's low-confidence frontier** (TOPD: every trace-aligned decision; TAD: all masked
positions). The "unmasking" finding supports the restriction: the useful signal is where the student is unsure, not where it is already right.

## 3. Tricks and pitfalls that apply to us (and what the trainer does about each)
| item | evidence | in the trainer |
|---|---|---|
| **KL direction** | TOPD: reverse > JSD > forward by 3 points on MATH500; forward matches the teacher's full support in noisy states and explodes on "ignorant" tokens; reverse is mode-seeking | `--kl reverse` default; `forward` and `jsd` kept as options (one ablation arm, see plan) |
| **top-k truncation** | BiLD top-8, TOPD top-10 (LLaDA), "k = 10 captures nearly all mass" for instruct models; the tail is noise | `--top-k` (default 0 = full; 10 is the first value to try if the full-vocab run is noisy) |
| **temperature** | τ ≤ 2 similar, τ ≥ 5 hurts (pre-training KD); selection must stay on raw T = 1 confidence because that is the gate's signal | `--kd-temp` (1.0); θ-selection always on raw p |
| **positions** | TOPD trace-aligned > random; TAD near = CE / far = KL; our prelim: frontier reproduces the gate, random masking does not | suffix cut, first K = 4 masks, p < θ; C4 = random mask control |
| **on-policy states** | TOPD on-policy > off-policy; our prelim measured on decode-time states only | prefixes = student's own rollouts (one pass, reused) |
| **early EOS / finish rate** | official recipe's cutoff story (36 % empty outputs); Dream-Coder's pad penalty; over teachers A(≥2.5) do not finish | λ·CE on the student's own token; finish rate is a reported column with a bar (plan §4); A(1.5) not A(2.5) as teacher |
| **learning rate** | official SFT 2e-6 (full FT, 7B); TOPD 2e-7 (full); LoRA usually 10–100× higher | LoRA 1e-4 with 10 % warm-up + linear decay; smoke checks the loss curve before the real run |
| **gradient on sparse positions** | ≈ 2 of 4 frontier positions per state are selected → few tokens per optimizer step | batch 32 states, 2 cuts per canvas; `sel_rate` and `empty_states` logged; if < 1 token/state on average, raise K or θ |
| **logit shift** | Dream predicts position p from hidden p−1 | `shifted_logits` (same as decoder/prelim) |
| **teacher sees the same corrupted state** | TOPD, OPDLM | yes (same `x`) |
| **don't train where the student is already right** | Unmasking OPD | that is the θ gate; C1 (all frontier) is the control that tests it |
| **overfitting after the peak** | TOPD: mild static overfitting past the best checkpoint | 2 epochs only; MATH last-300 read on the final checkpoint (no checkpoint selection on the test set) |

## 4. Changes made to the plan from this survey
- Default divergence **reverse KL** (was forward); forward kept as an ablation arm (`C5`, cheap: same teacher, one flag).
- `--top-k` and `--kd-temp` added, defaults off/1.0; not swept unless the full-vocab run is unstable.
- Everything else in `step3_plan_frontier_kd.md` stands.

Sources: [DreamLM/Dream](https://github.com/DreamLM/Dream) (`examples/run_sft_tulu3.sh`), [Dream-Coder paper](https://arxiv.org/abs/2509.01142),
[DreamReasoner repo](https://github.com/DreamLM/DreamReasoner) / [paper](https://arxiv.org/abs/2606.19257), [TOPD](https://arxiv.org/abs/2607.16872),
[TAD](https://arxiv.org/abs/2605.09536), [T3D](https://arxiv.org/abs/2602.12262), [Unmasking On-Policy Distillation](https://arxiv.org/abs/2605.10889),
[MiniLLM](https://arxiv.org/abs/2306.08543), [GKD](https://arxiv.org/abs/2306.13649), [pre-training distillation design space (top-k/temperature)](https://arxiv.org/abs/2410.16215).
