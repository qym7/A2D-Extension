# STEP3 — the training process, formulated (2026-10-15)

One page. What the trainer in `code/step3/train_frontier_kd.py` actually optimises, with the symbols fixed. Plan and arms: `step3_plan_frontier_kd.md`.

## 0. Objects
| symbol | meaning | where it comes from |
|---|---|---|
| S₀ | student init: M(.4,.6) (default) or Ins | config `student` |
| S = S₀ + LoRA | the model being trained | peft, r 64 / α 128, all attention + MLP projections |
| T | frozen teacher A(β) = D_b + τ_D + β·τ_A, β = 2.0 (A(1.5) for the Ins init) | `outputs/checkpoints/add-a10-b20` |
| p_M(· \| x, j) | distribution of model M at position j given canvas x (Dream shift: position j is read from hidden j−1) | `shifted_logits` |
| c_j = max_v p_S(v \| x, j) | the student's raw confidence at j (T = 1, no top-p) — **the gate's signal** | same forward as the loss |
| θ | hand-over threshold (0.8; 0.6 arm; 0.8 → 0.5 schedule arm) | `--theta`, `--theta-end` |

## 1. Data: the student's own trajectories (paid once)
For every prompt q (17 k), S₀ decodes once with the official MATH-500 sampler (512 tokens / 512 steps, T 0.1, top-p 0.9, entropy order)
and we store the finished canvas y plus, for every response position j, the step s_j at which it was written and the raw probability
κ_j of the token written there at that step. No teacher, no ground truth, no filtering: this is pure on-policy data for S₀.

## 2. Frames: rewinding the trajectory
The state of the sampler after step t is reconstructable from (y, s):

  x(t)_j = y_j if s_j ≤ t, else [MASK];        NEXT(t) = { j : s_j = min{ s_k : s_k > t } }  (what the sampler wrote at the next step).

A **frame** is a pair (x(t), NEXT(t)). Per canvas and epoch we take 3 frames:
- 2 sampled from L = { s_j − 1 : κ_j < θ } — the moment right before a low-confidence commit (where the gate would hand over);
- 1 sampled uniformly over all steps (keeps the loss honest on ordinary states).
If a canvas has no low-confidence commit, all 3 are uniform. Frames with prompt + response > 1024 tokens are skipped.

## 3. Loss on one frame
Forward the frame through S once: logits at NEXT(t) → p_S(·|x(t), j) and c_j for every j ∈ NEXT(t). Split:

  U = { j ∈ NEXT(t) : c_j < θ }   (unsure → learn the teacher)        R = NEXT(t) \ U   (sure → do not drift)

  ℓ_KD = mean_{j∈U}  KL( p_S(·|x,j) ‖ p_T(·|x,j) )        reverse KL, teacher forwarded on the same frame x(t)
  ℓ_anchor = mean_{j∈R}  KL( p_S(·|x,j) ‖ p_{S₀}(·|x,j) )  S₀ = the same network with the LoRA switched off (no extra model in memory)
  ℓ = ℓ_KD + λ·ℓ_anchor,   λ = 0.3        (means over U and R taken separately; a term with an empty set is 0)

So the per-position target is exactly what the decode-time gate does: teacher where the student is unsure, the untouched student elsewhere.
The gate's decision at inference (c < θ → take the teacher's token) becomes a soft, distribution-level version of itself at training time.

Options kept as flags: forward KL / JSD (`--kl`), top-k truncation of the teacher support (`--top-k`), temperature (`--kd-temp`), anchor = CE
on the rollout token (`--anchor ce`) or nothing (`--anchor none`), θ linearly decayed from `--theta` to `--theta-end` over the run.

## 4. Optimisation
AdamW (β 0.9/0.95, wd 0), lr 1e-4, 10 % warm-up then linear decay, grad-clip 1.0; micro-batch 1 frame (no padding), gradient accumulation to
32 frames per step; 2 epochs. Size: 17 k canvases × 3 frames × 2 epochs ≈ 102 k frames ≈ 3.2 k optimizer steps; each frame = one student
forward+backward (grad-ckpt) + one teacher forward + one LoRA-off student forward.
Logged per step: `kd`, `anchor`, `ce` (CE on the rollout token, monitoring only), `pos_per_frame` (|NEXT|), `unsure_rate` (|U|/|NEXT|),
`empty_frames` (U = ∅), `theta`, `lr`, `s_per_frame`.
Output: `lora/` and `merged/` (LoRA folded into bf16 weights + Dream's remote-code files → loadable by every eval harness).

## 5. Arms = the same loss with one thing changed
| arm | change | question |
|---|---|---|
| M | — (θ 0.8, T = A(2.0), S₀ = M(.4,.6)) | the method |
| M06 | θ 0.6 | less teacher |
| Msch | θ 0.8 → 0.5 over training | distil less and less |
| Mi | S₀ = Ins, T = A(1.5) | from the official model |
| C1 | U = NEXT(t) (no split, no anchor) | is the selection needed |
| C2 | ℓ = CE on the student's own rollout token, no teacher | is the teacher needed |
| C3 | T = A(1.1) | is "over" needed |
| C4 | x = random i.i.d. mask of y instead of x(t); loss on the 4 most confident masked positions | is the replayed state needed |
| C5 | forward KL | divergence direction |

## 6. What the two smokes check before the first wave
1. **Rollout smoke** (50 prompts, S₀ = M(.4,.6), batch 4): batched left-padded decoding runs; `commit_step` is a valid permutation of steps;
   κ has the expected shape (most ≈ 1, a tail below 0.8 with rate ≈ the decode-time hand-over rate ~10 %); s/prompt → cost of the 17 k pass.
2. **Training smoke** (arm M, 200 canvases, 20 steps): `unsure_rate` in the frames before low commits is well above the uniform frames';
   `kd` decreases; `anchor` stays small; the merged checkpoint loads and decodes.
