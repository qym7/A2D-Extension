# STEP3 plan — frontier over-distillation into one Dream-Instruct (proposed 2026-10-14, nothing submitted)

Goal (user, 2026-10-14): a **training-time** use of the over teacher — a token-level distillation signal, not rollout + SFT/RFT — that folds
the gate's gain into a single diffusion model decoded at 1×, as a **short adaptation** on an open dataset, without regressing outside math.
Deployment target: the trained student decoded with the *standard* Dream sampler (the gate is gone).

## 0. What STEP0–2 and the prelim fixed
- Inference ceiling of the gate on the Ins line: Ins 36.7 → Ins ⇐ A(1.5) @0.8 = 48.3, Ins ⇐ A(2.0) @0.8 = 49.0 (last 300), at 1.14×.
  A trained student at 1× that reaches ≈ 46–48 on the last 300 with no regression elsewhere is the success case.
- The signal lives at the **frontier of a prefix-visible state** (`step3_prelim_signal_check.md`): at the leftmost mask 96 % of the
  gate's low-confidence positions recur (precision 0.87); random masking finds 11–19 % of them. Targets: single A(β) agrees with the gate's
  written token 84 %, the three-A(β) consensus 90 % (coverage 77 %).
- Neighbours to position against (`dlm-train-inference-mismatch-literature`): OPDLM (AR teacher logits on rollout prefixes, forward KL),
  TRIMS (AR-guided trajectory masking SFT), dOPSD (self-distillation from later steps). None uses an **over-merged diffusion teacher on the
  student's own line**, none restricts the loss to the **student's low-confidence frontier**, and all need either an AR teacher or labels.
  The narrative stays "over distillation + selective positions"; the suffix corruption is borrowed machinery, not the contribution.

## 1. Data: prompts only, no labels
Prompts (verified on HF 2026-10-14, all public, none gated except LIMA = auto):
| source | rows | role | why |
|---|---:|---|---|
| `allenai/tulu-3-sft-mixture` (Ke Wang's 100k sample at `/mnt/lts4/scratch/home/kewang/sft/DFT/verl/data/tulu3_sft_100k_seed20260919_allassistant_v2/train.parquet`, read-only) | 8 000 stratified by `source` (all components incl. its math subsets; only safety-refusal / hard-coded sources dropped) | general backbone | Dream-Instruct's own SFT distribution; Tulu-3 → GSM8K/MATH-500 is the standard protocol (user, 2026-10-15) |
| `EleutherAI/hendrycks_math` train (all 7 subjects) | 6 000 | reasoning (in-domain by design) | the target; MATH-500 ⊂ MATH test, disjoint from train |
| `ise-uiuc/Magicoder-OSS-Instruct-75K` | 2 000 | code | HumanEval+ regression guard (HumanEval is hand-written, disjoint) |
| `HuggingFaceH4/no_robots` | 1 000 | chat / instruction following | IFEval guard (disjoint) |
Total 17 k prompts, single turn, Qwen chat template, prompt ≤ 512 tokens. The reference responses are **not used** (label-free);
they are kept only for the RFT/SFT control C2.
**GSM8K train itself is not added** (user, 2026-10-15): Tulu-3 already carries GSM8K/MATH-style subsets (open_math_2_gsm8k, personahub_math, numina),
which is the standard Tulu-3 → GSM8K/MATH-500 setting and is fine; we simply do not add the GSM8K train split on top.

Prefixes are **on-policy**: the student (Ins) decodes each prompt once with the standard Dream sampler (gen length 512, 512 steps,
top-p 0.9, temperature 0.1 — the official MATH-500 settings), giving the canvas the training states are cut from. This is what the prelim
validated (decode-time states); dataset responses as prefixes are off-policy and the prelim says nothing about them. Rollouts are
generated once and reused by every arm (one job, ≈ 17 k × ~8 s / 0.51 ≈ 40 GPU·h-units at batch 8; verify with a 50-prompt smoke).

## 2. Objective (revised 2026-10-15 after discussion: replay the trajectory, gate-shaped target)
The student's rollout is a recording: for every response token we know the sampler step that wrote it and the raw confidence it had.
A training **frame** is the recording rewound to step t — tokens written at steps ≤ t visible, the rest [MASK] — i.e. exactly a state the
student was in while decoding (no new noise is added; random masking is the control C4). NEXT(t) = the positions the sampler filled at step t+1.
Frames per canvas per epoch: 2 sampled from steps right before a low-confidence commit ("the moment before a likely mistake"), 1 uniform.

Student forward on the frame gives p_S at NEXT(t); c_j = max p_S(·|j) is the gate's signal. Teacher forward on the **same** frame gives p_T.

    unsure  j (c_j < θ=0.8):   D( p_S(j) , p_T(j) )                 reverse KL by default (`step3_recipe_survey.md`)   "learn the teacher"
    sure    j (c_j ≥ θ):       λ · KL( p_S(j) ‖ p_S0(j) ),  λ = 0.3  p_S0 = the student before training (LoRA disabled)  "do not drift"
    L = mean over unsure + λ · mean over sure      (means taken separately so the 90 % sure positions do not drown the signal)

So the per-frame target is the gate itself — the teacher where the student is unsure, the original student elsewhere. Positions beyond
NEXT(t) get no loss (prelim: they are uncertain for lack of left context, the gate never scores them).
Teacher = A(1.5) when the student is Ins; A(2.0) when the student is the best merge M(.4,.6) (the two gates measured in STEP1).

Why a wrong token already in the prefix is not a dead end: the frame *before* it was written is also in the recording, with that position still
masked and low-confidence — that frame is where the teacher corrects it, and the low-confidence frame sampling makes such frames the majority.
After the mistake, later frames teach "the best continuation given this prefix", exactly what the gate does at decode time (it keeps confident
wrong tokens too). A second round (re-record with the trained student, train again) is the on-policy iteration that improves the later frames.

Student init: run both **Ins** (the "enhance the official model" story, ceiling ≈ 48–49) and **M(.4,.6)** (higher start, ceiling ≈ 51; its weights
are free, so distilling the merge into Ins at the sure positions would be wasted compute — use it as the init instead). Same code, two configs.
Ground truth is not used in the main method (prefixes are the student's own text, GT tokens do not align); it appears only in the RFT control C6.

## 3. Arms (LoRA r = 64, α = 128, all linear modules, lr 1e-4, 2 epochs, batch 32 states, max length 1 024, bf16, grad-ckpt)
| arm | corruption | positions | target | tests |
|---|---|---|---|---|
| **M** gate distillation | replay | NEXT, p_S < 0.8 → teacher; sure → anchor to untouched student | A(1.5) / A(2.0) | the method |
| C1 no split | replay | every NEXT position → teacher | A(1.5) | is *selection* needed |
| C2 self-SFT | replay | NEXT, all | rollout token (CE) | is the *teacher* needed (= on-policy SFT / unfiltered RFT, dOPSD-like) |
| C3 non-over teacher | replay | NEXT, p_S < 0.8 | A(1.0) distribution | is *over* needed (A(1.0) gate = 44.3 vs A(1.5) 48.3) |
| C4 random-mask KD | i.i.d. mask r ~ U(0,1) | the 4 most confident masks, p_S < 0.8 | A(1.5) | is the *replayed state* needed (standard dLLM SFT corruption) |
| Msch θ schedule | replay | NEXT, p_S < θ(t), θ 0.8 → 0.5 linearly over training | A(2.0) | distil less and less as the student improves (user, 2026-10-15); on top of the natural shrinking of the unsure set |
| C5 forward KL | replay | NEXT, p_S < 0.8 | A(1.5), KL(p_T‖p_S) | divergence direction (TOPD says reverse; cheap check on our teacher) |
| C6 RFT | official random mask | all masked (official Dream SFT loss) | rollout tokens of MATH prompts whose answer matches GT | the classic "why not just RFT" baseline (MATH 6 k only) |
| C7 plain SFT | official random mask | all masked | the datasets' reference responses | "is it just more training" (same 17 k prompts) |
| C0 | — | — | — | Ins untouched (exists) |
Reference lines: gate Ins ⇐ A(1.5) @0.8 = 48.3 (1.14×), A(1.5) alone, best merge M(.4,.6) = 46.6 all-500.
Budget rule: M first with C2 (the cheapest and most damaging control); C1/C3/C4 only if M clears the success bar.

## 4. Evaluation (official harnesses in `dream_study`, standard sampler, student alone at 1×)
| benchmark | protocol | regression bar |
|---|---|---|
| MATH-500 | last 300 (ids 200–499), official `math500_minerva_v2.sh` and the lockstep @1024 (comparability with STEP1 tables) | primary: M ≥ Ins + 6 (≈ 43+) with McNemar p < 0.05 vs Ins; M > C2 by ≥ 3 |
| GSM8K | official (`dreaminst_gsm8k_full` exists as Ins reference); no GSM8K train split added (Tulu-3's math subsets are in) | ≥ Ins − 1 |
| HumanEval+ | EvalPlus official (`dreamv0_humanevalplus` protocol) | ≥ Ins − 2 |
| IFEval | official `eval/ifeval.sh` (1280 steps; 9 h) | ≥ Ins − 2 (prompt-level strict) |
Finish rate and answer length on MATH-500 are reported alongside (the over teacher's failure mode is early EOS/non-finishing;
the student must keep Ins's ~91 % finish rate).

Reading rules, fixed now:
- **Success**: M clears the MATH bar and all three guards. Then: consensus-target variant, θ/K sweep, full FT instead of LoRA.
- **Teacher not needed**: C2 within 2 points of M → the gain is on-policy frontier SFT (dOPSD/TRIMS territory); idea 1 as a training signal dies,
  the inference-time gate stays the deliverable.
- **Selection not needed**: C1 ≥ M − 1 → drop θ, the method is "frontier KD from an over teacher" (simpler; still ours via the teacher).
- **Over not needed**: C3 ≥ M − 1 → teacher quality is not the point; report honestly.
- **Regression**: any guard fails → lower lr / 1 epoch / λ = 0.5 once; if it still fails, the recipe is math-only and the "enhance Dream-Ins"
  framing is dropped (user already accepts "reasoning toward AR, no regression elsewhere" as the realistic goal).

## 5. Implementation
- Trainer: Dream's official `fsdp_sft_trainer.py` (Ke Wang's checkout `external/Dream`, read-only; copy into `code/step3/train/`).
  It already has `q_sample`, LoRA via peft (`model.lora_rank`), grad-ckpt, FSDP. Changes: (i) replace `q_sample` by the suffix cut
  (+ random-mask path for C4), (ii) a frozen teacher model loaded next to the student (bf16, no grad; 2 × 15 GB + LoRA activations fit a
  0.51 H200 at micro-batch 2), (iii) the loss in §2 with the frontier/θ mask, (iv) log |S| per state, mean KL, share of states with S = ∅.
- Rollout generation: one script on the standard sampler, records per prompt the canvas and the per-step confidence (for the "interesting
  cut" sampling) — `code/step3/rollout.py`; output `outputs/step3/rollouts/<split>.jsonl`.
- Smokes before any arm: (a) 200-state CPU/GPU check that S recovers the prelim's rates (mean |S| ≈ 2 of 4, ≈ 55 % of states with S ≠ ∅);
  (b) 20-step training smoke, loss decreasing, LoRA merge → 20-problem MATH decode not degenerate.
- Checkpoints: `outputs/checkpoints/step3-<arm>` (LoRA merged into bf16 weights, so every eval harness loads them like any Dream checkpoint).
- Job names hyphens only, `-h200f051`, `-g 0.51 --node_type h200`, via `modules/submit.sh`; everything in `../tmp.md`.

## 6. Cost (H200·0.51 units; estimates, to be replaced by smoke timings)
| item | GPU·h | wall |
|---|---:|---|
| rollouts, 17 k prompts (once) | ≈ 40 | 1 job, ~2 days, or 4 shards ~12 h |
| training per arm (34 k states × 2 epochs, student fwd+bwd + teacher fwd, LoRA) | ≈ 10–15 | ~0.6 day |
| eval per arm: MATH-500 last 300 lockstep 6 + official 2, GSM8K 2, HumanEval+ 1, IFEval 9 | ≈ 20 | 5 jobs in parallel, ~9 h |
| **M + C2** (first wave) | ≈ 110 | ~4 days |
| + C1, C3, C4 | + ≈ 105 | +3 days |
Ke Wang's reference point: full FT of Dream-Base on 100 k Tulu-3 (batch 256, 390 steps) took 3.4 h on 4 × B300.

## 7. Open before submission (user decisions)
1. Prefix source: on-policy rollouts (planned, +40 GPU·h) vs dataset responses (free, off-policy, prelim does not cover it).
2. Domain check first? Gate arms Ins / A(1.5) / Ins ⇐ A(1.5) @0.8 on HumanEval+ and IFEval (needs the lockstep decoder wired into those
   harnesses, ~1 day of work + ~25 GPU·h) would tell whether the teacher helps outside math *before* training on general prompts.
   My recommendation: skip it — the training arms answer the same question on the student directly, the gate check only de-risks the
   data mix, and the guards in §4 catch a regression either way.
3. Mix: 17 k as in §1 (GSM8K train removed 2026-10-15), or math-only 6 k (cheaper, answers the mechanism question first, ignores the "enhance all of Dream-Ins" goal).
