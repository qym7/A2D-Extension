# STEP3 — Distillation: one model that does what the gate did (2026-10-16, in progress)

STEP2 is closed (`step2_verdict.md`): the decode-time gate Ins ⇐ A(2.0) @0.8 is the ceiling (49.0, 2× cost) and nothing combined beyond it.
STEP3 trains that behaviour into a single 1× model. Recipe: `technical_details/step3_training_formulation.md`; plan and reading rules:
`technical_details/step3_plan_frontier_kd.md`.

## Takeaways (1k test, 2026-10-16; full last-300 numbers)
1. **A 1× model reaches the 2× gate's accuracy** after 185 optimizer steps on 1 k training prompts (13 minutes of training):
   50.3 vs Ins 36.7 on the last 300 MATH-500 problems, p < 0.001; the gate it was distilled from: 49.0.
2. **The teacher is what matters.** Self-SFT on the same rollouts with the same trainer (C2) stays at Ins (36.3, p = 1.0 vs Ins).
3. The gain has the gate's signature: finish rate 76 → 92 % **and** accuracy on finished problems 48.5 → 54.7 (the gate: 93 % / 52.9).
4. **The Tulu-3-only run (tulu3-1k) is NOT done** — the 2026-10-16 attempt was invalid (two pipeline bugs, see "Invalid tulu3-1k run"); it is re-queued.
   Its one usable observation: a model trained on only **119 canvases for 22 steps** also scored 50.3 — the effect saturates very early.
5. **Generalisation (official IFEval / HumanEval+ harnesses, random-100 IFEval):** the distilled model keeps Ins-level instruction following (64 vs 62)
   and best-merge-level code (60.4 vs 61.0) while the teacher A(2.0) itself is bad at IF (51). It took the teacher's math, not its damage.
   The 2× gate scores 70 on IFEval and 59.8 on HumanEval+ — the gate itself does not damage IF / code either.
6. **AR-teacher baseline (2026-10-17):** distilling Qwen2.5-7B-Instruct into Ins on prefix-masked rollouts (same data, trainer, budget) gives
   46.7 on MATH (vs 50.3 for the diffusion over-teacher) and **drops IFEval to 52** (p = 0.01 vs distilled M). The diffusion teacher wins on both axes.

## The three rows

| row | what is evaluated | how it was made | training data | test-time cost |
|---|---|---|---|---|
| **Ins** | `Dream-org/Dream-v0-Instruct-7B`, unchanged | — | — | 1× |
| **C2** | Ins + LoRA, merged into one checkpoint | self-SFT: cross-entropy on the token Ins itself wrote at each replayed step (same frames, same optimiser, **no teacher**) | the 1 k rollouts | 1× |
| **M** | Ins + LoRA, merged into one checkpoint | gate distillation: on the frames before a low-confidence commit, reverse KL to A(2.0); on confident frames, KL to untouched Ins (λ 0.3) | the 1 k rollouts | 1× |

All three rows are decoded **by one model alone** with the identical sampler (`code/step1/lockstep_gated.py`, mode `free`, 1024 steps, top-p 0.9,
conf-temp 0.1): no teacher, no gate, no second forward pass at test time. The teacher A(2.0) = D_b + τ_D + 2.0·τ_A
(`outputs/checkpoints/add-a10-b20`) is used only inside the training loss.

Hyper-parameters (M and C2 identical except the loss): LoRA r 64 / α 128 on all attention + MLP projections, lr 1e-4 (10 % warm-up, linear
decay), 32 frames per step, micro-batch 4, 2 epochs over 1 000 canvases × 3 frames = 6 000 frames = 185 steps, θ = 0.8.
Checkpoints: `outputs/checkpoints/step3-mixpool-1k-{M,C2}/{lora,merged}`, logs `train_log.jsonl`, args `args.json`.

## Training data (what the students saw)
`outputs/step3/prompts.jsonl` ids 0–999, a fixed-seed sample of the 17 k STEP3 prompt pool (`code/step3/build_prompts.py`, seed 20261015):

| source | n (of 1 000) | what |
|---|---|---|
| Tulu-3 SFT mixture (`allenai/tulu-3-sft-mixture`, all subsets except wildjailbreak / wildguard / hard_coded) | 489 | general instruction data incl. its math subsets (open_math_2_gsm8k, personas-math, …) |
| MATH **train** split (`EleutherAI/hendrycks_math`, train) | 335 | competition math, wrapped as `Problem:\n…\n\nSolution:` |
| Magicoder-OSS-Instruct | 122 | code |
| no_robots | 54 | chat |

Only the **prompts** are used. Each student decodes every prompt itself once (`code/step3/rollout.py`, official Dream sampler, 512 tokens /
512 steps, T 0.1, top-p 0.9; `outputs/step3/rollouts/ins/`), recording the step and raw confidence of every written token. No reference answers,
no ground truth, no filtering by correctness enter training. The training signal is the teacher's next-token distribution at the student's own
low-confidence steps (M) or the student's own token (C2).

**Why this mix, and is it a standard recipe?** Yes for the sources, no for how they are used.
- *Tulu-3 SFT mixture* is the open standard post-training set (AI2) and is what Dream-v0-Instruct itself was fine-tuned on (the official
  `examples/run_sft_tulu3.sh`; the Dream report: SFT on Tulu 3 + SmolLM2). So its prompts are the student's own training distribution, and
  "train on Tulu-3, read GSM8K / MATH-500 / IFEval / HumanEval" is the usual protocol. Its math subsets stay in (that is where the standard
  GSM8K/MATH signal comes from); GSM8K train is not added on top.
- *MATH train* is the usual prompt source for math RL / distillation on diffusion LMs (d1, TOPD, …): in-domain for the target metric, disjoint
  from MATH-500 by construction.
- *Magicoder-OSS-Instruct* and *no_robots* are standard code and chat sets; they are there so the student keeps seeing code and chat prompts
  during training, which is what the HumanEval+ / IFEval guards check.
- What is **not** standard: no reference responses are used at all. Standard SFT fits the dataset's answers; here the dataset only supplies
  prompts, the student writes its own answers, and the teacher's distribution at the student's unsure steps is the only external signal.
  The proportions (≈ ½ Tulu-3, ⅓ MATH, the rest code/chat) are ours and were not tuned.

**Decision 2026-10-16 (user): go with the most standard recipe = Tulu-3 only, native proportions.** Tulu-3 already has the sub-structure we were
adding by hand: in the 100 k sample it is ≈ 36 % math (personahub_math 15.9 %, NuminaMath-TIR 6.9 %, personas-math-grade 5.4 %,
open_math_2_gsm8k 5.3 %, intermediate-algebra 2.1 %), ≈ 15 % code (evol_codealpaca 11.4 %, personahub_code 3.7 %), ≈ 40 % chat / general
(wildchat, aya, flan, oasst, no_robots 1 %), 3 % IF data, ≈ 12 % safety (wildjailbreak, wildguard, coconot). So MATH train, Magicoder and
no_robots on top were redundant (no_robots is even a Tulu-3 subset), and the equal-share stratification distorted Tulu's own mix. The next pool
is `outputs/step3/prompts_tulu3.jsonl` (`build_prompts.py --mix tulu3-native`, now the default: one shuffled uniform draw of 17 k from all
Tulu-3 sources, nothing excluded, prompt ≤ 512 tokens; a run with `n_prompts: N` takes ids 0–N−1, which is a native-proportion sample of size N;
`pipeline/config.yaml: prompt_set: tulu3`). The mixpool-1k numbers above were obtained on the earlier mix17k pool and stay labelled as such.

The **tulu3-1k run** (launched 2026-10-16) uses ids 0–999 of this pool: 415 math (personahub_math 179, NuminaMath-TIR 84, personas-math-grade 64,
open_math_2_gsm8k 63, intermediate-algebra 25), 123 code (evol_codealpaca 98, personahub_code 25), 289 chat / general (aya 97, wildchat 89,
flan 85, oasst 9, no_robots 4, sciriff 3, table_gpt 2), 42 IF, 131 safety (wildguard 68, wildjailbreak 51, coconot 12). Math is 41 % rather
than the raw 36 % because the ≤ 512-token filter removes more long chat prompts than math prompts. Leakage check of these 1 000 prompts against
MATH-500 ids 200–499: 0 containments, 0 shared 13-grams.

## Test data and leakage
Test = MATH-500 (`HuggingFaceH4/MATH-500`, the standard 500-problem subset of the MATH **test** split), ids 200–499 (last 300, user rule
2026-10-12). Checked (`tmp.md` 2026-10-16): the 1 000 training prompts vs the 300 test problems, normalised text —
0 exact matches, 0 containments, 1 shared 13-gram, which is the multiple-choice boilerplate "(A) Line (B) Circle … Enter the letter of the
correct option." on two different problems (train 875 vs test 255). MATH train and test are disjoint by construction; Tulu-3 is decontaminated
against MATH test by its authors. → **no leakage from the training prompts to the test problems.** The student rollouts are generated on the
training prompts only, never on MATH-500.

## Generalisation: the 1× distilled model vs both end points and the best merge (2026-10-16)
Every row is **one model decoding alone** (the gate row is the 2× decode-time gate it was distilled from). MATH on the SAME ids per column; IFEval and
HumanEval+ with the **official harnesses and the official decode settings** (IFEval 1280/1280, HumanEval+ 512/512 T0.1, both on H200·0.51 except where noted).
Subsets are **fixed random** lists (never the first/last N): MATH rand100 = seed-0 random 100 of all 500 ids; IFEval rand100 = seed-0 random 100 of the 541
prompts (`code/step3/guards/ifeval_rand100.yaml`, official scorer re-exported). Ins and M(.4,.6) IFEval / HumanEval+ were **not re-run**: their rows are
read off their existing official full-set logs (IFEval full 541: Ins 56.6, M(.4,.6) 66.5 prompt-strict; HumanEval+ Ins 2026-09-19). McNemar cells:
wins-of-row / wins-of-column on the shared items, exact two-sided p.

| row | MATH last 300 | MATH rand100/500 | IFEval rand100 (strict) | HumanEval+ (164) | MATH100 vs Ins | IFEval vs Ins | HE+ vs Ins | MATH100 vs M(.4,.6) | IFEval vs M(.4,.6) | HE+ vs M(.4,.6) |
|---|---:|---:|---:|---:|---|---|---|---|---|---|
| Ins (student, untouched) | 36.7 | 34 | 62 | 57.3 | — | — | — | 9/17, p=0.17 | 13/16, p=0.71 | 10/18, p=0.18 |
| M(.4,.6) best merge | 42.0 | 42 | 65 | 61.0 | 17/9, p=0.17 | 16/13, p=0.71 | 18/10, p=0.18 | — | — | — |
| A(2.0) over addition (= the teacher) | 46.7 | 55 | 51 | 56.7 | 29/8, p=0.00 | 8/19, p=0.05 | 19/18, p=1.00 | 22/9, p=0.03 | 8/22, p=0.02 | 16/23, p=0.34 |
| gate Ins ⇐ A(2.0) @0.8 (2×) | 49.0 | 51 | 70 | 59.8 | 22/5, p=0.00 | 17/9, p=0.17 | 21/16, p=0.51 | 18/9, p=0.12 | 12/7, p=0.36 | 16/19, p=0.74 |
| M distilled (mixpool-1k) | 50.3 | 58 | 64 | 60.4 | 30/6, p=0.00 | 14/12, p=0.85 | 17/12, p=0.46 | 24/8, p=0.01 | 7/8, p=1.00 | 14/17, p=0.72 |
| AR-teacher baseline: Ins ⇐ Qwen2.5-7B-Instruct, prefix mask (mixpool-1k) | 46.7 | 51 | 52 | 57.3 | 26/9, p=0.01 | 12/22, p=0.12 | 15/15, p=1.00 | 19/10, p=0.14 | 5/18, p=0.01 | 10/18, p=0.18 |

- **MATH**: distilled M is the best row everywhere (last 300: 50.3; rand100 of all 500: 58), significantly above Ins and above the best merge, and at
  the level of the 2× gate it was distilled from (51 on rand100) while decoding once.
- **IFEval**: the teacher A(2.0) is **bad** at instruction following (51, p = 0.05 vs Ins, p = 0.02 vs M(.4,.6)) — exactly the "over" end point's known damage.
  The distilled student does **not** inherit it: 64 vs Ins 62 / M(.4,.6) 65, p ≈ 1. The 2× gate itself scores **70** (25/6 vs its own donor A(2.0),
  p < 0.001; 17/9 vs Ins, p = 0.17; 13/7 vs distilled M, p = 0.26) — handing over only the low-confidence tokens turns a bad-IF donor into a
  small IF gain; the distilled model keeps Ins level but not the gate's 70 (decoded with `code/step3/guards/gate_ifeval.py`: same gate code and
  argmax tokens as the gate's MATH row, official prompts and scorer, not lm_eval's sampled decode). On the full 541 set the best merge beats Ins by 10 points (66.5 vs 56.6, p < 0.001);
  on this random 100 the gap is 3 points, p = 0.71 — a 100-prompt subset cannot resolve a 10-point IFEval difference, which is why the validation
  set below is 100 **per task** but any real claim goes to the full set.
- **HumanEval+**: distilled M 60.4 ≈ gate 59.8 ≈ best merge 61.0 > Ins 57.3 > teacher 56.7; none of the pairwise differences is significant (164 problems).
  The teacher is not better at code, and the student did not get worse.
- **AR-teacher baseline (2026-10-17, a colleague's request: "mask the rollout AR-style and distil from the AR model directly")**: same Ins rollouts, same
  trainer, same budget (3 frames per canvas, 185 steps, 0.46 h), but the corruption is a random **prefix cut** (prefix visible, suffix masked, loss at the
  first masked position = the one place an AR model is defined) and the teacher is **Qwen/Qwen2.5-7B-Instruct** (causal logits on the clean sequence;
  same vocabulary as Dream). Selection kept (student p < 0.8 → teacher, else anchor). It **does** teach math — 46.7 on the last 300, 51 on rand100 (26/9 vs
  Ins, p = 0.01) — but less than the over-added diffusion teacher (distilled M 50.3 / 58; 9/16 vs AR, p = 0.23), and it **costs instruction following**:
  IFEval 52, 5/18 vs M(.4,.6) (p = 0.01) and 4/16 vs distilled M (p = 0.01), although Qwen2.5-7B-Instruct itself is a strong IF model. Code unchanged
  (57.3 = Ins). Training signature differs too: the student was "unsure" at only 17 % of the prefix-cut positions (60 % at the gate's replayed states), so
  the AR teacher was consulted on far fewer positions. One recipe, untuned (no JustOPD-style entropy top-25 % or L2R student rollouts); the
  all-positions variant (`ARall`) and a matched-teacher-strength control (a 7B diffusion teacher as strong as Qwen on MATH) are open.
- **Where the AR baseline loses (2026-10-17 read).** Finish rate, accuracy on finished problems and length on the last 300; IFEval length on the rand100:

  | row | MATH rand100 | MATH finish | acc \| finished | MATH len (tok) | IFEval rand100 | IFEval len (chars) | HumanEval+ |
  |---|---:|---:|---:|---:|---:|---:|---:|
  | Ins | 34 | 76 % | 48.5 | 375 | 62 | 1622 | 57.3 |
  | M(.4,.6) best merge | 42 | 81 % | 51.9 | 448 | 65 | 1753 | 61.0 |
  | A(2.0) teacher | 55 | 94 % | 49.5 | 1005 | 51 | 4627 | 56.7 |
  | gate Ins ⇐ A(2.0) @0.8 (2×) | 51 | 93 % | 52.9 | 795 | 70 | 2375 | 59.8 |
  | M distilled (mixpool-1k) | 58 | 92 % | 54.7 | 639 | 64 | 2826 | 60.4 |
  | AR-teacher baseline | 51 | **75 %** | 61.9 | 664 | **52** | 2381 | 57.3 |

  The AR baseline's finish rate is Ins's (75 vs 76 %); its high accuracy-on-finished is a **selection effect**, not better math: on the 218 problems
  that both it and distilled M finish, both score **62.8**; on the 58 problems M finishes and the AR baseline does not, M gets 24 % (the hard ones).
  So the AR teacher taught nothing the diffusion teacher did not, and the gate's main gain (finishing: 76 → 92 %) is exactly what prefix-valid states
  cannot teach — the stall happens mid-canvas in any-order decoding. On IFEval the lost instructions are constraint types (no_comma 5, start/end
  quotation 3, repeat-prompt 3, placeholders 3, response language 1): the answers read as generic Qwen-style text that ignores the prompt's format
  clause. Consequence: a **joint** (diffusion + AR teacher) distillation has no complementary signal to combine on this evidence — not planned.
- Reading: the selective loss took the teacher's math and left its instruction-following / code damage behind — on this evidence. Caveats: one seed,
  one training set, 100-prompt subsets for IF; GSM8K not run.
- Raw: `outputs/step3/guards/<ins|m46|a20|gate|mixpool1k|ar-mixpool1k>/{ifeval_rand100,humaneval}.json` (per-item), `guards/math500_rand100_seed0_results.json`;
  AR arm: `outputs/checkpoints/step3-mixpool-1k-AR/` (args, train_log, W&B 49u7anm9), MATH `outputs/step3/step3-mixpool-1k-AR/math500/`.

## Results (1k test; table regenerated by `code/step3/results_table.py` → `technical_details/step3_results_t1k.md`)

Every row below is decoded by **one model** on the same 300 MATH-500 problems (ids 200–499), except the gate row (2×, two models at decode
time). The two **end-point baselines** of STEP0 are in the table: the best merge M(.4,.6) (convex line) and the over addition A(2.0)
(addition line; it is also the teacher). McNemar = wins-of-row / wins-of-column, exact p.

| row | acc | finish | acc\|finished | avg len (tok) | vs Ins | vs C2 | vs M(.4,.6) | vs A(2.0) | HW |
|---|---:|---:|---:|---:|---|---|---|---|---|
| Ins (student, untouched) | 36.7 | 76% | 48.5 | 375 | — | 37/36, p=1.00 | 26/42, p=0.07 | 32/62, p=0.00 | B300 |
| M(.4,.6) best merge (end point 1) | 42.0 | 81% | 51.9 | 448 | 42/26, p=0.07 | 45/28, p=0.06 | — | 35/49, p=0.16 | B300 |
| A(2.0) over addition (end point 2 = the teacher) | 46.7 | 94% | 49.5 | 1005 | 62/32, p=0.00 | 58/27, p=0.00 | 49/35, p=0.16 | — | H200 |
| gate Ins ⇐ A(2.0) @0.8 (2×) | 49.0 | 93% | 52.9 | 795 | 62/25, p=0.00 | 64/26, p=0.00 | 51/30, p=0.03 | 40/33, p=0.48 | H200 |
| **M** = Ins ⇐ A(2.0) distilled (mixpool-1k pool), **θ = 0.8**, λ_anchor 0.3, reverse KL, lr 1e-4, 2 ep, LoRA r64 (1×) | **50.3** | 92% | 54.7 | 639 | 63/22, p=0.00 | 59/17, p=0.00 | 51/26, p=0.01 | 46/35, p=0.27 | H200 |
| **C2** = Ins self-SFT (CE on its own rollout token, no teacher), lr 1e-4, 2 ep, LoRA r64 (1×) | 36.3 | 80% | 45.6 | 856 | 36/37, p=1.00 | — | 28/45, p=0.06 | 27/58, p=0.00 | H200 |

**θ (the threshold) in M** is the student's raw confidence below which a frame is handed to the teacher: at the position the sampler writes next,
c = max_v p_S(v | state); c < θ → reverse KL to A(2.0); c ≥ θ → KL to untouched Ins (weight λ_anchor). The same θ also selects the training
frames (2 of 3 frames per canvas are taken right before a commit whose rollout-time confidence was < θ). θ = 0.8 is the decode-time gate's value
(STEP1 sweep); arms M06 (θ 0.6) and Msch (0.8 → 0.5) are the planned θ ablations. Flags: `--theta`, `--theta-end`, `--lambda-anchor`, `--kl`.
avg len = mean response length in Dream tokens up to the end of the decoded text (the canvas is always 1024).

What the table says, in order of confidence:
1. **M ≫ Ins and M ≫ C2** (both p < 0.001, ≈ 60 wins vs ≈ 20): the distilled 1× model is better than what it started from, and the teacher is
   the reason (self-SFT does nothing).
2. **M ≥ both end points.** M beats the best merge (51/26, p = 0.01) and is at least level with the over addition / teacher
   (46/35, p = 0.27) while being a 1× model with Ins's weights + a LoRA — no merge, no vector arithmetic at test time.
3. **M ≥ the 2× gate it was distilled from** (50.3 vs 49.0; the gate was the ceiling we set out to reach).
4. It has the gate's two signatures: finish rate 76 → 91–92 % (teacher-level) and accuracy on finished problems 48.5 → 55 (above the teacher's 49.5).

Training provenance: `outputs/checkpoints/step3-mixpool-1k-{M,C2}/` (`args.json`, `train_log.jsonl`, `run.log`); W&B project `a2d-step3`,
runs `step3-mixpool-1k-M` / `step3-mixpool-1k-C2` (MATH logged as `eval/math500_last300_*`). Note: the mixpool-1k M job's container restarted twice before
completing (`run.log` shows three starts, one `[done]`); the checkpoint is from the single complete run, the log file holds the partial attempts too.

## What a truly useful evaluation needs (and which of it we have)
| requirement | status | action |
|---|---|---|
| same hardware for every row | Ins and M(.4,.6) are B300 numbers (2026-09-30); all STEP3 rows H200 | re-decode Ins and M(.4,.6) on H200, last 300 (2 jobs, ≈ 5 GPU·h) |
| full 300 problems, not 200 | M's shards 0 and 2 running | automatic; table regenerates |
| the model did not lose elsewhere | IFEval rand100 64 (Ins 62), HumanEval+ 60.4 (Ins 57.3) — see Generalisation | GSM8K and full-541 IFEval still open |
| not a seed artefact | one seed | second training seed of M (0.5 h + 7 GPU·h eval) |
| not a 1k artefact | one data size | 8k offline (or 4 × 2k online) — also tells whether 1k is already the plateau |
| statistics that respect pairing | done: exact McNemar on identical problem ids | keep; never compare accuracies across different id sets |
| no leakage | checked (previous section) | re-check whenever the prompt pool changes |
| the controls that kill alternative stories | C2 (teacher needed?) done; C1 (selection needed?), C3 (over needed?), C4 (replayed state needed?), C5 (KL direction) not run | run after the guards, on the same rollouts (0.5 h training each) |

## Training recipe (exactly what `code/step3/train_frontier_kd.py` ran for M, from `args.json` + `run.log`)

| item | value |
|---|---|
| student / teacher | Ins (`Dream-org/Dream-v0-Instruct-7B`) / A(2.0) (`outputs/checkpoints/add-a10-b20`), both bf16; teacher frozen, kept in memory |
| trainable | LoRA r 64, α 128, dropout 0, no bias, on q/k/v/o/gate/up/down projections of every layer (≈ 160 M params); base weights frozen |
| anchor model | the same network with the LoRA disabled (`disable_adapter`), no extra copy |
| data | 1 000 Ins rollouts → 988 canvases (12 dropped: response < 4 tokens or a single commit step); 3 frames per canvas per epoch (2 before a commit with κ < θ, 1 uniform); frames with prompt + response > 1024 tokens skipped |
| steps | 2 epochs × 988 × 3 = 5 928 frames / 32 per optimizer step = **185 optimizer steps** |
| batch | 32 frames per step = micro-batch 4 (auto-probed: 8 OOMs at 69.7 GB on H200·0.51) × 8 gradient-accumulation steps |
| loss | θ = 0.8: c < θ → reverse KL(p_S ‖ p_T); c ≥ θ → λ·KL(p_S ‖ p_{S₀}), λ = 0.3; temperature 1.0, no top-k truncation; means over the two position sets taken separately |
| optimizer | AdamW, β = (0.9, 0.95), ε default (1e-8), **weight decay 0** |
| lr schedule | peak 1e-4, linear warm-up over 10 % of the steps (18 steps), then linear decay to 0 at step 185 |
| grad clip | global norm 1.0 (on the LoRA parameters) |
| precision | bf16 weights, bf16 autocast forward, fp32 LoRA master params in AdamW, gradient checkpointing on |
| seed | 0 (frame sampling and torch) |
| cost | 0.22 h on one H200·0.51, peak 69.7 GB; 0.13 s per frame |
| output | `lora/` (adapter) and `merged/` (LoRA folded into bf16 weights + Dream remote code) — every evaluation loads `merged/` |

Not used: dropout, label smoothing, EMA, KL warm-up, curriculum on θ (the Msch arm would add one), teacher top-k, length normalisation.
Monitoring per step (`train_log.jsonl`, W&B `train/*`): `kd`, `anchor`, `ce` (CE on the rollout token, not optimised), `unsure_rate`, `empty_frames`, `lr`, `peak_gb`.

## How the training code was built
Written for this step, not adapted from a trainer: `code/step3/train_frontier_kd.py` is ≈ 300 lines of plain PyTorch + HF `transformers` +
`peft` (LoRA), single GPU, no Lightning, no verl, no accelerate. What is reused unchanged: Dream's official sampler for the rollouts
(`diffusion_generate`, same settings as the MATH-500 protocol), Dream's logit convention (position p is predicted from hidden p−1, as in the
official SFT trainer and in our decoder), peft for LoRA exactly as the official `run_sft_tulu3.sh` enables it, and our STEP1 lockstep decoder
for evaluation. What is new: trajectory replay as the corruption, the confidence-split loss, the self-anchor, the frame sampler, the OOM
micro-batch probe, and the merge/save path (`merge_lora.py`, working around Dream's `generation_config.validate`). The official verl FSDP
trainer was not used because its loss (random-mask CE on reference answers) is exactly what this step replaces, and it needs 4 GPUs.

## Invalid tulu3-1k run (2026-10-16) — what went wrong, what it still tells us

Intended: the same recipe on 1 000 Tulu-3-only prompts. Two pipeline bugs made it something else:
1. the prompt-file path was passed as an environment variable, which never reaches the job container → the rollout jobs decoded the **mix17k pool** (same prompts as mixpool-1k);
2. `rollouts.jsonl` is appended incrementally, and after a snakemake restart an existing file counted as finished → `all.jsonl` was concatenated with **122 rows**.
So "M (tulu3-1k)" was trained on **119 mix-pool canvases for 22 optimizer steps** — and still scored **50.3 / 91 % finish / 813 tokens** on the last 300
(vs M(mixpool-1k) 33/33, p = 1.0). Artefacts moved to `trash/n1k_invalid_20261016/` (README there); W&B run `step3-tulu3-1k-M` is that invalid run.
Fixes (committed): prompt file is an explicit runner argument; rollout shards write a `done` marker that the Snakefile waits for; MATH shards are
complete only when `summary.json` exists. The Tulu-3-only run is re-queued under the same tag with the fixed pipeline (15 jobs, ≈ 20 GPU·h).
What the accident says: the MATH gain is reached after ~700 frames — the 1 k / 185-step budget is already far past saturation for this metric.
That makes the guards (code / IF) and the length read the informative part of the next run, not MATH.

## Related: distilling an AR teacher into a diffusion student (why we do not just do that; 2026-10-16)

A colleague's question: why not distil the AR model (Qwen) directly? It has been tried — and every method has to retreat to **prefix-valid
states**, because an AR model defines p(x_j | x_<j) only and cannot condition on a partially filled, any-order canvas:

| work | how the AR teacher is made to fit | result |
|---|---|---|
| JustOPD (arXiv 2610.17397) | student (Dream-7B) rolls out **left-to-right during training** so every state is a prefix; Qwen3-14B teacher scored on the prefix; loss on the 25 % highest-teacher-entropy positions; any-order decoding restored at inference | Dream-7B MATH-500 @1024: 36.8 → 48.0 (JSD), 43.8 (reverse KL) |
| ForkLeft (arXiv 2609.32448) | entropy-first rollout, then the written part is **fixed as a prefix** and the teacher supervises under that shared prefix | Efficient-DLM-4B MATH-500 72.6 → 79.6 (teacher Qwen3-30B-A3B) |
| OPDLM (arXiv 2606.06712) | student is block-diffusion; teacher scores only the **causal prefix of the final sequence x₀**, never a masked state | AR → DLM conversion with 15–7000× fewer tokens |
| TIDE (arXiv 2604.26951) | cross-tokenizer chunk-level likelihood matching | +1.5 avg over 8 benchmarks |
| Distilled DLMs / Target Concrete Score (ICLR'25 submission) | AR model used to form a target concrete score | early |

JustOPD states the constraint explicitly: intermediate denoising states "do not generally correspond to valid prefixes under the ARM's
training distribution". Our teacher A(2.0) is itself a diffusion model, so it supervises the student on the **states the student actually
visits** (the gate's states, confidence order, both-sided context) and at the positions the gate selects — the thing an AR teacher cannot do.
Sequence-level AR distillation (SFT on AR outputs) is the other option and is what A2D's failed-transfer list and our planned C7 control cover.

Directly comparable number: JustOPD moves Dream-7B from 36.8 to 48.0 on MATH-500 @1024 with a 14B AR teacher; we move Ins (36.7 on our last-300)
to 50.3 with a 7B over-added teacher, 1 k prompts, 185 steps. Same problem set needed before this is quoted side by side (TODO).

**We then ran it ourselves (2026-10-17, arm `AR` in `train_frontier_kd.py`):** `--corruption l2r --teacher-type ar`, teacher Qwen2.5-7B-Instruct, the same
Ins rollouts / selection / anchor / budget as M. Result (full row in "Generalisation"): MATH last 300 **46.7** (M 50.3, Ins 36.7), IFEval rand100 **52**
(M 64, Ins 62), HumanEval+ 57.3 (M 60.4). So on our data the AR teacher is a weaker math teacher than the over-added diffusion teacher *and* damages
instruction following, even with a stronger-on-paper teacher. Caveat: one untuned recipe; the AR teacher was consulted on only 17 % of positions
(student rarely unsure at a prefix-valid state).

## Caveats on the current numbers
- All 300 problems are in for every row (the mixpool-1k merge rule had fired before two re-run shards finished; merged by hand 2026-10-16 13:30).
- The Ins row was decoded on **B300** (2026-09-30), M and C2 on **H200·0.51**. The rule is one hardware column per table; an H200 re-run of
  Ins on the last 300 is the next job to request before any number is quoted outside.
- C2's outputs are longer and partly degenerate (repeated "The final answer is …"; mean 3 029 characters vs Ins 888); lr 1e-4 is on the high
  side for plain self-SFT. It does not change its accuracy (38.0) but it means C2 is a conservative control, not a tuned one.
- Reading rule from the plan still applies: success = ≥ Ins + 6 **and** > C2 + 3 **and** no guard regression.

## TODO (2026-10-16, @Amel — please read / add)

**Validation set — DONE 2026-10-16 (user: "we need validation data to confirm whether each multi-stage round is meaningful").**
Three fixed random-100 lists in `code/step3/guards/`, all disjoint from every reported test set; one checkpoint = 3 parallel jobs,
`modules/submit_step3_validation.sh <ckpt> <label>` (≈ 6.5 GPU·h ≈ 3.3 CHF), results in `outputs/step3/validation/<label>/{math,ifeval,mbpp}.json`:

| family | list | protocol | GPU·h |
|---|---|---|---:|
| math | `val_math_test_minus500_rand500_seed0.jsonl` rows 0–99 — random 100 of the **MATH test split minus MATH-500** (4 500 problems; MATH-500 is a uniform random subset of that split, so the difficulty profile is the same: L1–L5 = 9/18/21/26/27 % vs the pool's 10/19/21/25/25 %; disjoint from every reported id and from the rollout pool, which is MATH *train*). Built by `build_math_val.py` (official minerva gold extraction, checked on all 500 MATH-500 answers; official prompt template byte-for-byte). **Replaces** the MATH-500 ids 0–199 list of 2026-10-16, rejected by the user on 2026-10-17 because the rand100-of-500 test column also draws from ids 0–199 (`trash/val_math500_first200_rejected_20261017/`) | the test decoder (lockstep free @1024), `modules/run_math_val.sh` | 1.5 |
| instruction following | `val_ifeval_rand100_seed1_disjoint.json` — random 100 of the 441 IFEval prompts not in the reported `ifeval_rand100` | official 1280/1280, task `ifeval_val100` | 3.5 |
| coding | `val_mbppplus_rand100_seed0.json` — random 100 of the 378 MBPP+ tasks (HumanEval+ stays a test set) | official evalplus dllm 512/512, `evalplus_subset.py` | 0.6 |

Reference points on the new math val100 are being decoded (Ins, distilled M; 2026-10-17); the old list's free numbers (Ins 46, M(.4,.6) 51, A(2.0) 61, gate 65) no longer apply.
**Rule (user 2026-10-17): every training variant from now on is tuned on this validation set only — recipe search, θ, frames, multi-stage rounds, tulu3-1k, controls. Not all three families per variant: math val100 first (+ the cheap MBPP+ 100); IFEval val100 only for the variant math picks and for the Ins / M references.** Recipe search (2026-10-17, `modules/submit_step3_recipe.sh` = train + validation): one knob each vs M — epochs 4, random half of the
rollouts (500, seed 0), θ 0.6, θ 0.4. The test sets are touched once, with the recipe validation picks.
Not used for anything yet: GSM8K (too easy, moves in the wrong direction under merging). Hand-written sanity prompts: still to add.

**First mission (@Amel), before anything else below:** train the distilled model on the Tulu-3-only pool (`tulu3-1k`: 1 000 prompts sampled directly
from Tulu-3 at its native proportions, nothing else), with the recipe exactly as it is (Ins ⇐ A(2.0), θ = 0.8, λ = 0.3, 2 epochs), and then
test generalisation on all three families: **math** (MATH-500 last 300, GSM8K), **coding** (HumanEval+), **instruction following** (IFEval).
The question is only: does the Tulu-only student keep the MATH gain of mixpool-1k *and* lose nothing on code / IF. Pipeline: `code/step3/pipeline`
(tag `tulu3-1k`, arm M; guards via `guard_arms`). Status 2026-10-16: not done — the first tulu3-1k attempt trained on the wrong pool (see "Invalid tulu3-1k run"); the fixed pipeline is ready to relaunch. Everything else in this list comes after that answer.

Ordered by what each one changes about the claim. Costs are H200·0.51 GPU·h. The two bigger items (θ search, multi-stage distillation) have their own sub-sections below.
0. **Fix one recipe first** (user 2026-10-16) — the current recipe (1 k prompts, 2 epochs, 1024-token sequences, θ 0.8, λ 0.3) was never tuned, and every
   later run should use ONE unchanged recipe. Sequence length stays fixed at 1024 (the official MATH-500 setting; do not touch). Vary only the amount of
   training, one knob at a time, same MATH last 300 + guards:
   (a) **epochs** 2 → 4 (same rollouts; 370 steps, ≈ 0.5 h training + 7 GPU·h eval);
   (b) **number of rollout sequences** 1 k → 2 k prompts (new rollouts, 2 epochs; ≈ 10 GPU·h rollouts + eval);
   (c) **frames per rollout** 3 → 6. A rollout is not trained on as one sequence: from its 512 denoising steps we pick 3 intermediate states ("frames":
   2 just before a low-confidence commit + 1 uniform) and each frame is one training example. Doubling it = more examples from the same rollouts,
   no new decoding (`--low-frames 4 --uniform-frames 2`, ≈ 0.5 h training + 7 GPU·h eval).
   Amel decides the order / which of (a)–(c) to run. If none beats 50.3 by more than noise, the 185-step recipe is final; if one does, that is the recipe.
   Do this before θ search and multi-stage.
1. **Clean training set** (`tulu3-1k`) — first attempt 2026-10-16 invalid (see "Invalid tulu3-1k run"); re-queued with the fixed pipeline, ≈ 20 GPU·h. Then guards.
2. **θ search** — see §TODO-θ below. ≈ 8 GPU·h per point.
3. **Multi-stage distillation** — see §TODO-multi-stage below. Same GPU·h as offline, serial wall-clock.
4. **Same hardware for the baselines** — Ins and M(.4,.6) are B300 numbers; re-decode on H200 (last 300, 2 jobs, ≈ 5 GPU·h).
5. **Guards** — IFEval rand100 / HumanEval+ done (see "Generalisation"); still open: GSM8K, IFEval full 541, and a length read (M is +80 % tokens vs Ins).
6. **Remaining controls** on the same rollouts (≈ 8 GPU·h each): C1 no split, C3 teacher A(1.1), C4 random mask, C5 forward KL; C2 at a lower lr.
7. **Second seed + 8 k prompts** — is 1 k the plateau, is 50.5 stable.
8. **Trainer checkpoint / resume** — the mixpool-1k M container restarted twice silently; add periodic LoRA + optimizer saves before any multi-hour run.
9. **Second student line** — M(.4,.6) init (arm Mm), later a second model family (DreamReasoner / LLaDA-Instruct).

### TODO-θ: threshold search
We used the gate's θ = 0.8 unchanged. Grid in steps of 0.2 (0.1 was too expensive): θ ∈ {0.4, 0.6, 0.8}, plus the decaying schedule 0.8 → 0.5
(`--theta-end`, arm Msch). Rollouts are shared, so each point = 0.5 h training + 7 GPU·h MATH. Read: acc, finish rate, avg length, and `unsure_rate`
(share of positions that went to the teacher) — the latter tells whether a lower θ really means less teacher.

### TODO-multi-stage: online distillation (roll out with the updated student)

**In one sentence.** Today (offline): S₀ = Ins rolls out all 1 000 prompts once, then we train once → S₁. Online: split the prompts into K chunks;
roll out chunk 1 with S₀, train → S₁; roll out chunk 2 **with S₁**, train further → S₂; … → S_K. Teacher T, loss and θ never change;
only *who writes the trajectories* changes.

**Notation.** S₀ = Ins; T = A(2.0), frozen; S_k = the student after round k (S₀ plus one LoRA that keeps being trained, so S_k starts from S_{k−1});
Q_k = the k-th chunk of prompts; p_S(·|x, j) = model S's distribution at position j of canvas x; c_j = max_v p_S(v|x, j) = the student's confidence there.

**Round k = 1, …, K**
1. **Rollout.** S_{k−1} decodes every prompt in Q_k with the official sampler (512 tokens / 512 steps); record the canvas, and per position the step it
   was written and the confidence at that moment.
2. **Frames.** Per rollout, 3 training states (x, NEXT): 2 taken just before a commit with confidence < θ, 1 at a uniform step — as in the offline run.
3. **Train.** Starting from S_{k−1}, on those frames,

   S_k = argmin_S  E_{(x, NEXT)} Σ_{j ∈ NEXT} [ 1[c_j < θ] · KL( p_S(·|x,j) ‖ p_T(·|x,j) )  +  λ · 1[c_j ≥ θ] · KL( p_S(·|x,j) ‖ p_{S₀}(·|x,j) ) ]

   i.e. where S is unsure it moves toward T; where S is sure it is held at S₀. c_j is computed with the S being trained.

After round K, S_K is merged and evaluated like the offline model. K = 1 is exactly the offline run.

**Initial plan.** 1 000 Tulu-3 prompts (the tulu3-1k pool), K = 4 chunks of 250, θ = 0.8, λ = 0.3, 2 epochs per round, lr 1e-4 with warm-up only in round 1;
compare S₄ against the offline S₁ on the same MATH last 300 (+ guards). GPU·h ≈ offline (every prompt decoded once, every frame trained once);
wall-clock ≈ 4 × (1.4 h rollout + 0.1 h train). Expected gain: from round 2 on, the "unsure" positions are the ones the student *still* gets wrong,
not the ones already fixed (TOPD's on-policy argument). Price: controls C1 … C5 can share only the round-1 rollouts.

**Validation gate per round (user 2026-10-16).** After every round k, run the validation set on S_k (`submit_step3_validation.sh`, ≈ 6.5 GPU·h).
Continue to round k+1 only if MATH-val(S_k) ≥ MATH-val(S_{k−1}) and neither IFEval-val nor MBPP+-val dropped by more than 3 points; otherwise
stop and keep S_{k−1}. The test sets are touched once, with the round that validation picked.

**Later / needed**
- Variant: moving teacher T_k = over step re-applied on top of S_k.
- Optional: decaying θ_k (TODO-θ); test the two separately first.
- Code: `rollout.py --lora` (decode with the adapter), `--resume-lora` in the trainer, Snakefile round dependencies.

## Next
1. Finish the 300 (automatic). 2. Ins on H200, last 300 (one job). 3. Guards on M. 4. 8 k offline vs 4 × 2 k online rounds (`tmp.md` 2026-10-15 plan),
M06 (θ 0.6) and Mm (M(.4,.6) init) as the next arms.
