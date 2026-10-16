# STEP3 — Distillation: one model that does what the gate did (2026-10-16, in progress)

STEP2 is closed (`step2_verdict.md`): the decode-time gate Ins ⇐ A(2.0) @0.8 is the ceiling (49.0, 2× cost) and nothing combined beyond it.
STEP3 trains that behaviour into a single 1× model. Recipe: `technical_details/step3_training_formulation.md`; plan and reading rules:
`technical_details/step3_plan_frontier_kd.md`.

## Takeaways (1k test, 2026-10-16)
1. **A 1× model reaches the 2× gate's accuracy** after 185 optimizer steps on 1 k training prompts (half an hour of training):
   50.5 vs Ins 38.5 on the same 200 MATH-500 problems, p < 0.001.
2. **The teacher is what matters.** Self-SFT on the same rollouts with the same trainer (C2) stays at Ins (38.0, p = 1.0 vs Ins).
3. The gain has the gate's signature: finish rate 77 → 94 % **and** accuracy on finished problems 50 → 54 (the gate: 93 % / 52.9).

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
Checkpoints: `outputs/checkpoints/step3-t1k-{M,C2}/{lora,merged}`, logs `train_log.jsonl`, args `args.json`.

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
`pipeline/config.yaml: prompt_set: tulu3`). The t1k numbers above were obtained on the earlier mix17k pool and stay labelled as such.

The **n1k run** (launched 2026-10-16) uses ids 0–999 of this pool: 415 math (personahub_math 179, NuminaMath-TIR 84, personas-math-grade 64,
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

## Generalisation: what is and is not shown yet
- Shown: the gain transfers from **mixed training prompts** (half of them not math) to **held-out competition math**, with the student
  decoding alone. The test set is the same held-out set on which STEP0–2 measured Ins, the merges, the teachers and the gate, so all numbers
  are comparable (hardware caveat below).
- Not shown yet: the three guards (GSM8K, HumanEval+, IFEval; official harnesses, `pipeline/config.yaml: guard_arms`) — these decide whether M
  **lost** anything elsewhere. They run after the MATH read, on the arm that clears the MATH bar.
- Not shown yet: a second seed / a second model family; and the 8 k / online runs that tell whether 1 k is already the plateau.

## Results (1k test; table regenerated by `code/step3/results_table.py` → `technical_details/step3_results_t1k.md`)

Every row below is decoded by **one model** on the same MATH-500 problems (last 300 ∩ what every row has; 200 while M's last two shards
run), except the gate row (2×, two models at decode time). The two **end-point baselines** of STEP0 are in the table: the best merge
M(.4,.6) (convex line) and the over addition A(2.0) (addition line; it is also the teacher). McNemar = wins-of-row / wins-of-column, exact p.

| row | acc | finish | acc\|finished | avg len (tok) | vs Ins | vs C2 | vs M(.4,.6) | vs A(2.0) | HW |
|---|---:|---:|---:|---:|---|---|---|---|---|
| Ins (student, untouched) | 38.5 | 77% | 50.0 | 356 | — | 26/25, p=1.00 | 17/26, p=0.22 | 22/41, p=0.02 | B300 |
| M(.4,.6) best merge (end point 1) | 43.0 | 84% | 50.9 | 433 | 26/17, p=0.22 | 28/18, p=0.18 | — | 23/33, p=0.23 | B300 |
| A(2.0) over addition (end point 2 = the teacher) | 48.0 | 94% | 50.8 | 1009 | 41/22, p=0.02 | 36/16, p=0.01 | 33/23, p=0.23 | — | H200 |
| gate Ins ⇐ A(2.0) @0.8 (2×) | 50.5 | 93% | 54.3 | 792 | 38/14, p=0.00 | 40/15, p=0.00 | 30/15, p=0.04 | 27/22, p=0.57 | H200 |
| **M** = Ins ⇐ A(2.0) distilled, **θ = 0.8**, λ_anchor 0.3, reverse KL, lr 1e-4, 2 ep, LoRA r64 (1×) | **50.5** | **94%** | **54.0** | 644 | 39/15, p=0.00 | 40/15, p=0.00 | 32/17, p=0.04 | 28/23, p=0.58 | H200 |
| **C2** = Ins self-SFT (CE on its own rollout token, no teacher), lr 1e-4, 2 ep, LoRA r64 (1×) | 38.0 | 81% | 46.9 | 857 | 25/26, p=1.00 | — | 18/28, p=0.18 | 16/36, p=0.01 | H200 |

**θ (the threshold) in M** is the student's raw confidence below which a frame is handed to the teacher: at the position the sampler writes next,
c = max_v p_S(v | state); c < θ → reverse KL to A(2.0); c ≥ θ → KL to untouched Ins (weight λ_anchor). The same θ also selects the training
frames (2 of 3 frames per canvas are taken right before a commit whose rollout-time confidence was < θ). θ = 0.8 is the decode-time gate's value
(STEP1 sweep); arms M06 (θ 0.6) and Msch (0.8 → 0.5) are the planned θ ablations. Flags: `--theta`, `--theta-end`, `--lambda-anchor`, `--kl`.
avg len = mean response length in Dream tokens up to the end of the decoded text (the canvas is always 1024).

What the table says, in order of confidence:
1. **M ≫ Ins and M ≫ C2** (both p < 0.001, 39–40 wins vs 15): the distilled 1× model is better than what it started from, and the teacher is
   the reason (self-SFT does nothing).
2. **M ≥ both end points.** M beats the best merge (32/17, p = 0.04) and is at least level with the over addition / teacher (28/23, p = 0.58)
   while being a 1× model with Ins's weights + a LoRA — no merge, no vector arithmetic at test time.
3. **M = the 2× gate it was distilled from** (50.5 vs 50.5; the gate is the ceiling we set out to reach).
4. It has the gate's two signatures: finish rate 77 → 94 % (teacher-level) and accuracy on finished problems 50 → 54 (above the teacher's 50.8).

Training provenance for these two runs: `outputs/checkpoints/step3-t1k-{M,C2}/{args.json,train_log.jsonl,run.log}`; W&B project `a2d-step3`,
runs `step3-t1k-M` / `step3-t1k-C2` (back-filled from the logs). Note: the M job's container restarted twice before completing (`run.log` shows
three starts, one `[done]`); the checkpoint is from the single complete run, the log file holds the partial attempts too.

## What a truly useful evaluation needs (and which of it we have)
| requirement | status | action |
|---|---|---|
| same hardware for every row | Ins and M(.4,.6) are B300 numbers (2026-09-30); all STEP3 rows H200 | re-decode Ins and M(.4,.6) on H200, last 300 (2 jobs, ≈ 5 GPU·h) |
| full 300 problems, not 200 | M's shards 0 and 2 running | automatic; table regenerates |
| the model did not lose elsewhere | not measured | GSM8K, HumanEval+, IFEval official harnesses on M (`guard_arms`), ≈ 20 GPU·h |
| not a seed artefact | one seed | second training seed of M (0.5 h + 7 GPU·h eval) |
| not a 1k artefact | one data size | 8k offline (or 4 × 2k online) — also tells whether 1k is already the plateau |
| statistics that respect pairing | done: exact McNemar on identical problem ids | keep; never compare accuracies across different id sets |
| no leakage | checked (previous section) | re-check whenever the prompt pool changes |
| the controls that kill alternative stories | C2 (teacher needed?) done; C1 (selection needed?), C3 (over needed?), C4 (replayed state needed?), C5 (KL direction) not run | run after the guards, on the same rollouts (0.5 h training each) |

## How the training code was built
Written for this step, not adapted from a trainer: `code/step3/train_frontier_kd.py` is ≈ 300 lines of plain PyTorch + HF `transformers` +
`peft` (LoRA), single GPU, no Lightning, no verl, no accelerate. What is reused unchanged: Dream's official sampler for the rollouts
(`diffusion_generate`, same settings as the MATH-500 protocol), Dream's logit convention (position p is predicted from hidden p−1, as in the
official SFT trainer and in our decoder), peft for LoRA exactly as the official `run_sft_tulu3.sh` enables it, and our STEP1 lockstep decoder
for evaluation. What is new: trajectory replay as the corruption, the confidence-split loss, the self-anchor, the frame sampler, the OOM
micro-batch probe, and the merge/save path (`merge_lora.py`, working around Dream's `generation_config.validate`). The official verl FSDP
trainer was not used because its loss (random-mask CE on reference answers) is exactly what this step replaces, and it needs 4 GPUs.

## Caveats on the current numbers
- 200 of the 300 problems are in (M shards 0 and 2 are being re-run after a job-tracking bug, 2026-10-16 morning); the full-300 column and
  `technical_details/step3_results_t1k.md` follow.
- The Ins row was decoded on **B300** (2026-09-30), M and C2 on **H200·0.51**. The rule is one hardware column per table; an H200 re-run of
  Ins on the last 300 is the next job to request before any number is quoted outside.
- C2's outputs are longer and partly degenerate (repeated "The final answer is …"; mean 3 029 characters vs Ins 888); lr 1e-4 is on the high
  side for plain self-SFT. It does not change its accuracy (38.0) but it means C2 is a conservative control, not a tuned one.
- Reading rule from the plan still applies: success = ≥ Ins + 6 **and** > C2 + 3 **and** no guard regression.

## TODO (2026-10-16, @Amel — please read / add)
Ordered by what each one changes about the claim. Costs are H200·0.51 GPU·h. The two bigger items (θ search, multi-stage distillation) have their own sub-sections below.

1. **Clean training set** (running: `n1k`) — redo M on Tulu-3-only native proportions, same 1 k / MATH last 300. ≈ 20 GPU·h. Pass = M ≥ Ins + 6 again.
2. **θ search** — see §TODO-θ below. ≈ 8 GPU·h per point.
3. **Multi-stage distillation** — see §TODO-multi-stage below. Same GPU·h as offline, serial wall-clock.
4. **Same hardware for the baselines** — Ins and M(.4,.6) are B300 numbers; re-decode on H200 (last 300, 2 jobs, ≈ 5 GPU·h).
5. **Guards** — GSM8K / HumanEval+ / IFEval on M (≈ 20 GPU·h), plus a length read (M is +80 % tokens vs Ins).
6. **Remaining controls** on the same rollouts (≈ 8 GPU·h each): C1 no split, C3 teacher A(1.1), C4 random mask, C5 forward KL; C2 at a lower lr.
7. **Second seed + 8 k prompts** — is 1 k the plateau, is 50.5 stable.
8. **Trainer checkpoint / resume** — the t1k M container restarted twice silently; add periodic LoRA + optimizer saves before any multi-hour run.
9. **Second student line** — M(.4,.6) init (arm Mm), later a second model family (DreamReasoner / LLaDA-Instruct).

### TODO-θ: threshold search
We used the gate's θ = 0.8 unchanged. Grid in steps of 0.2 (0.1 was too expensive): θ ∈ {0.4, 0.6, 0.8}, plus the decaying schedule 0.8 → 0.5
(`--theta-end`, arm Msch). Rollouts are shared, so each point = 0.5 h training + 7 GPU·h MATH. Read: acc, finish rate, avg length, and `unsure_rate`
(share of positions that went to the teacher) — the latter tells whether a lower θ really means less teacher.

### TODO-multi-stage: online distillation (roll out with the updated student)
**Setup**
- S₀ = Ins (student init), T = A(2.0) (teacher, frozen), θ fixed (whatever TODO-θ picks), λ = 0.3.
- One LoRA Δ, carried across rounds; Δ₀ = 0, so S_k = S₀ + Δ_k.
- Prompts split into disjoint chunks Q₁ … Q_K (e.g. K = 4 × 250 for a 1 k run).

**Round k = 1 … K**
- (a) Rollout with the current student: D_k = { Sample(S_{k−1}, q) : q ∈ Q_k }. Each sample = (y, s, κ): canvas, commit step and commit confidence per position.
- (b) Frames: F_k = { (x(t), NEXT(t)) : τ ∈ D_k, t ∈ frames(τ, θ) } — as now, 2 frames before a low-κ commit + 1 uniform.
- (c) Update Δ from Δ_{k−1}:
  S_k = argmin_Δ E_{(x,j)∈F_k} [ 1[c_j < θ] · KL( p_{S₀+Δ}(·|x,j) ‖ p_T(·|x,j) ) + λ · 1[c_j ≥ θ] · KL( p_{S₀+Δ}(·|x,j) ‖ p_{S₀}(·|x,j) ) ]
  with c_j = max_v p_{S₀+Δ}(v | x, j) the *current* student's confidence; the anchor is always the original S₀.

**Properties**
- K = 1 is exactly the offline run we have.
- Each prompt is rolled out once → same GPU·h as offline; wall-clock becomes serial (K × (rollout + train)).
- Why it should help: frames are the student's current states, so a mistake fixed in round k is not re-learned from stale prefixes in round k+1 (TOPD's on-policy argument).
- Cost of the design: controls cannot share rollouts after round 1.

**Later / needed**
- Variant: moving teacher T_k = over step re-applied on top of S_k.
- Optional: combine with a θ schedule θ_k (TODO-θ), but test the two separately first.
- Code: `rollout.py --lora`, `--resume-lora` in the trainer, Snakefile round dependencies.

## Next
1. Finish the 300 (automatic). 2. Ins on H200, last 300 (one job). 3. Guards on M. 4. 8 k offline vs 4 × 2 k online rounds (`tmp.md` 2026-10-15 plan),
M06 (θ 0.6) and Mm (M(.4,.6) init) as the next arms.
