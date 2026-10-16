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

## Caveats on the current numbers
- 200 of the 300 problems are in (M shards 0 and 2 are being re-run after a job-tracking bug, 2026-10-16 morning); the full-300 column and
  `technical_details/step3_results_t1k.md` follow.
- The Ins row was decoded on **B300** (2026-09-30), M and C2 on **H200·0.51**. The rule is one hardware column per table; an H200 re-run of
  Ins on the last 300 is the next job to request before any number is quoted outside.
- C2's outputs are longer and partly degenerate (repeated "The final answer is …"; mean 3 029 characters vs Ins 888); lr 1e-4 is on the high
  side for plain self-SFT. It does not change its accuracy (38.0) but it means C2 is a conservative control, not a tuned one.
- Reading rule from the plan still applies: success = ≥ Ins + 6 **and** > C2 + 3 **and** no guard regression.

## Next
1. Finish the 300 (automatic). 2. Ins on H200, last 300 (one job). 3. Guards on M. 4. 8 k offline vs 4 × 2 k online rounds (`tmp.md` 2026-10-15 plan),
M06 (θ 0.6) and Mm (M(.4,.6) init) as the next arms.
