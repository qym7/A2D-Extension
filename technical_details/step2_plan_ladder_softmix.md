# STEP2 plan — two extensions of the content gate on one line (proposed 2026-10-13, revised after user feedback: mean, not ladder; no λ)

Both stay on the clean line **Ins ⇐ A(β)**: student S = Ins, teachers on the addition line A(β) = D_b + τ_D + β·τ_A.
Protocol unchanged: MATH-500 last 300 (ids 200–499), lockstep @1024 (one position per step), top_p 0.9, conf_temp 0.1, argmax token,
H200·0.51, 3 shards per arm. The **student always picks the position** (settled in `step1_methods.md` §2); only the token rule changes.

## 0. Notation (one step)
At step t the student S proposes position j_t (its most confident masked position) and, for every model m ∈ {S, T_1, …}, let
p_m = softmax(z_m(j_t)) be that model's raw next-token distribution at j_t (T = 1, no top-p), c_m = max p_m its confidence, y_m = argmax p_m its token.

| rule | token written at j_t | existing result (last 300) |
|---|---|---|
| student alone | y_S | 36.7 |
| content gate @θ (current) | y_S if c_S ≥ θ else y_T | Ins ⇐ A(2.0) @0.8 = 49.0 |
| teacher alone | y_T | A(2.0) = 46.7 |

## Experiment A — teacher mean along β (user, 2026-10-13: mean rather than a ladder)
Teachers T_1 = A(0.5), T_2 = A(1.0), T_3 = A(2.0). At a low-confidence step the token is the argmax of the **mean of the teachers'
probability distributions**; the student's token otherwise.

    y = y_S                                     if c_S ≥ θ
      = argmax[ (p_T1 + p_T2 + p_T3) / 3 ]      otherwise

Why mean rather than vote: three teachers on one line mostly agree, so a majority vote reduces to "A(2.0) unless the two milder ones
agree against it" and needs a tie rule; the mean uses the full distributions, has no tie, and is the same operation as Experiment B
(so one code path). The mean of *outputs* along β is also a different object from the weight-space midpoint already tested
((0.7,1.3) = 44.3, which failed) — that is exactly the comparison worth having. For free, the run logs how often a 3-way vote
would have chosen a different token than the mean (`n_vote_differs`), so the vote variant is answered without a second run.

Arms (all @0.8):
| arm | compute | purpose |
|---|---:|---|
| A1 Ins ⇐ A(0.5) @0.8 | 2× | single-teacher reference for T_1 (new) |
| A2 Ins ⇐ A(1.0) @0.8 | 2× | single-teacher reference for T_2 (new; A(1.1) @0.8 exists on the first 200 only = 57.5) |
| A3 Ins ⇐ mean{A(0.5), A(1.0), A(2.0)} @0.8 | 4× | the method |
| — Ins ⇐ A(2.0) @0.8 | 2× | exists: 49.0 (= T_3 alone) |

Reading rules, fixed now:
- **Success**: A3 > max(A1, A2, 49.0) by ≥ 3 points on the last 300 with McNemar p < 0.1 → averaging teachers along β adds information a single teacher lacks.
- **Kill**: A3 within 2 points of the best single-teacher arm → "more teachers on one line" is dead (consistent with `step2_cascade_two_lines.md` §4).
- Diagnostic: rate 2 of A3 vs the singles, and `n_vote_differs`. If the mean's token equals A(2.0)'s on > 97 % of substituted tokens, A3 is A(2.0) in disguise regardless of accuracy.
- Prior: kill; expected A3 ∈ [48, 51].

## Experiment B — soft mixing (mean of student and teacher) at the low-confidence steps (user: plain mean, no λ to tune)
Rule `content_softmix @θ`, T = A(2.0):

    y = y_S                            if c_S ≥ θ
      = argmax[ (p_S + p_T) / 2 ]      otherwise

Probability space, raw softmax (T = 1), the same p the gate reads. The adaptive-λ variant is dropped (too many knobs).

Arms:
| arm | θ | compute | purpose |
|---|---:|---:|---|
| B1 softmix @0.8 | 0.8 | 2× | mean at unsure steps |
| B3 always-mix @1.0 | 1.0 | 2× | **control: plain 2-model probability ensemble** (every step mixes); decides whether the gate is needed at all |
| — content gate @0.8 | 0.8 | 2× | exists: 49.0 (hard replacement) |

Reading rules, fixed now:
- **Gate matters**: B1 > B3 by ≥ 3 points (p < 0.1). Otherwise the method is "a 2-model probability ensemble" (prior art: logit averaging) and the gate is a compute saver only — must be written that way.
- **Soft beats hard**: B1 > 49.0 by ≥ 2 points (p < 0.1). Otherwise hard replacement stays (simpler; distillation targets need hard tokens anyway).
- Prior: B3 ≈ 47–50 (at the 94 % of steps where both are confident the mean's argmax is the student's own token, so always-mix ≈ a gate with a lower effective θ); B1 ≈ 49 ± 2; "gate matters" likely fails or is marginal. This is the experiment that decides whether the method is distinguishable from an ensemble.

## Implementation (no edits to files that running jobs use)
- `scripts/step1_lockstep_gated_v2.py` = copy of `step1_lockstep_gated.py` + one new mode `content_mean` (`--other` takes a comma-separated list of 1–3 teachers; at a low-confidence step the token is the argmax of the mean of the listed models' distributions, with `--with-student` adding the student to the mean). A3 = three teachers without the student; B1/B3 = one teacher with the student. Content gate / both / random code paths untouched (byte-identical outputs on the existing modes is
  checked by re-running one existing problem in the smoke).
- `modules/run_gated_shard5.sh` = shard4 pointing at v2, with mode tokens `mean:θ` (teachers only) and `smix:θ` (student + teacher).
- Checkpoints: A(0.5) and A(1.0) built once at `outputs/checkpoints/add-a10-b05`, `add-a10-b10` (build-at-host, flock-serialised, as before).
- Gates before submission: `py_compile`, CPU dry run of the token rule on saved logits, one 2-problem smoke per new mode on the GPU, then the 6 arms × 3 shards.
- Output dirs: `details/logs/gated_ins_add-a10-b05_1024/math500/t08-last300-s*`, `…b10…/t08-…`, `details/logs/mean_ins_1024/math500/b05-b10-b20-t08-last300-s*`,
  `details/logs/softmix_ins_add-a10-b20_1024/math500/{t08,t10}-last300-s*`.

## Cost
5 arms × 3 shards = 15 jobs; 2× arms ≈ 6 h per shard, the 4× mean ≈ 12 h per shard at 0.51 → ≈ 35 GPU·h (H200·0.51 units), one day of wall
time if all 15 schedule at once. Teachers are called every step (no lazy call yet), so A3's 4× is the price of the diagnostic.

## What each outcome changes in STEP2
| A3 | B1 vs B3 | consequence |
|---|---|---|
| kill | gate matters | pipeline stays "one teacher, hard gate"; teacher-axis expansion only across lines (cascade) |
| kill | gate does not matter | method = probability ensemble with a confidence-based compute saver; rewrite §1 accordingly |
| success | any | teacher mean becomes the pipeline's expansion rule; add A(1.5) / drop A(0.5) to test what the mean needs |
