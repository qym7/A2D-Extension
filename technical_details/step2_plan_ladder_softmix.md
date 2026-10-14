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
Gated exactly like the content gate (user, 2026-10-13): the mean is consulted **only at steps where c_S < θ = 0.8**; elsewhere the student writes its own token.
Teachers T_1 = A(1.5), T_2 = A(2.0), T_3 = A(2.5) — a regular grid centred on the known-good over-addition teacher (user, 2026-10-13: 0.5/1.0/2.0 was an odd set; A(3.0) is collapsed so 2.5 is the last usable point above). At a low-confidence step the token is the argmax of the **mean of the teachers'
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
| A1 Ins ⇐ A(1.5) @0.8 | 2× | single-teacher reference for T_1 (new) |
| A2 Ins ⇐ A(2.5) @0.8 | 2× | single-teacher reference for T_3 (new; past the peak, before the A(3.0) collapse) |
| A3 Ins ⇐ mean{A(1.5), A(2.0), A(2.5)} @0.8 | 4× | the method |
| — Ins ⇐ A(2.0) @0.8 | 2× | exists: 49.0 (= T_2 alone) |

Reading rules, fixed now:
- **Success**: A3 > max(A1, A2, 49.0) by ≥ 3 points on the last 300 with McNemar p < 0.1 → averaging teachers along β adds information a single teacher lacks.
- **Kill**: A3 within 2 points of the best single-teacher arm → "more teachers on one line" is dead (consistent with `step2_cascade_two_lines.md` §4).
- Diagnostic: rate 2 of A3 vs the singles, and `n_vote_differs`. If the mean's token equals A(2.0)'s (T_2) on > 97 % of substituted tokens, A3 is A(2.0) in disguise regardless of accuracy.
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
- `scripts/step1/lockstep_gated_v2.py` = copy of `step1_lockstep_gated.py` + one new mode `content_mean` (`--other` takes a comma-separated list of 1–3 teachers; at a low-confidence step the token is the argmax of the mean of the listed models' distributions, with `--with-student` adding the student to the mean). A3 = three teachers without the student; B1/B3 = one teacher with the student. Content gate / both / random code paths untouched (byte-identical outputs on the existing modes is
  checked by re-running one existing problem in the smoke).
- `modules/run_gated_shard5.sh` = shard4 pointing at v2, with mode tokens `mean:θ` (teachers only) and `smix:θ` (student + teacher).
- Checkpoints: A(1.5) and A(2.5) built once at `outputs/checkpoints/add-a10-b15`, `add-a10-b25` (build-at-host, flock-serialised, as before).
- Gates before submission: `py_compile`, CPU dry run of the token rule on saved logits, one 2-problem smoke per new mode on the GPU, then the 6 arms × 3 shards.
- Output dirs: `details/logs/gated_ins_add-a10-b15_1024/math500/t08-last300-s*`, `…b25…/t08-…`, `details/logs/mean_ins_1024/math500/b15-b20-b25-t08-last300-s*`,
  `details/logs/softmix_ins_add-a10-b20_1024/math500/{t08,t10}-last300-s*`.

## Cost
5 arms × 3 shards = 15 jobs; 2× arms ≈ 6 h per shard, the 4× mean ≈ 12 h per shard at 0.51 → ≈ 35 GPU·h (H200·0.51 units), one day of wall
time if all 15 schedule at once. Teachers are called every step (no lazy call yet), so A3's 4× is the price of the diagnostic.

## What each outcome changes in STEP2
| A3 | B1 vs B3 | consequence |
|---|---|---|
| kill | gate matters | pipeline stays "one teacher, hard gate"; teacher-axis expansion only across lines (cascade) |
| kill | gate does not matter | method = probability ensemble with a confidence-based compute saver; rewrite §1 accordingly |
| success | any | teacher mean becomes the pipeline's expansion rule; widen/narrow the grid (1.0–2.5 vs 1.75–2.25) to test what the mean needs |

## Smoke result for B (2026-10-13, ids 200–201, zero new knowledge needed to read it)
`smix @0.8` and `smix @1.0` produce **identical texts**. At θ = 1.0 every one of the 2048 committed positions is mixed, and the mean's argmax
is the student's own token at 2026 of them; the 22 changed tokens are exactly the 22 that @0.8 changes (its 159 low-confidence steps keep the
student's token 137 times). So for the student+teacher mean the gate is a pure compute saver and B3 is redundant; B reduces to B1 @0.8.
It also says what B1 will be: an effective token-change rate of ≈ 1 % (hard gate @0.8: ≈ 5 %), i.e. a much lower dose. Expectation: B1 lands
between the student and the hard gate, and "soft beats hard" fails. Not yet run on the last 300 (user hold on submissions).

## Results (2026-10-14, last 300, H200·0.51)
| arm | acc | finish | acc on finished | rate 1 / rate 2 | vs Ins | vs A(2.0) | vs hard gate 49.0 |
|---|---:|---:|---:|---|---|---|---|
| B1 soft mix Ins+A(2.0) @0.8 | **46.3** | 87 % | 53.1 | 8.2 % / 1.32 % | 54/25, 0.00 | 39/40, 1.00 | 32/40, 0.41 |
| A1 Ins ⇐ A(1.5) @0.8 | 48.3 | 95 % | 51.1 | 11.4 % / 3.24 % | 55/20, 0.00 | 40/35, 0.64 | 28/30, 0.90 |
| A2 Ins ⇐ A(2.5) @0.8 | 47.0 | 92 % | 51.1 | 15.6 % / 6.60 % | 54/23, 0.00 | 33/32, 1.00 | 30/36, 0.54 |
| extra: Ins ⇐ A(1.0) @0.8 (teacher not over) | 44.3 | 91 % | 48.9 | 9.7 % / 1.75 % | 48/25, 0.01 | — | 27/41, 0.11 |
| A3 Ins ⇐ mean{A(1.5), A(2.0), A(2.5)} @0.8 | not run | | | | | | |

Reading by the pre-registered rules:
- **B — "soft beats hard" fails.** B1 = 46.3 vs hard gate 49.0 (32/40, p = 0.41): as the smoke predicted, averaging with the student lowers the dose (rate 2 1.3 % vs 5.0 %) and lands between the student and the hard gate, ≈ the teacher alone. Hard replacement stays. "Gate matters" was already answered by the smoke (B3 ≡ B1, the gate is a compute saver for the soft rule).
- **A — not run; kill by prior.** The three single-teacher arms are a plateau (48.3 / 49.0 / 47.0, every pair p > 0.5), their correct sets overlap no more than two θ values of one arm, and the 3-way majority vote (48.3) equals a single arm (`step2_cascade_two_lines.md` §5). A mean of their distributions has nothing to add; the 4× run is not worth its cost.
- **β rule for the pipeline**: on the Ins line the gain needs an *over* teacher — β = 1.0 (≈ best addition) gives 44.3, β ∈ [1.5, 2.5] gives 47–49, A(3.0) collapsed. Rate 2 grows monotonically with β (1.75 → 3.24 → 4.96 → 6.60 %) while accuracy is flat past 1.5, so the rule "smallest β past the accuracy peak of the line" (here 1.5) costs nothing and changes the fewest tokens.
