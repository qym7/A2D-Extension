# STEP2 plan — two extensions of the content gate on one line (proposed 2026-10-13, not submitted)

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

## Experiment A — teacher ladder along β (escalate to the stronger teacher only when the milder one is also unsure)
Teachers T_1 = A(0.5), T_2 = A(1.0), T_3 = A(2.0) (β ascending = increasingly AR-like, increasingly far from the student).

Rule `content_ladder @θ`: write the token of the **first** model in the ordered list [S, T_1, T_2, T_3] whose confidence is ≥ θ; if none, T_3's token.

    y = y_S   if c_S ≥ θ
      = y_T1  elif c_T1 ≥ θ
      = y_T2  elif c_T2 ≥ θ
      = y_T3  otherwise

Logged per problem: the share of committed tokens taken from each of the four levels, and (as now) rate 1 / rate 2 / finish / acc.

Arms (all @0.8, the θ fixed by the existing sweep):
| arm | compute (teachers run every step) | purpose |
|---|---:|---|
| A1 Ins ⇐ A(0.5) @0.8 | 2× | single-teacher reference for T_1 (new) |
| A2 Ins ⇐ A(1.0) @0.8 | 2× | single-teacher reference for T_2 (new; A(1.1) @0.8 exists on the first 200 only = 57.5) |
| A3 ladder (0.5, 1.0, 2.0) @0.8 | 4× | the method |
| — Ins ⇐ A(2.0) @0.8 | 2× | exists: 49.0 (= T_3 alone as teacher) |

Reading rules, fixed now:
- **Success**: A3 > max(A1, A2, 49.0) by ≥ 3 points on the last 300 with McNemar p < 0.1. Then the ladder adds information the single teacher does not have (the milder teacher's token is better where it is confident).
- **Kill**: A3 ≈ the best single-teacher arm (within 2 points) — the ladder just reduces to whichever teacher it ends up using most. Then "more teachers on one line" is dead, consistent with §4 of `step2_cascade_two_lines.md`.
- Diagnostic either way: the share of tokens per level. If < 3 % of tokens come from T_1 ∪ T_2 the ladder is A(2.0) in disguise regardless of accuracy.
- My prior: kill. Ins ⇐ A(1.1) @0.8 (57.5) < Ins ⇐ A(2.0) @0.8 (59.0) on the first 200, and the gate never exceeds its strongest teacher; I expect A3 ∈ [48, 51].

## Experiment B — soft mixing instead of hard replacement at the low-confidence steps
Rule `content_softmix(λ) @θ`: at a low-confidence step, write the argmax of the **probability mixture** instead of the teacher's token.

    y = y_S                                   if c_S ≥ θ
      = argmax[(1 − λ)·p_S + λ·p_T]           otherwise,   T = A(2.0)

λ = 1 recovers the current gate; λ = 0 the student. Probability space (not logit space) so that λ is the share of teacher mass — the same
quantity as "dose" in the dose curve, now applied per token rather than per step. Mixing is on raw softmax outputs (T = 1), the same
p the gate reads.

Arms:
| arm | θ | λ | compute | purpose |
|---|---:|---|---:|---|
| B1 softmix | 0.8 | 0.5 | 2× | equal mixture at unsure steps |
| B2 softmix, adaptive | 0.8 | λ = 1 − c_S/θ | 2× | the less sure the student, the more teacher (λ → 1 as c_S → 0, λ → 0 at c_S = θ): no new hyperparameter, continuous version of the gate |
| B3 always-mix (plain ensemble) | 1.0 | 0.5 | 2× | **control: is the gate needed at all once we mix?** θ = 1 ⇒ every step mixes |
| — content gate @0.8 | 0.8 | 1 | 2× | exists: 49.0 |

Reading rules, fixed now:
- **Gate matters**: B1 or B2 > B3 by ≥ 3 points (p < 0.1). Otherwise the whole method is "a 2-model probability ensemble", which is prior art
  (logit averaging), and the confidence gate is only a compute saver — must be written that way.
- **Soft beats hard**: B1 or B2 > 49.0 by ≥ 2 points (p < 0.1). Otherwise hard replacement stays (simpler; it is also what distillation targets need).
- **Kill B2 specifically** if it is within 1 point of B1: adaptive λ buys nothing over a constant.
- My prior: B3 (always-mix) lands near the teacher or slightly above (≈ 47–50) — at the 94 % of steps where both models are confident the
  mixture's argmax is almost always the student's own token, so always-mix ≈ gate with a lower effective θ. B1/B2 ≈ 49 ± 2. I expect
  "gate matters" to fail or be marginal; this is the experiment that decides whether the method is distinguishable from an ensemble.

## Implementation (no edits to files that running jobs use)
- `scripts/step1_lockstep_gated_v2.py` = copy of `step1_lockstep_gated.py` + modes `content_ladder` (`--other` takes a comma-separated list, ascending β) and
  `content_softmix` (`--lam <float>|adaptive`). Content gate / both / random code paths untouched (byte-identical outputs on the existing modes is
  checked by re-running one existing problem in the smoke).
- `modules/run_gated_shard5.sh` = shard4 pointing at v2, with `ladder:θ` and `softmix:λ:θ` mode tokens.
- Checkpoints: A(0.5) and A(1.0) built once at `outputs/checkpoints/add-a10-b05`, `add-a10-b10` (build-at-host, flock-serialised, as before).
- Gates before submission: `py_compile`, CPU dry run of the token rule on saved logits, one 2-problem smoke per new mode on the GPU, then the 6 arms × 3 shards.
- Output dirs: `details/logs/gated_ins_add-a10-b05_1024/math500/t08-last300-s*`, `…b10…/t08-…`, `details/logs/ladder_ins_1024/math500/b05-b10-b20-t08-last300-s*`,
  `details/logs/softmix_ins_add-a10-b20_1024/math500/{lam05-t08,lamad-t08,lam05-t10}-last300-s*`.

## Cost
6 arms × 3 shards = 18 jobs; 2× arms ≈ 6 h per shard, the 4× ladder ≈ 12 h per shard at 0.51 → ≈ 40 GPU·h (H200·0.51 units), one day of wall
time if all 18 schedule at once. Teachers are called every step (no lazy call yet), so the ladder's 4× is the price of the diagnostic.

## What each outcome changes in STEP2
| A3 | B1/B2 vs B3 | consequence |
|---|---|---|
| kill | gate matters | pipeline stays "one teacher, hard gate"; teacher-axis expansion only across lines (cascade) |
| kill | gate does not matter | method = probability ensemble with a confidence-based compute saver; rewrite §1 accordingly |
| success | any | teacher ladder becomes the pipeline's expansion rule; add A(1.5) to test monotonicity |
