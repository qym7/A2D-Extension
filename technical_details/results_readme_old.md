# Idea 1 — results

> **目录（2026-10-12 起按 3 步组织）**：`STEP0_setup_baselines.md`（待写）→ `step1_methods.md`（方法 + 四对全表，已写）→ **`step2_verdict.md`（完整结果表 + 判定，已写）**。
> 生成表 `technical_details/main_table.md`，机制分解 `technical_details/dose_curve_mechanism.md`，写法 `technical_details/naming.md`。下面是旧的按实验代号的记录。

## S0 baseline gate — **PASSED, from existing artifacts** (2026-09-28)

The four A2D Table 2 cells already exist at n=500 on disk and match the published values exactly:

| arm | path under `dream_study/step2_instruct/outputs/lockstep/math500/` | n | accuracy | published |
|---|---|---:|---:|---:|
| parent free | `dreamv0-parent-free/` | 500 | **38.4** | 38.4 ✓ |
| content port | `dreamv0-content/` | 500 | **42.8** | 42.8 ✓ |
| schedule port | `dreamv0-schedule/` | 500 | **42.2** | 42.2 ✓ |
| merge free | `dreamv0-merge-free/` | 500 | **45.0** | 45.0 ✓ |

Determinism check already done by that project: each off-diagonal cell was run from both driver sides
and gave byte-identical outcomes (214/214 and 211/211 correct, 0 discordant).

### My own S0 runs failed the gate, and the cause was my protocol error

`yqin-i1-s0-m500-{free,a2d}-b300` at **n=200** gave parent 45.5 and merge 50.5 — about +6 points above
the published 38.4 / 45.0. Not noise: a systematic offset from the wrong sample size. I had taken
"first 200 samples" from A2D §2 footnote 1, which governs the §2 analysis; **the lockstep protocol is
n=500** (`iclr/overleaf/plots/lockstep/README.md`: "MATH-500, n=500 … The free arms reproduce the
official lm_eval numbers exactly"). The gate did its job — it caught my mistake rather than letting a
wrong baseline through. The two redundant n=200 port runs were stopped.

## A bug the same document caught, before it wasted a sweep

`details/draft_scripts/lockstep_gated.py` originally gated on the driver's **protocol-sharpened**
confidence `pd`. That same README records that the sharpened confidence "saturates at 1.0 everywhere"
and "cannot rank tokens by confidence" — which is exactly why the instrument's author added a raw
value on 2026-09-21. A gate on the sharpened number would have been degenerate: identical to free mode
for every θ<1 and identical to the content port at θ≥1, i.e. an entire 6-point sweep of null results
that looked like a finding. The gate now uses `praw`, and the run also records a 20-bin histogram of
the raw confidence over committed positions so the θ grid can be checked against the actual
distribution.

## E1 — running (2026-09-28)

`yqin-i1-e1-a09-t{00,03,05,07,09,10}-b300`: host = the A2D merge (45.0, the best known scalar point,
so Rule 3's bar), donor = `D_b + 0.9·τ_A`, θ ∈ {0, 0.3, 0.5, 0.7, 0.9, 1.0}, n=500.
θ=0 must reproduce 45.0 exactly (in-sweep plumbing check); θ=1.0 is the full content port from this
donor. Past-peak donors α=1.2 and 1.5 are building (`yqin-i1-build-a{12,15}-b300`) for the second
pass.

Useful prior from the existing records: the consulted model's argmax differs from the driver's at only
**3.4%** of committed positions, and where they disagree the consulted model is **less** sure
(sharpened p 0.89 vs 0.996 where they agree). So the intervention budget SPEC.md asks for (≤10%
substitution rate) is compatible with the disagreement rate — the gate cannot substitute more than
3.4% of positions in the first place, which makes the "selective" requirement easy to satisfy and
shifts the risk entirely onto whether those 3.4% are worth anything.
