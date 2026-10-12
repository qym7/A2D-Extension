# A2D extension — idea 1: "over distillation" (leveraging an over-merged model at decode time)

Post-A2D exploration on Dream-v0-7B ↔ Qwen2.5-7B. Three documents, in reading order:

| file | what it holds |
|---|---|
| `step0_setup_baselines.md` | notation, protocol (lockstep @1024, last-300 default), single-model baselines incl. the AR reference, two facts that condition every reading (finish-rate effect; AR under-scored) |
| `step1_methods.md` | the method, the full student ⇐ teacher table (all pairs × θ, three subsets, replacement rates), and the mechanism ablations |
| `step2_verdict.md` | consolidated results, what died and why, what survives, main takeaway, next steps (generalisation + pipeline + distillation) |

Everything else is in `technical_details/`, prefixed by the step it supports (`step0_*` baselines / headroom / subset checks, `step1_*` generated tables, dose-curve decomposition, votes, the older 512-era notes, `step2_*` idea notes) — kept for traceability, not required to follow the story.

`scripts/` is organised the same way:

| prefix | what |
|---|---|
| `step0_*` | baseline runner (`step0_run_lockstep_m500.sh`, `step0_submit_s0.sh`), headroom / AR-help figures, DreamReasoner analyses, subset-reliability checks |
| `step1_*` | the gated decoder `step1_lockstep_gated.py` (modes: content_gated, schedule_gated, both_gated, content_random, content_random_uncond), `step1_merge_shards.py` (sharded runs → `*-merged`), `step1_main_table.py` (→ `technical_details/step1_main_table.md`), votes, dose/finish decompositions |

Job submitters shared across ideas live one level up in `../modules/` (`run_gated_shard*.sh`, `submit.sh`); they call `details/draft_scripts/lockstep_gated.py`, which is a symlink to `scripts/step1_lockstep_gated.py`.
`spec.md` / `task.md` / `analysis.md` are the working documents.
Raw outputs (`outputs/`, `details/logs/`, `results/`), checkpoints and `trash/` are not tracked.

Current state (2026-10-13): the content gate M(.4,.6) ⇐ A(2.0) @0.8 reaches 54.2 on MATH-500 (best single point 51.4) at 2× decode; the mechanism is ablated (`step1_methods.md` §2): confidence selectivity is real (random at the same dose = student), the teacher's decode order adds nothing; the AR reference under lenient scoring (71) is out of reach of this model family on MATH.
