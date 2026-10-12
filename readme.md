# A2D extension — idea 1: "over distillation" (leveraging an over-merged model at decode time)

Post-A2D exploration on Dream-v0-7B ↔ Qwen2.5-7B. Three documents, in reading order:

| file | what it holds |
|---|---|
| `step0_setup_baselines.md` | notation, protocol (lockstep @1024, last-300 default), single-model baselines incl. the AR reference, two facts that condition every reading (finish-rate effect; AR under-scored) |
| `step1_methods.md` | the method, the full student ⇐ teacher table (all pairs × θ, three subsets, replacement rates), and the mechanism ablations |
| `step2_verdict.md` | consolidated results, what died and why, what survives, main takeaway, next steps (generalisation + pipeline + distillation) |

Everything else is in `technical_details/` (generated tables, mechanism decomposition, votes, subset checks, the older per-experiment notes) — kept for traceability, not required to follow the story.
`spec.md` / `task.md` / `analysis.md` are the working documents; `details/draft_scripts/` holds the analysis and decoding scripts; `commands/` the job submitters.
Raw outputs (`outputs/`, `details/logs/`, `results/`), checkpoints and `trash/` are not tracked.

Current state (2026-10-12): the content gate M(.4,.6) ⇐ A(2.0) @0.8 reaches 54.2 on MATH-500 (best single point 51.4) at 2× decode; confidence-selectivity and decode-order explanations are under ablation (`step1_methods.md` §2); the AR reference under lenient scoring (71) is out of reach of this model family on MATH.
