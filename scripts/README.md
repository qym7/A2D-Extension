# scripts — one folder per step

| folder | step | contents |
|---|---|---|
| `step0/` | baselines, merge/addition lines, motivation figures | `fig_motivation*.py`, help/harm plots, subset checks, DreamReasoner, `run_lockstep_m500.sh` |
| `step1/` | the content gate and its ablations | `lockstep_gated.py` / `lockstep_gated_v2.py` (decoders), `merge_shards.py`, `main_table.py`, `arness_from_records.py`, votes, EOS analyses |
| `step2/` | cascade / multi-teacher analysis | (post-hoc analyses live in `technical_details/step2_*.md`; no scripts yet) |
| `step3/` | training-time over-distillation | `signal_check.py` (prelim), `build_prompts.py`, `rollout.py`, `train_frontier_kd.py` (draft), `merge_math_shards.py`, `results_table.py`; orchestration in `../pipeline/` (Snakemake) |

Job runners / submitters shared across ideas stay one level up in `../modules/` (`run_gated_shard*.sh`, `run_step3_signal.sh`, `submit.sh`);
the copies they execute during a job are in `details/draft_scripts/` (never edited while jobs run).
Every script writes under `details/logs/`, `technical_details/` or `outputs/` relative to the idea root (`Path(__file__).resolve().parents[2]`).
