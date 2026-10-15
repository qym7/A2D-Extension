# code — one folder per step; every run output goes to `../outputs/stepN/`

| folder | step | contents |
|---|---|---|
| `step0/` | baselines, merge/addition lines, motivation figures | line sweeps, help/harm plots, `fig_motivation*.py`, subset-size checks, DreamReasoner, `run_lockstep_m500.sh` |
| `step1/` | the content gate and its ablations | `lockstep_gated.py` / `lockstep_gated_v2.py` (the gate decoder, all modes; executed by `../../modules/run_gated_shard5.sh`), `merge_shards.py`, `main_table.py`, `arness_from_records.py`, EOS / dose analyses |
| `step2/` | multi-model / logit-space combinations | `theta_vote.py`, `weight_vote.py` (answer-level votes); the soft-mix / max-conf modes live in `step1/lockstep_gated_v2.py`; cascades are post-hoc in `technical_details/step2_*.md` |
| `step3/` | distillation training | `build_prompts.py` → `rollout.py` → `train_frontier_kd.py` (draft) → `merge_math_shards.py`, `results_table.py`; `signal_check.py` (prelim); `pipeline/` (Snakemake end-to-end) |

Outputs (untracked), same split: `../outputs/checkpoints/` (shared across steps), `../outputs/step0/{lockstep,lockstep1024,dr_aime,official_eval}`,
`../outputs/step1/{gated*,arness50,arness_lambda_b300}`, `../outputs/step2/{softmix*,maxconf*}`, `../outputs/step3/{prompts.jsonl,rollouts,step3_signal_check*,step3-<arm>}`.
Scripts locate the idea root with `Path(__file__).resolve().parents[2]`. Job runners / submitters shared across ideas stay in `../../modules/`
(`run_gated_shard5.sh`, `run_step3_*.sh`, `run_official_eval.sh`, `submit.sh`, `submit_wait.sh`).
