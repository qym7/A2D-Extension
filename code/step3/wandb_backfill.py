"""wandb_backfill.py --train-out <ckpt dir> [--project a2d-step3] — create the W&B run for an arm trained BEFORE W&B logging existed
(t1k runs of 2026-10-15): replays <train_out>/train_log.jsonl step by step, uploads args.json as config, writes <train_out>/wandb.json so
code/step3/wandb_log_eval.py can append evaluations. Zero GPU; run on the login node under a memory cap."""
import argparse, json, os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-out", required=True); ap.add_argument("--project", default="a2d-step3")
    a = ap.parse_args()
    out = Path(a.train_out)
    if (out / "wandb.json").exists():
        print("[wandb] run exists:", json.load(open(out / "wandb.json"))); return
    import wandb
    os.environ.setdefault("WANDB_DIR", str(out.parent / "wandb"))
    cfg = json.load(open(out / "args.json")) if (out / "args.json").exists() else {}
    run = wandb.init(project=a.project, name=out.name, config=cfg, dir=os.environ["WANDB_DIR"])
    n = 0
    for l in open(out / "train_log.jsonl"):
        rec = json.loads(l); run.log({f"train/{k}": v for k, v in rec.items() if k != "step"}, step=rec["step"]); n += 1
    json.dump(dict(project=a.project, id=run.id, name=run.name, mode="online"), open(out / "wandb.json", "w"))
    run.summary["train/steps"] = n; run.finish(); print(f"[wandb] backfilled {n} steps -> {run.name} ({run.id})")


if __name__ == "__main__":
    main()
