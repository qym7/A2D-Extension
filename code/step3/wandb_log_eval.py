"""wandb_log_eval.py --train-out <ckpt dir> --summary <summary.json> --prefix math500 [--extra k=v ...]
Append an evaluation result to the W&B run of a trained arm (run id in <train_out>/wandb.json, written by train_frontier_kd.py).
Called by the pipeline after the MATH shards are merged and after each guard; also usable by hand. Zero GPU."""
import argparse, json, os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-out", required=True); ap.add_argument("--summary", required=True); ap.add_argument("--prefix", required=True)
    ap.add_argument("--extra", nargs="*", default=[])
    a = ap.parse_args()
    meta_f = Path(a.train_out) / "wandb.json"
    if not meta_f.exists():
        print("[wandb] no wandb.json in", a.train_out, "- skipping"); return
    meta = json.load(open(meta_f)); summ = json.load(open(a.summary))
    import wandb
    os.environ.setdefault("WANDB_DIR", str(Path(a.train_out).parent / "wandb"))
    run = wandb.init(project=meta["project"], id=meta["id"], resume="allow", dir=os.environ["WANDB_DIR"])
    vals = {f"eval/{a.prefix}_{k}": v for k, v in summ.items() if isinstance(v, (int, float))}
    for kv in a.extra:
        k, v = kv.split("=", 1); vals[f"eval/{a.prefix}_{k}"] = float(v)
    for k, v in vals.items():
        run.summary[k] = v
    run.log(vals)
    run.finish(); print("[wandb] logged", vals)


if __name__ == "__main__":
    main()
