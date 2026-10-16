"""Score an existing full-541 IFEval samples file on the ifeval_rand100 subset (zero GPU): prompt-level strict / loose acc on the same
100 doc_ids the ifeval_rand100 task uses. Usage: python score_ifeval_subset.py <samples_ifeval_*.jsonl> [--out done.json --label L --model M]"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent)); from rand100_utils import IDS

ap = argparse.ArgumentParser(); ap.add_argument("samples"); ap.add_argument("--out"); ap.add_argument("--label", default=""); ap.add_argument("--model", default="")
a = ap.parse_args()
rows = {}
for l in open(a.samples):
    r = json.loads(l); rows[r["doc_id"]] = r
sub = [rows[i] for i in IDS]; assert len(sub) == len(IDS), (len(sub), len(IDS))
strict = 100 * sum(r["prompt_level_strict_acc"] for r in sub) / len(sub); loose = 100 * sum(r["prompt_level_loose_acc"] for r in sub) / len(sub)
res = {"task": "ifeval_rand100_offline", "label": a.label, "model": a.model, "score": round(strict, 1), "loose": round(loose, 1), "n": len(sub),
       "source": a.samples, "per_doc": {str(i): int(rows[i]["prompt_level_strict_acc"]) for i in IDS}}
print(json.dumps({k: v for k, v in res.items() if k != "per_doc"}))
if a.out:
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(json.dumps(res) + "\n")
