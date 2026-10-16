"""EvalPlus (HumanEval+ / MBPP+) on an explicit task-id subset, same generation + scoring as the official `python -m evalplus.codegen` /
`evalplus.evaluate` (dllm backend, 512 tokens / 512 steps, T 0.1): the dataset getter is patched to the listed ids, everything else is
untouched. Usage (from the evalplus repo dir, PYTHONPATH=.):
  python evalplus_subset.py --dataset mbpp --ids val_mbppplus_rand100_seed0.json --model <ckpt> --root evalplus_val_h200 --out mbpp.json --label L
Writes <out> = {task, label, model, score (plus pass@1 %), base, n, per_task}. 2026-10-16."""
import argparse, glob, json, os, sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", choices=["humaneval", "mbpp"], required=True); ap.add_argument("--ids", required=True); ap.add_argument("--model", required=True)
ap.add_argument("--root", required=True); ap.add_argument("--out", required=True); ap.add_argument("--label", default="")
ap.add_argument("--max-new-tokens", type=int, default=512); ap.add_argument("--steps", type=int, default=512); ap.add_argument("--temperature", type=float, default=0.1)
a = ap.parse_args()
ids = set(json.load(open(a.ids)))

import evalplus.data as D, evalplus.codegen as C, evalplus.evaluate as E
_get = {"humaneval": D.get_human_eval_plus, "mbpp": D.get_mbpp_plus}[a.dataset]
def _sub(*k, **kw):
    full = _get(*k, **kw); out = {t: p for t, p in full.items() if t in ids}
    assert len(out) == len(ids), (len(out), len(ids)); return out
for mod in (C, E):
    setattr(mod, {"humaneval": "get_human_eval_plus", "mbpp": "get_mbpp_plus"}[a.dataset], _sub)

entry = C.run_codegen   # the function behind `python -m evalplus.codegen` (Fire)
entry(model=a.model, dataset=a.dataset, root=a.root, backend="dllm", trust_remote_code=True, temperature=a.temperature,
      max_new_tokens=a.max_new_tokens, diffusion_steps=a.steps)
base = os.path.basename(a.model.rstrip("/"))
cands = [f for f in glob.glob(f"{a.root}/{a.dataset}/*{base}_dllm*.jsonl") if not f.endswith(".raw.jsonl")]
samples = max(cands, key=os.path.getmtime)
res_file = str(Path(a.out).with_suffix(".eval_results.json"))
E.evaluate(dataset=a.dataset, samples=samples, output_file=res_file)
e = json.load(open(res_file))["eval"]; assert set(e) == ids, (len(e), len(ids))
plus = {t: int(any(x["plus_status"] == "pass" for x in v)) for t, v in e.items()}; basep = {t: int(any(x["base_status"] == "pass" for x in v)) for t, v in e.items()}
res = {"task": f"{a.dataset}plus_subset", "label": a.label, "model": a.model, "score": round(100 * sum(plus.values()) / len(ids), 1),
       "base": round(100 * sum(basep.values()) / len(ids), 1), "n": len(ids), "samples": samples, "per_task": plus}
Path(a.out).write_text(json.dumps(res) + "\n"); print(json.dumps({k: v for k, v in res.items() if k != "per_task"}))
