#!/usr/bin/env python3
"""IFEval (random-100 subset) decoded by the decode-time gate host <= donor @theta (2026-10-16, user: the gate row needs IF / code numbers).

Same decode loop as the gate's MATH row (code/step1/lockstep_gated_v2.run_gated, mode content_gated: the host proposes, the donor's token
is taken where the host's raw p < theta), same protocol knobs as the official IFEval run (1280 new tokens / 1280 steps, confidence at T 0.1,
top-p 0.9, entropy order; token = argmax as in every lockstep row). Prompts = the exact chat-templated strings the official lm_eval run used
(read from an existing ifeval_rand100 samples log, doc_id order = rand100_utils.IDS order). Scored with the official lm_eval ifeval scorer.
  python gate_ifeval.py --samples <lm_eval samples_ifeval_rand100_*.jsonl> --host H --donor D --theta 0.8 --out-dir OUT --done OUT.json --label gate
Resumes from OUT/texts.jsonl. Done-file: {task, label, score (prompt-strict, %), loose, n, per_doc{IDS[i]: 0/1}} = the format of the other rows."""
import argparse, json, random, sys, time
from pathlib import Path
import torch

I1 = Path("/mnt/lts4/scratch/home/yqin/DllmAR/new_ideas/idea1_selective_distill")
LMEVAL = Path("/mnt/lts4/scratch/home/yqin/DllmAR/dream_study/dream_official_eval/eval_instruct")
sys.path.insert(0, str(I1 / "code/step1")); sys.path.insert(0, str(LMEVAL))
import lockstep_gated_v2 as GV  # noqa: E402  (also puts lockstep_port on the path)
from lm_eval.tasks.ifeval.utils import process_results  # noqa: E402

IDS = sorted(random.Random(0).sample(range(541), 100))   # = rand100_utils.IDS (not imported: that module pulls lm_eval's yaml machinery)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True); ap.add_argument("--host", required=True); ap.add_argument("--donor", required=True)
    ap.add_argument("--theta", type=float, default=0.8); ap.add_argument("--steps", type=int, default=1280); ap.add_argument("--max-new", type=int, default=1280)
    ap.add_argument("--top-p", type=float, default=0.9); ap.add_argument("--conf-temp", type=float, default=0.1); ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out-dir", type=Path, required=True); ap.add_argument("--done", required=True); ap.add_argument("--label", default="gate")
    a = ap.parse_args()
    GV.THETA = a.theta; GV.GATED_MODE = "content_gated"; GV.SEED = a.seed; GV.OUTDIR = None   # per-item gate counts keyed by math ids: off here
    docs = []
    for l in open(a.samples):
        r = json.loads(l); docs.append({"doc_id": r["doc_id"], "doc": r["doc"], "text": r["arguments"]["gen_args_0"]["arg_0"]})
    docs = sorted(docs, key=lambda d: d["doc_id"])[: a.n]
    assert len(docs) == a.n and [d["doc_id"] for d in docs] == list(range(a.n)), "samples log must hold doc_id 0..n-1"
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.host, trust_remote_code=True)
    host = AutoModel.from_pretrained(a.host, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda().eval()
    donor = AutoModel.from_pretrained(a.donor, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda().eval()
    mask_id = tok.convert_tokens_to_ids("<|mask|>")
    if mask_id is None or mask_id < 0:
        mask_id = host.config.mask_token_id
    a.out_dir.mkdir(parents=True, exist_ok=True); texts_f = a.out_dir / "texts.jsonl"
    done = {json.loads(l)["doc_id"] for l in open(texts_f)} if texts_f.exists() else set()
    print(f"[gate-ifeval] host={a.host} donor={a.donor} theta={a.theta} n={len(docs)} steps={a.steps} resume={len(done)}", flush=True)
    t0 = time.time()
    for k, d in enumerate(docs):
        if d["doc_id"] in done:
            continue
        ids = tok(d["text"], return_tensors="pt").input_ids.cuda(); P = ids.shape[1]
        x = torch.cat([ids, torch.full((1, a.max_new), mask_id, dtype=ids.dtype, device=ids.device)], dim=1)
        rng = torch.Generator().manual_seed(a.seed * 1000003 + k)
        with torch.no_grad():
            x, _ = GV.run_gated(host, donor, x, mask_id, mode="content", steps=a.steps, eps=a.eps, top_p=a.top_p, conf_temp=a.conf_temp, record=False, rng=rng)
        gen = x[0, P:]
        text = tok.decode(gen.tolist()).split(tok.eos_token)[0]          # = lm_eval diffllm decode
        m = process_results(d["doc"], [text])
        rec = {"doc_id": d["doc_id"], "key": d["doc"]["key"], "text": text, "strict": int(m["prompt_level_strict_acc"]), "loose": int(m["prompt_level_loose_acc"])}
        with open(texts_f, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  [{k + 1}/{len(docs)}] strict={rec['strict']} loose={rec['loose']} len={int(gen.numel())} ({time.time() - t0:.0f}s)", flush=True)
    rows = {json.loads(l)["doc_id"]: json.loads(l) for l in open(texts_f)}
    assert len(rows) == len(docs), (len(rows), len(docs))
    strict = 100 * sum(r["strict"] for r in rows.values()) / len(rows); loose = 100 * sum(r["loose"] for r in rows.values()) / len(rows)
    out = {"task": "ifeval_rand100_gate", "label": a.label, "model": f"gate {a.host} <= {a.donor} @{a.theta}", "score": round(strict, 1), "loose": round(loose, 1),
           "n": len(rows), "per_doc": {str(IDS[i]): rows[i]["strict"] for i in range(len(docs))}, "texts": str(texts_f)}
    Path(a.done).parent.mkdir(parents=True, exist_ok=True); Path(a.done).write_text(json.dumps(out))
    print(f"[gate-ifeval] {a.label} -> strict {strict:.1f} loose {loose:.1f} (n={len(rows)})", flush=True)


if __name__ == "__main__":
    main()
