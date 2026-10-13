"""AR-ness of the lockstep decode from --record runs (details/logs/arness50/math500/<arm>/records/<id>.json). Zero GPU.
Order = the sequence of committed canvas positions (one per step; the last step's leftovers flattened in increasing position order).
Metrics: DiffuCoder Global/Local AR-ness@k (arness_metrics.py, full 1024-token canvas incl. post-EOS padding, as App. A.2) and the
same on the response only (positions before the first EOS), plus Spearman(order, position) and adjacency for continuity with A2D.
Also checks that the recorded texts equal the earlier non-record runs of the same arm (argmax decode is deterministic)."""
import json, sys, statistics as st
from collections import Counter
from pathlib import Path
import numpy as np
I1 = Path(__file__).resolve().parents[1]; L = I1 / "details/logs"
sys.path.insert(0, "/mnt/lts4/scratch/home/yqin/DllmAR/dream_study/step2_instruct/scripts")
from arness_metrics import global_arness_at_k, local_arness_at_k
ARMS = {  # arm: (record dir, earlier run dir for the determinism check)
 "Ins alone": ("ins-free", "lockstep1024/math500/parent-free"),
 "A(2.0) alone": ("a20-free", "lockstep1024/math500/add20-free"),
 "Ins ⇐ A(2.0) content @0.8": ("ins-to-a20-t08", "gated_ins_add-a10-b20_1024/math500/t08"),
 "Ins ⇐ A(2.0) position+content @0.8": ("ins-to-a20-both08", "gated_ins_add-a10-b20_1024/math500/both08-merged"),
 "M(.4,.6) alone": ("m46-free", "lockstep1024/math500/merge-free"),
 "M(0,1) alone": ("m01-free", "lockstep1024/math500/l10-free"),
 "M(.4,.6) ⇐ M(0,1) content @0.8": ("m46-to-m01-t08", "gated1024/math500/e1b-mrg10-t08-merged"),
 "M(.4,.6) ⇐ M(0,1) position+content @0.8": ("m46-to-m01-both08", "gated1024/math500/e1b-mrg10-both08-merged"),
}
def spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y)); return float(np.corrcoef(rx, ry)[0, 1])
rows = []
for name, (rd, prev) in ARMS.items():
    d = L / "arness50/math500" / rd
    m = {k: [] for k in ("g1", "l1", "l8", "g1r", "l1r", "sp", "adj", "len")}
    for f in sorted((d / "records").glob("*.json")):
        r = json.load(open(f)); P = r["prompt_len"]
        order = [p - P for s in r["steps"] for p in s["pos"]]
        assert sorted(order) == list(range(len(order))), f
        eos = Counter(r["final_ids"][-8:]).most_common(1)[0][0]
        Lr = r["final_ids"].index(eos) if eos in r["final_ids"] else len(order)
        m["len"].append(Lr)
        m["g1"].append(global_arness_at_k(order, 1)); m["l1"].append(local_arness_at_k(order, 1)); m["l8"].append(local_arness_at_k(order, 8))
        resp = [p for p in order if p < Lr]; rank = {p: i for i, p in enumerate(sorted(resp))}; resp = [rank[p] for p in resp]
        if len(resp) > 8:
            m["g1r"].append(global_arness_at_k(resp, 1)); m["l1r"].append(local_arness_at_k(resp, 1))
        m["sp"].append(spearman(order, list(range(len(order))))); m["adj"].append(sum(abs(b - a) == 1 for a, b in zip(order, order[1:])) / (len(order) - 1))
    # determinism check against the earlier run
    new = {x["id"]: x["text"] for x in map(json.loads, open(d / "texts.jsonl"))}
    old = {x["id"]: x["text"] for x in map(json.loads, open(L / prev / "texts.jsonl"))} if (L / prev / "texts.jsonl").exists() else {}
    same = sum(new[i] == old.get(i) for i in new); acc = 100 * sum(x["ok"] for x in map(json.loads, open(d / "texts.jsonl"))) / len(new)
    rows.append((name, len(new), acc, st.mean(m["g1"]), st.mean(m["l1"]), st.mean(m["l8"]), st.mean(m["g1r"]), st.mean(m["l1r"]), st.mean(m["sp"]), st.mean(m["adj"]), st.mean(m["len"]), same))
out = ["| arm | n | acc | Global@1 (full) | Local@1 (full) | Local@8 (full) | Global@1 (response) | Local@1 (response) | Spearman | adjacency | response len | texts = earlier run |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
for r in rows:
    out.append(f"| {r[0]} | {r[1]} | {r[2]:.0f} | {r[3]:.3f} | {r[4]:.3f} | {r[5]:.3f} | {r[6]:.3f} | {r[7]:.3f} | {r[8]:.3f} | {r[9]:.3f} | {r[10]:.0f} | {r[11]}/{r[1]} |")
print("\n".join(out))
