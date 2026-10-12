import json, statistics as st
from math import comb
L = "/mnt/lts4/scratch/home/yqin/DllmAR/new_ideas/idea1_selective_distill/details/logs/"
def tx(p): return {r["id"]: r["ok"] for r in map(json.loads, open(L + p + "/texts.jsonl"))}
def mc(a, b, n=200):
    w = sum(a[i] and not b[i] for i in range(n)); l = sum(b[i] and not a[i] for i in range(n)); m = w + l
    return w, l, (min(1, 2 * sum(comb(m, k) for k in range(min(w, l) + 1)) / 2**m) if m else 1)
REF = {"Ins": tx("lockstep1024/math500/parent-free"), "merge": tx("lockstep1024/math500/merge-free"), "add1.1": tx("lockstep1024/math500/add11-free")}
for k, v in REF.items(): print(f"{k:8s} alone {sum(v[i] for i in range(200))}/200")
ARMS = {f"Ins→{n} θ={t[1]}.{t[2]}": f"{d}/math500/{t}" for n, d in [("add1.1", "gated_ins_add-a10-b11_1024"), ("add2.0", "gated_ins_add-a10-b20_1024"), ("merge", "gated_ins_mrg0406_1024")] for t in ["t02", "t04"]}
print("\n| arm | acc | vs Ins (win/loss, p) | vs merge | vs add1.1 | steps using donor / problem (mean, % problems ≥1) | tokens substituted / problem | sub_rate | disagree |")
for k, p in ARMS.items():
    a = tx(p); acc = sum(a[i] for i in range(200))
    g = {}
    for r in map(json.loads, open(L + p + "/gate_per_item.jsonl")): g[r["id"]] = r
    gs = [g[i] for i in range(200) if i in g]
    ss = [r["n_steps_with_sub"] for r in gs]; ns = [r["n_steps"] for r in gs]
    S = json.load(open(L + p + "/gate_stats.json"))
    cells = []
    for r in ["Ins", "merge", "add1.1"]:
        w, l, pv = mc(a, REF[r]); cells.append(f"{w}/{l}, p={pv:.2f}")
    print(f"| {k} | {acc}/200 | " + " | ".join(cells) + f" | {st.mean(ss):.1f} of {st.mean(ns):.0f} ({100*sum(x>0 for x in ss)/len(ss):.0f}%) | {st.mean(r['n_substituted'] for r in gs):.1f} | {sum(r['n_substituted'] for r in gs)/sum(r['n_committed'] for r in gs):.4f} | {sum(r['n_differ'] for r in gs)/sum(r['n_committed'] for r in gs):.4f} | n_logged={len(gs)}")
