"""Does a random 50-problem subset of MATH-500 (first 200, all @1024 lockstep decoder) keep the full-200 trend? Zero GPU.

Systems = every arm with all 200 problems done. Checks, for one fixed subset (seed 0) and for 2000 random subsets:
Spearman of accuracies (subset vs full-200), fraction of system pairs whose order agrees (pairs tied on full skipped),
and whether the top system stays on top. Writes details/subset50/{subset50_ids.json, check.md}.
"""
import json, random
from pathlib import Path
L = Path("/mnt/lts4/scratch/home/yqin/DllmAR/new_ideas/idea1_selective_distill/details/logs")
OUT = L.parent / "technical_details/subset_checks"
def tx(p):
    f = L / p / "texts.jsonl"
    return {r["id"]: int(r["ok"]) for r in map(json.loads, open(f))} if f.exists() else {}
SYS = {"Ins": "lockstep1024/math500/parent-free", "merge": "lockstep1024/math500/merge-free",
       "add1.1": "lockstep1024/math500/add11-free", "add2.0": "lockstep1024/math500/add20-free"}
for T, d in [("add1.1", "gated_ins_add-a10-b11_1024"), ("add2.0", "gated_ins_add-a10-b20_1024"), ("merge", "gated_ins_mrg0406_1024")]:
    for t in ["t02", "t04", "t06", "t08"]:
        SYS[f"Ins→{T} θ0.{t[2]}"] = f"{d}/math500/{t}"
R = {k: tx(p) for k, p in SYS.items()}
R = {k: v for k, v in R.items() if all(i in v for i in range(200))}
names = list(R)
full = {k: sum(R[k][i] for i in range(200)) / 2 for k in names}  # percent
def rank(xs):
    o = sorted(range(len(xs)), key=lambda i: xs[i]); r = [0.0] * len(xs); i = 0
    while i < len(o):
        j = i
        while j + 1 < len(o) and xs[o[j + 1]] == xs[o[i]]: j += 1
        for k in range(i, j + 1): r[o[k]] = (i + j) / 2
        i = j + 1
    return r
def spearman(a, b):
    ra, rb = rank(a), rank(b); n = len(a); ma, mb = sum(ra) / n, sum(rb) / n
    c = sum((x - ma) * (y - mb) for x, y in zip(ra, rb)); return c / (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** .5
def check(ids):
    sub = {k: 100 * sum(R[k][i] for i in ids) / len(ids) for k in names}
    pairs = [(a, b) for a in names for b in names if a < b and full[a] != full[b]]
    agree = sum((full[a] - full[b]) * (sub[a] - sub[b]) > 0 for a, b in pairs) / len(pairs)
    top = max(names, key=full.get)
    return sub, spearman([full[k] for k in names], [sub[k] for k in names]), agree, max(names, key=sub.get) == top
ids0 = sorted(random.Random(0).sample(range(200), 50))
sub0, rho0, ag0, top0 = check(ids0)
rng = random.Random(1); sims = [check(rng.sample(range(200), 50))[1:] for _ in range(2000)]
q = lambda xs, p: sorted(xs)[int(p * (len(xs) - 1))]
rhos, ags = [s[0] for s in sims], [s[1] for s in sims]
OUT.mkdir(exist_ok=True)
(OUT / "subset50_ids.json").write_text(json.dumps(dict(seed=0, pool="MATH-500 ids 0..199", ids=ids0)))
L2 = [f"# Random 50-problem subset vs full 200 (MATH-500 first 200, @1024)\n", f"subset (seed 0): {ids0}\n",
      "| system | full 200 (%) | subset 50 (%) |", "|---|---:|---:|"]
L2 += [f"| {k} | {full[k]:.1f} | {sub0[k]:.0f} |" for k in sorted(names, key=lambda k: -full[k])]
L2 += ["", f"seed-0 subset: Spearman {rho0:.2f}, pairwise order agreement {100*ag0:.0f}%, top system kept: {top0}",
       f"2000 random subsets: Spearman median {q(rhos,.5):.2f} (5–95%: {q(rhos,.05):.2f}–{q(rhos,.95):.2f}); "
       f"pairwise agreement median {100*q(ags,.5):.0f}% (5–95%: {100*q(ags,.05):.0f}–{100*q(ags,.95):.0f}%); "
       f"top kept {100*sum(s[2] for s in sims)/len(sims):.0f}%"]
(OUT / "check.md").write_text("\n".join(L2) + "\n"); print("\n".join(L2))
# pairwise agreement by size of the full-200 gap
for lo, hi in [(0, 3), (3, 6), (6, 10), (10, 100)]:
    pr = [(a, b) for a in names for b in names if a < b and lo <= abs(full[a] - full[b]) < hi]
    if not pr: continue
    rng = random.Random(2); tot = 0
    for _ in range(500):
        ids = rng.sample(range(200), 50); s = {k: sum(R[k][i] for i in ids) for k in names}
        tot += sum((full[a] - full[b]) * (s[a] - s[b]) > 0 for a, b in pr) / len(pr)
    print(f"gap {lo}-{hi} pp on full: {len(pr)} pairs, order kept on a random 50 subset {100*tot/500:.0f}% (ties count as not kept)")
