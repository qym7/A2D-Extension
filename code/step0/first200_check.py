"""Is MATH-500 'first 200' (ids 0..199) a faithful stand-in for all 500? Zero GPU.
Uses every official A2D Dream-v0 MATH-500 run with all 500 docs (@512, official lm_eval) and every @1024 lockstep run with 500.
Reports per-system acc on first 200 / last 300 / all 500, Spearman(first200, all500), and pairwise order agreement.
"""
import glob, json
from pathlib import Path
OE = "/mnt/lts4/scratch/home/yqin/DllmAR/dream_study/step2_instruct/outputs/official_eval/"
L = Path("/mnt/lts4/scratch/home/yqin/DllmAR/new_ideas/idea1_selective_distill/outputs")
def oe(run):
    f = sorted(glob.glob(OE + run + "/*/samples_*.jsonl"))[-1]
    return {d["doc_id"]: int(d["exact_match"]) for d in map(json.loads, open(f))}
S = {}
for d in sorted(glob.glob(OE + "mathins_*")):
    try: r = oe(Path(d).name)
    except Exception: continue
    if len(r) == 500 and set(r) == set(range(500)): S["@512 " + Path(d).name.replace("mathins_", "")] = r
for k, p in {"@1024 Ins": "step0/lockstep1024/math500/parent-free", "@1024 M(.4,.6)": "step0/lockstep1024/math500/merge-free",
             "@1024 M(0,1)": "step0/lockstep1024/math500/l10-free", "@1024 M⇐M(0,1)@0.3": "step1/gated1024/math500/e1b-mrg10-t03",
             "@1024 M⇐M(0,1)@0.7": "step1/gated1024/math500/e1b-mrg10-t07"}.items():
    S[k] = {r["id"]: int(r["ok"]) for r in map(json.loads, open(L / p / "texts.jsonl"))}
def rk(x):
    o = sorted(range(len(x)), key=lambda i: x[i]); r = [0.0] * len(x); i = 0
    while i < len(o):
        j = i
        while j + 1 < len(o) and x[o[j + 1]] == x[o[i]]: j += 1
        for q in range(i, j + 1): r[o[q]] = (i + j) / 2
        i = j + 1
    return r
def spear(a, b):
    ra, rb = rk(a), rk(b); n = len(a); ma, mb = sum(ra) / n, sum(rb) / n
    return sum((x - ma) * (y - mb) for x, y in zip(ra, rb)) / (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** .5
rows = []
for k, r in S.items():
    a, b, c = (100 * sum(r[i] for i in rg) / len(rg) for rg in (range(200), range(200, 500), range(500)))
    rows.append((k, a, b, c))
print(f"{len(rows)} systems\n| system | first 200 | last 300 | all 500 | first200 − all500 |\n|---|---:|---:|---:|---:|")
for k, a, b, c in sorted(rows, key=lambda x: -x[3]): print(f"| {k} | {a:.1f} | {b:.1f} | {c:.1f} | {a-c:+.1f} |")
for tag in ["@512", "@1024"]:
    R = [x for x in rows if x[0].startswith(tag)]
    if len(R) < 3: continue
    pr = [(x, y) for x in R for y in R if x[0] < y[0] and x[3] != y[3]]
    ag = sum((x[1] - y[1]) * (x[3] - y[3]) > 0 for x, y in pr) / len(pr)
    off = [x[1] - x[3] for x in R]
    print(f"\n{tag}: {len(R)} systems, Spearman(first200, all500) = {spear([x[1] for x in R], [x[3] for x in R]):.2f}, "
          f"pairwise order agreement {100*ag:.0f}%, mean offset first200−all500 = {sum(off)/len(off):+.1f} pp (min {min(off):+.1f}, max {max(off):+.1f})")
    for lo, hi in [(0, 3), (3, 6), (6, 100)]:
        p2 = [(x, y) for x, y in pr if lo <= abs(x[3] - y[3]) < hi]
        if p2: print(f"   all-500 gap {lo}-{hi}pp: {len(p2)} pairs, order kept on first 200: {100*sum((x[1]-y[1])*(x[3]-y[3])>0 for x,y in p2)/len(p2):.0f}%")
