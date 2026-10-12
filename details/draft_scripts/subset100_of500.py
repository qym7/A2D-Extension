"""Random 100-of-500 subset check (2026-10-12, user). Zero GPU. For every system with all 500 ids, does a random 100
subset reproduce the full-500 ranking? Reports (a) one fixed subset (seed 0) side by side, (b) over 2000 random subsets,
P(pairwise order preserved) binned by the full-500 gap. Writes details/subset50/subset100_of500_ids.json + check_500.md."""
import json, random, itertools
from pathlib import Path
L = Path(__file__).resolve().parents[1] / "logs"; OUT = Path(__file__).resolve().parents[1] / "technical_details/subset_checks"
def T(p): return {r["id"]: r["ok"] for r in map(json.loads, open(L / p / "texts.jsonl"))}
S = {"Ins": T("lockstep1024/math500/parent-free"), "M(.4,.6)": T("lockstep1024/math500/merge-free"), "M(0,1)": T("lockstep1024/math500/l10-free"),
     "A(2.0)": T("lockstep1024/math500/add20-free"),
     "M46<=M01@0.3": T("gated1024/math500/e1b-mrg10-t03"), "M46<=M01@0.7": T("gated1024/math500/e1b-mrg10-t07"),
     "Ins<=M46@0.6": T("gated_ins_mrg0406_1024/math500/t06"), "Ins<=M46@0.8": T("gated_ins_mrg0406_1024/math500/t08"),
     "Ins<=A20@0.6": T("gated_ins_add-a10-b20_1024/math500/t06"), "Ins<=A20@0.8": T("gated_ins_add-a10-b20_1024/math500/t08"),
     "M46<=A20@0.6": T("gated_mrg0406_add-a10-b20_1024/math500/t06-merged"), "M46<=A20@0.8": T("gated_mrg0406_add-a10-b20_1024/math500/t08")}
ar = "/mnt/lts4/scratch/home/yqin/DllmAR/dream_study/step2_instruct/outputs/official_eval/mathins_rev-qweninst-plain-t1024-h200-full"
import glob
rows = [json.loads(l) for l in open(sorted(glob.glob(ar + "/*/samples_math_500_*.jsonl"))[-1])]
S["AR Qwen-Ins"] = {r["doc_id"]: int(bool(r["exact_match"])) for r in rows}
S = {k: v for k, v in S.items() if all(i in v for i in range(500))}
acc = lambda d, ids: 100 * sum(d[i] for i in ids) / len(ids)
full = {k: acc(v, range(500)) for k, v in S.items()}
rng = random.Random(0); sub0 = sorted(rng.sample(range(500), 100))
(OUT / "subset100_of500_ids.json").write_text(json.dumps(sub0))
md = ["# 100-of-500 random subset vs full 500 (2026-10-12)\n", "## Fixed subset (seed 0, ids in subset100_of500_ids.json)\n",
      "| system | full 500 | subset 100 | rank full | rank sub |", "|---|---:|---:|---:|---:|"]
sub = {k: acc(v, sub0) for k, v in S.items()}
rf = {k: r for r, k in enumerate(sorted(full, key=lambda k: -full[k]), 1)}; rs = {k: r for r, k in enumerate(sorted(sub, key=lambda k: -sub[k]), 1)}
for k in sorted(full, key=lambda k: -full[k]): md.append(f"| {k} | {full[k]:.1f} | {sub[k]:.1f} | {rf[k]} | {rs[k]} |")
# pairwise order preservation over random subsets, binned by full gap
pairs = [(a, b) for a, b in itertools.combinations(S, 2) if abs(full[a] - full[b]) > 0]
bins = {"<2": [], "2-4": [], "4-6": [], "6-10": [], ">=10": []}
def binof(g): return "<2" if g < 2 else "2-4" if g < 4 else "4-6" if g < 6 else "6-10" if g < 10 else ">=10"
rng = random.Random(1)
for _ in range(2000):
    ids = rng.sample(range(500), 100); a_ = {k: acc(v, ids) for k, v in S.items()}
    for a, b in pairs:
        g = abs(full[a] - full[b]); bins[binof(g)].append((a_[a] - a_[b]) * (full[a] - full[b]) > 0)
md += ["", "## Over 2000 random 100-subsets: P(pairwise order same as full 500), by full-500 gap (pp)\n", "| gap | pairs×draws | P(order kept) |", "|---|---:|---:|"]
for k, v in bins.items(): md.append(f"| {k} | {len(v)} | {100*sum(v)/len(v):.0f}% |" if v else f"| {k} | 0 | — |")
md += ["", "Note: a 100-subset has SE ≈ 5 pp on a single accuracy; only orderings with full-500 gaps ≳ 6 pp are reliably kept."]
(OUT / "check_500.md").write_text("\n".join(md) + "\n"); print("\n".join(md))
