#!/usr/bin/env python3
"""Same as dose_length.py but on the INTERSECTION of all seven arms, so every number is on the
same problems. Adds the exact paired McNemar reasoning comparison on both-finished problems,
which is the only licensed cross-arm accuracy comparison."""
import json, os
from pathlib import Path
from math import comb
os.environ.setdefault("HF_HOME", "/mnt/lts4/scratch/home/yqin/DllmAR/dream_study/hf_cache")
from transformers import AutoTokenizer

CK = "/mnt/lts4/scratch/home/yqin/DllmAR/new_ideas/idea1_selective_distill/outputs/checkpoints/merge_l04"
tok = AutoTokenizer.from_pretrained(CK, trust_remote_code=True)
L = Path("/mnt/lts4/scratch/home/yqin/DllmAR/new_ideas/idea1_selective_distill/details/logs/lockstep/math500")
ARMS = [("s0-free-parent", "parent (tau_A = 0)"), ("e3-mrg04-free", "lambda 0.4"),
        ("s0-free-a2d", "lambda 0.6  <- A2D BEST"), ("e3-mrg08-free", "lambda 0.8  (over)"),
        ("e3-mrg10-free", "lambda 1.0  (over)"), ("donor-a12-free", "alpha 1.2   (over)"),
        ("donor-a15-free", "alpha 1.5   (over)")]

D = {}
for tag, lab in ARMS:
    p = L / tag / "texts.jsonl"
    if not p.exists(): continue
    per = {}
    for l in p.read_text().splitlines():
        if not l.strip(): continue
        r = json.loads(l); t = r["text"]
        per[r["id"]] = (int(r["ok"]), "\\boxed" in t,
                        len(tok(t, add_special_tokens=False).input_ids))
    D[tag] = (lab, per)

common = sorted(set.intersection(*[set(v[1]) for v in D.values()]), key=int)
print(f"all {len(D)} arms on the SAME {len(common)} MATH-500 problems "
      f"(free mode, single model, 512 steps, identical code path)\n")
print(f"{'arm':<28}{'official acc':>13}{'finish%':>9}{'acc|finished':>13}"
      f"{'len med':>9}{'at cap':>8}")
ref = None
tbl = {}
for tag, lab in ARMS:
    if tag not in D: continue
    per = D[tag][1]; v = [per[i] for i in common]; n = len(v)
    fin = [x for x in v if x[1]]
    tbl[tag] = v
    print(f"{lab:<28}{100*sum(a for a,_,_ in v)/n:>13.1f}"
          f"{100*len(fin)/n:>9.1f}"
          f"{(100*sum(a for a,_,_ in fin)/len(fin) if fin else float('nan')):>13.1f}"
          f"{sorted(c for _,_,c in v)[n//2]:>9}"
          f"{100*sum(1 for _,_,c in v if c>=510)/n:>7.1f}%")

def mcnemar(a, b):
    """exact two-sided McNemar on problems BOTH arms finished"""
    n01 = n10 = n = 0
    for x, y in zip(a, b):
        if not (x[1] and y[1]): continue
        n += 1
        if x[0] and not y[0]: n10 += 1
        elif y[0] and not x[0]: n01 += 1
    m = n01 + n10
    if m == 0: return n, 0.0, 1.0
    p = sum(comb(m, k) for k in range(0, min(n01, n10) + 1)) / 2 ** m * 2
    return n, 100 * (n01 - n10) / n, min(1.0, p)

base = tbl.get("s0-free-a2d")
print(f"\npaired reasoning vs lambda 0.6 (exact McNemar, only problems BOTH finished):")
for tag, lab in ARMS:
    if tag not in tbl or tag == "s0-free-a2d": continue
    n, d, p = mcnemar(base, tbl[tag])
    print(f"  {lab:<28} {d:+6.1f} pt   n={n:<4} p={p:.3f}")
