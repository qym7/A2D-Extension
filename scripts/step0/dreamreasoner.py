#!/usr/bin/env python3
"""Same question on DreamReasoner-8B: does its parent terminate and does the merge stop terminating?

DreamReasoner is a BLOCK diffusion model (block length 32) with an 8k budget, so the structure
differs from Dream-v0. Trajectories were already saved by step7_reasoning_merge.
Streamed line by line -- the files are up to 114 MB.
"""
import json, statistics as st
from pathlib import Path
EOS = {151643, 151645}
R = Path("/mnt/lts4/scratch/home/yqin/DllmAR/dream_study/step7_reasoning_merge/outputs/math500")
ARMS = [("parent-m500-nothink-8k-h200",  "DreamReasoner-8B parent"),
        ("rmg-b020-m500-nothink-8k-h200","reasoning merge beta=0.2"),
        ("drtr-m500-nothink-8k-h200",    "transfer-only"),
        ("drbase-m500-raw-8k-b300",      "DreamReasoner-Base raw")]
print(f"{'arm':<30}{'n':>4}{'EOS占提交':>11}{'长度中位':>10}{'长度均值':>10}"
      f"{'顶cap':>8}{'第一EOS步中位':>14}{'有EOS':>8}{'corr(步,位)':>12}")
for a, lab in ARMS:
    f = list((R / a / "traj").glob("*.traj.jsonl"))
    if not f: print(f"{lab:<30} no traj"); continue
    n = tot = eos_n = nany = 0; lens = []; first = []; capn = 0
    sx = sy = sxx = syy = sxy = 0.0; m = 0
    for line in open(f[0]):
        if not line.strip(): continue
        r = json.loads(line); us, ids, G = r["unmask_step"], r["final_ids"], r["gen_len"]
        n += 1
        e = next((i for i, t in enumerate(ids) if t in EOS), G)
        lens.append(e); capn += int(e >= G - 2)
        fe = None
        for i, t in enumerate(ids):
            if us[i] < 0: continue
            tot += 1
            if t in EOS:
                eos_n += 1
                x, y = us[i], i / G
                sx += x; sy += y; sxx += x*x; syy += y*y; sxy += x*y; m += 1
                if fe is None or us[i] < fe: fe = us[i]
        if fe is not None: first.append(fe); nany += 1
    ls = sorted(lens); fs = sorted(first)
    cr = float("nan")
    if m > 1:
        num = sxy - sx*sy/m
        den = ((sxx - sx*sx/m) * (syy - sy*sy/m)) ** 0.5
        cr = num/den if den else float("nan")
    print(f"{lab:<30}{n:>4}{100*eos_n/max(tot,1):>10.1f}%{ls[len(ls)//2]:>10}"
          f"{sum(ls)/len(ls):>10.0f}{100*capn/n:>7.1f}%"
          f"{(fs[len(fs)//2] if fs else -1):>14}{nany:>5}/{n}{cr:>+12.3f}")
