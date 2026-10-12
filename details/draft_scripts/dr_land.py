#!/usr/bin/env python3
"""Fast landscape pass over every DreamReasoner MATH-500 arm with 500 results on disk.
No tokenizer (that would be slow over ~60 arms x 500 x up to 8k tokens): char length is used as a
RELATIVE length measure within this set, and flagged as such. 'at-budget' uses char length >= 0.95
of that arm's own max, which is a proxy -- the token-exact version comes in the zoom-in pass.
"""
import json, re
from pathlib import Path
R=Path("/mnt/lts4/scratch/home/yqin/DllmAR/dream_study/step7_reasoning_merge/outputs/math500")
def rep(s,n=40):
    if len(s)<2*n: return 0.0
    sh=[s[i:i+n] for i in range(0,len(s)-n,n)]
    return 1-len(set(sh))/len(sh)
rows=[]
for d in sorted(R.iterdir()):
    f=d/"results.partial.jsonl"
    if not f.is_file(): continue
    recs=[]
    for line in open(f):
        if line.strip(): recs.append(json.loads(line))
    if len(recs)<400: continue
    n=len(recs)
    ok=sum(1 for r in recs if r.get("ok"))
    pred=sum(1 for r in recs if (r.get("pred") or "").strip())
    L=sorted(len(r.get("resp") or "") for r in recs)
    rp=[rep(r.get("resp") or "") for r in recs]
    deg=sum(1 for x in rp if x>0.3)
    rows.append((d.name,n,100*ok/n,100*pred/n,L[n//2],100*deg/n))
rows.sort(key=lambda r:-r[2])
print(f"{'arm':<42}{'n':>5}{'acc':>7}{'有pred':>8}{'字符长度中位':>13}{'复读率':>8}")
for a,n,o,p,l,dg in rows:
    print(f"{a:<42}{n:>5}{o:>7.1f}{p:>7.1f}%{l:>13}{dg:>7.1f}%")
