#!/usr/bin/env python3
"""The same analysis we did on Dream-v0, applied to DreamReasoner:
官方acc / 写完率 / 写完acc / 配对增益(精确McNemar) / 长度 / 顶cap.

MATH-500: per-problem records in results.partial.jsonl. Only the '-m500-8k' suffix group is used,
          so every arm shares one protocol (the suffix spread is 2.6-2.8 pts, larger than the sweep).
AIME24+25: no response text is stored, only eval.log per-problem lines, so length/degeneracy cannot
          be computed there -- acc / finish / finished-acc / paired McNemar can.
"""
import json, os, re
from math import comb
from pathlib import Path
os.environ.setdefault("HF_HOME","/mnt/lts4/scratch/home/yqin/DllmAR/dream_study/hf_cache")
from transformers import AutoTokenizer
tk=AutoTokenizer.from_pretrained("/mnt/lts4/scratch/home/yqin/DllmAR/new_ideas/idea1_selective_distill/outputs/checkpoints/merge_l04",trust_remote_code=True)
R=Path("/mnt/lts4/scratch/home/yqin/DllmAR/dream_study/step7_reasoning_merge/outputs")

def mcnemar(a,b,finA,finB):
    n01=n10=n=0
    for i in a:
        if not (finA[i] and finB.get(i)): continue
        n+=1
        if a[i] and not b[i]: n10+=1
        elif b[i] and not a[i]: n01+=1
    m=n01+n10
    if m==0: return n,0.0,1.0
    return n,100*(n01-n10)/n, min(1.0, 2*sum(comb(m,k) for k in range(min(n01,n10)+1))/2**m)

# ---------------- MATH-500 ----------------
ARMS=[("parent-m500-8k","lambda 0 (parent)",1.0),("rmg-b010-m500-8k","lambda 0.1",0.9),
      ("rmg-b020-m500-8k","lambda 0.2  <- sel",0.8),("rmg-b030-m500-8k","lambda 0.3",0.7),
      ("rmg-b040-m500-8k","lambda 0.4",0.6)]
D={}
for a,lab,tD in ARMS:
    f=R/"math500"/a/"results.partial.jsonl"
    if not f.is_file(): continue
    per={}
    for line in open(f):
        if not line.strip(): continue
        d=json.loads(line); resp=d.get("resp") or ""
        per[d["i"]]=dict(ok=bool(d["ok"]), fin=bool((d.get("pred") or "").strip()),
                         ntok=len(tk(resp,add_special_tokens=False).input_ids))
    D[a]=(lab,tD,per)
common=sorted(set.intersection(*[set(v[2]) for v in D.values()]))
ref="rmg-b020-m500-8k"
print(f"### DreamReasoner MATH-500 @8192 (suffix '-m500-8k', thinking on, same {len(common)} problems)\n")
print(f"{'arm':<22}{'tau_D':>6}{'官方acc':>8}{'写完率':>8}{'写完acc':>8}{'配对增益 vs λ0.2 (p)':>24}{'长度中位':>9}{'顶cap':>7}")
for a,lab,tD in ARMS:
    if a not in D: continue
    lab,tD,per=D[a]; v=[per[i] for i in common]; n=len(v)
    fin=[x for x in v if x["fin"]]; ls=sorted(x["ntok"] for x in v)
    okm={i:per[i]["ok"] for i in common}; fm={i:per[i]["fin"] for i in common}
    rr=D[ref][2]
    if a==ref: pair="参考"
    else:
        nn,dd,pp=mcnemar({i:rr[i]["ok"] for i in common},okm,{i:rr[i]["fin"] for i in common},fm)
        pair=f"{dd:+.1f} (n={nn}, p={pp:.3f})"
    print(f"{lab:<22}{tD:>6.1f}{100*sum(x['ok'] for x in v)/n:>8.1f}{100*len(fin)/n:>7.1f}%"
          f"{(100*sum(x['ok'] for x in fin)/len(fin) if fin else float('nan')):>8.1f}{pair:>24}"
          f"{ls[n//2]:>9}{100*sum(1 for x in v if x['ntok']>=8100)/n:>6.1f}%")

# ---------------- AIME ----------------
pat=re.compile(r"\[(aime\d+) (\d+)/(\d+)\] gold=(.*?) pred=(.*?) (OK|x) tok=(\d+)")
AARMS=[("parent","lambda 0 (parent)",1.0),("rmg-b010","lambda 0.1",0.9),("rmg-b020","lambda 0.2  <- sel",0.8),
       ("rmg-b030","lambda 0.3",0.7),("rmg-b040","lambda 0.4",0.6),("rmg-b060","lambda 0.6 OVER",0.4)]
A={}
for a,lab,tD in AARMS:
    f=R/"aime"/a/"eval.log"
    if not f.is_file(): continue
    per={}
    for line in open(f, errors="ignore"):
        m=pat.search(line)
        if m:
            t,idx,_,gold,pred,ok,tok=m.groups()
            per[f"{t}-{idx}"]=dict(ok=(ok=="OK"), fin=bool(pred.strip()) and pred.strip() not in ("None","-"))
    if per: A[a]=(lab,tD,per)
if A:
    com=sorted(set.intersection(*[set(v[2]) for v in A.values()]))
    ref="rmg-b020"
    print(f"\n### DreamReasoner AIME24+25 @8192 (same {len(com)} problems; 每题都顶 8192，长度无变化)\n")
    print(f"{'arm':<22}{'tau_D':>6}{'官方acc':>8}{'写完率':>8}{'写完acc':>8}{'配对增益 vs λ0.2 (p)':>24}")
    for a,lab,tD in AARMS:
        if a not in A: continue
        lab,tD,per=A[a]; v=[per[i] for i in com]; n=len(v)
        fin=[x for x in v if x["fin"]]
        if a==ref: pair="参考"
        else:
            rr=A[ref][2]
            nn,dd,pp=mcnemar({i:rr[i]["ok"] for i in com},{i:per[i]["ok"] for i in com},
                             {i:rr[i]["fin"] for i in com},{i:per[i]["fin"] for i in com})
            pair=f"{dd:+.1f} (n={nn}, p={pp:.3f})"
        print(f"{lab:<22}{tD:>6.1f}{100*sum(x['ok'] for x in v)/n:>8.1f}{100*len(fin)/n:>7.1f}%"
              f"{(100*sum(x['ok'] for x in fin)/len(fin) if fin else float('nan')):>8.1f}{pair:>24}")
