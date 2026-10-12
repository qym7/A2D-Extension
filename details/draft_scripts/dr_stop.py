#!/usr/bin/env python3
"""How does DreamReasoner stop? If it halts on EOS, committed positions are a contiguous prefix
0..L-1 and everything after is never committed (unmask_step = -1). If it pads a fixed canvas like
Dream-v0, every position is committed. Also: are the EOS tokens a short run at the end of a block?"""
import json
from pathlib import Path
EOS={151643,151645}
f=list(Path("/mnt/lts4/scratch/home/yqin/DllmAR/dream_study/step7_reasoning_merge/outputs/math500/parent-m500-nothink-8k-h200/traj").glob("*.traj.jsonl"))[0]
n=0; prefix=0; runs=[]; blocks=[]
for line in open(f):
    if not line.strip(): continue
    r=json.loads(line); us,ids,G=r["unmask_step"],r["final_ids"],r["gen_len"]
    c=[i for i in range(G) if us[i]>=0]
    if not c: continue
    n+=1
    L=max(c)+1
    prefix += int(len(c)==L)            # committed positions form a contiguous prefix?
    blocks.append(L)
    # trailing EOS run length
    k=0
    for i in range(L-1,-1,-1):
        if ids[i] in EOS: k+=1
        else: break
    runs.append(k)
    if n<=3:
        print(f"  problem {n}: committed {len(c)} positions, max index {L-1}, "
              f"contiguous prefix={len(c)==L}, trailing EOS run={k}, L mod 32={L%32}")
    if n>=500: break
runs.sort(); blocks.sort()
print(f"\nn={n}")
print(f"committed positions form a contiguous prefix: {prefix}/{n}")
print(f"committed length L: median {blocks[len(blocks)//2]}, and L mod 32 == 0 for "
      f"{sum(1 for L in blocks if L%32==0)}/{n} problems  (block length 32?)")
print(f"trailing EOS run: median {runs[len(runs)//2]}, mean {sum(runs)/len(runs):.1f}, max {runs[-1]}")
