#!/usr/bin/env python3
"""What does losing EOS cost a BLOCK diffusion model? Generated tokens = committed positions.
A full-canvas model always pays gen_len; a block model pays only what it generates before halting."""
import json
from pathlib import Path
EOS={151643,151645}
R=Path("/mnt/lts4/scratch/home/yqin/DllmAR/dream_study/step7_reasoning_merge/outputs/math500")
for a,lab in [("parent-m500-nothink-8k-h200","DreamReasoner parent (tau_D=1.0)"),
              ("rmg-b020-m500-nothink-8k-h200","reasoning merge b=0.2 (tau_D=0.8)"),
              ("drtr-m500-nothink-8k-h200","transfer-only (tau_D=0)")]:
    f=list((R/a/"traj").glob("*.traj.jsonl"))
    if not f: continue
    tot=0; n=0; halted=0; G=0
    for line in open(f[0]):
        if not line.strip(): continue
        r=json.loads(line); us,G=r["unmask_step"],r["gen_len"]
        c=sum(1 for x in us if x>=0); tot+=c; n+=1
        halted += int(c < G)
    print(f"{lab:<36} 平均生成 {tot/n:7.0f} token / 预算 {G}   halt 掉的题 {halted}/{n} "
          f"({100*halted/n:.0f}%)   相对 parent 的生成成本 x{(tot/n)/1:.0f}")
