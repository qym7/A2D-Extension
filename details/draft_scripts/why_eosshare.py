#!/usr/bin/env python3
"""Is 'EOS share of commits' comparable across architectures? Check how much of the canvas is
actually committed at all (unmask_step == -1 means never committed), and how many EOS tokens
each problem emits. Dream-v0 = full canvas, no early stop -> every slot must be filled.
DreamReasoner = block diffusion with EOS early stopping -> generation halts, slots stay uncommitted.
"""
import json
from pathlib import Path
EOS = {151643, 151645}
SETS = [
 ("DreamReasoner parent", "/mnt/lts4/scratch/home/yqin/DllmAR/dream_study/step7_reasoning_merge/outputs/math500/parent-m500-nothink-8k-h200/traj"),
 ("DreamReasoner merge b0.2", "/mnt/lts4/scratch/home/yqin/DllmAR/dream_study/step7_reasoning_merge/outputs/math500/rmg-b020-m500-nothink-8k-h200/traj"),
 ("DreamReasoner transfer-only", "/mnt/lts4/scratch/home/yqin/DllmAR/dream_study/step7_reasoning_merge/outputs/math500/drtr-m500-nothink-8k-h200/traj"),
 ("Dream-v0 0.9tD+1.1tA", "/mnt/lts4/scratch/home/yqin/DllmAR/dream_study/step2_instruct/outputs/official_eval/mathins_m500-dream-ins-09aD-11aA-plain/traj"),
 ("Dream-v0 0.9tD+0.0tA", "/mnt/lts4/scratch/home/yqin/DllmAR/dream_study/step2_instruct/outputs/official_eval/mathins_m500-dream-ins-09aD-00aA-plain/traj"),
]
print(f"{'arm':<30}{'budget':>8}{'已提交位置占比':>15}{'每题EOS个数中位':>17}{'EOS占已提交':>13}")
for lab, d in SETS:
    f = list(Path(d).glob("*.traj.jsonl"))
    if not f: print(f"{lab:<30} no traj"); continue
    com = tot = eos = 0; per = []
    for line in open(f[0]):
        if not line.strip(): continue
        r = json.loads(line); us, ids, G = r["unmask_step"], r["final_ids"], r["gen_len"]
        tot += G
        c = [i for i in range(G) if us[i] >= 0]
        com += len(c)
        ne = sum(1 for i in c if ids[i] in EOS)
        eos += ne; per.append(ne)
    per.sort()
    print(f"{lab:<30}{r['gen_len']:>8}{100*com/tot:>14.1f}%{per[len(per)//2]:>17}{100*eos/com:>12.1f}%")
