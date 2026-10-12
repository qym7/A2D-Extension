#!/usr/bin/env python3
"""Concretely: how does Dream-v0 put EOS on the canvas? Official runs, MATH-500 @512.
unmask_step[i] is the step position i was committed, final_ids[i] the token there."""
import json, os
from pathlib import Path
os.environ.setdefault("HF_HOME","/mnt/lts4/scratch/home/yqin/DllmAR/dream_study/hf_cache")
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained("/mnt/lts4/scratch/home/yqin/DllmAR/new_ideas/idea1_selective_distill/outputs/checkpoints/merge_l04",trust_remote_code=True)
EOS={151643,151645}
R=Path("/mnt/lts4/scratch/home/yqin/DllmAR/dream_study/step2_instruct/outputs/official_eval")
for a,lab in [("mathins_m500-dream-ins-09aD-00aA-plain","0.9*tau_D + 0.0*tau_A   (acc 39.6, 247 EOS/题)"),
              ("mathins_m500-dream-ins-09aD-11aA-plain","0.9*tau_D + 1.1*tau_A   (acc 51.0, 1 EOS/题)")]:
    f=list((R/a/"traj").glob("*.traj.jsonl"))[0]
    print(f"\n{'='*94}\n{lab}\n{'='*94}")
    shown=0
    for line in open(f):
        if not line.strip(): continue
        r=json.loads(line); us,ids,G=r["unmask_step"],r["final_ids"],r["gen_len"]
        eo=[i for i in range(G) if ids[i] in EOS]
        L=eo[0] if eo else None
        # 画布地图: 每个字符 = 8 格, E=全是EOS, e=部分, .=没有EOS
        m="".join("E" if all(ids[j] in EOS for j in range(k,min(k+8,G)))
                  else ("e" if any(ids[j] in EOS for j in range(k,min(k+8,G))) else ".")
                  for k in range(0,G,8))
        contig = (L is not None) and all(ids[j] in EOS for j in range(L,G))
        print(f"\n题 {r['key'][:8]}  第一个EOS位置={L}  EOS总数={len(eo)}  "
              f"从L到末尾全是EOS={contig}  提交步序是否严格左→右={all(us[i]==i+1 for i in range(G))}")
        print(f"  画布(每字符=8格, E=该段全EOS, .=无EOS): {m}")
        if L is not None:
            print(f"  L 之前最后 12 个 token: {tok.decode(ids[max(0,L-12):L])!r}")
            print(f"  L 处那一格被提交的步数 = {us[L]}  (总 512 步)")
        shown+=1
        if shown>=3: break
