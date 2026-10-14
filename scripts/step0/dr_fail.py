#!/usr/bin/env python3
"""DreamReasoner: on which problems does the over-merged arm lose to the best merge, and why?
The only over-merged DreamReasoner arm with outputs on disk is transfer-only (tau_D = 0, acc 70.4);
there is no beta=0.4/0.6 convex checkpoint, so this is the tau_D=0 extreme, not a mild over-merge.
"""
import json, os, re
from pathlib import Path
os.environ.setdefault("HF_HOME","/mnt/lts4/scratch/home/yqin/DllmAR/dream_study/hf_cache")
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained("/mnt/lts4/scratch/home/yqin/DllmAR/new_ideas/idea1_selective_distill/outputs/checkpoints/merge_l04",trust_remote_code=True)
R=Path("/mnt/lts4/scratch/home/yqin/DllmAR/dream_study/step7_reasoning_merge/outputs/math500")
ARMS=[("parent-m500-nothink-8k-h200","parent (tau_D=1.0)"),
      ("rmg-b020-m500-nothink-8k-h200","best merge convex b=0.2 (tau_D=0.8)"),
      ("drtr-m500-nothink-8k-h200","transfer-only (tau_D=0)  <- over-merged")]

def rep_ratio(s, n=40):
    """fraction of 40-char shingles that are duplicates -- a degeneracy detector"""
    if len(s) < 2*n: return 0.0
    sh=[s[i:i+n] for i in range(0,len(s)-n,n)]
    return 1 - len(set(sh))/len(sh)

D={}
for a,lab in ARMS:
    per={}
    for line in open(R/a/"results.partial.jsonl"):
        if not line.strip(): continue
        d=json.loads(line)
        resp=d.get("resp") or ""
        per[d["i"]]=dict(ok=bool(d["ok"]), pred=(d.get("pred") or "").strip(),
                         resp=resp, ntok=len(tok(resp,add_special_tokens=False).input_ids),
                         rep=rep_ratio(resp), boxed=("\\boxed" in resp))
    D[a]=(lab,per)
common=sorted(set.intersection(*[set(v[1]) for v in D.values()]))
print(f"同样 {len(common)} 道题\n")
print(f"{'arm':<38}{'acc':>6}{'有pred':>8}{'有boxed':>9}{'长度中位':>10}{'重复度中位':>11}{'顶预算':>8}")
for a,lab in ARMS:
    lab,per=D[a]; v=[per[i] for i in common]; n=len(v)
    ls=sorted(x["ntok"] for x in v); rs=sorted(x["rep"] for x in v)
    print(f"{lab:<38}{100*sum(x['ok'] for x in v)/n:>6.1f}"
          f"{100*sum(1 for x in v if x['pred'])/n:>7.1f}%"
          f"{100*sum(x['boxed'] for x in v)/n:>8.1f}%"
          f"{ls[n//2]:>10}{rs[n//2]:>11.2f}"
          f"{100*sum(1 for x in v if x['ntok']>=8000)/n:>7.1f}%")

best=D["rmg-b020-m500-nothink-8k-h200"][1]; over=D["drtr-m500-nothink-8k-h200"][1]
lose=[i for i in common if best[i]["ok"] and not over[i]["ok"]]
win =[i for i in common if over[i]["ok"] and not best[i]["ok"]]
print(f"\n最佳 merge 对、过度 merge 错: {len(lose)} 道")
print(f"过度 merge 对、最佳 merge 错: {len(win)} 道   (净 -{len(lose)-len(win)})")
cats={"没产出pred(无法解析)":0,"有pred但答错":0}
deg=0; cap=0
for i in lose:
    o=over[i]
    if not o["pred"]: cats["没产出pred(无法解析)"]+=1
    else: cats["有pred但答错"]+=1
    deg+=int(o["rep"]>0.3); cap+=int(o["ntok"]>=8000)
print(f"\n在这 {len(lose)} 道输掉的题上，过度 merge 的输出长什么样：")
for k,v in cats.items(): print(f"  {k}: {v} ({100*v/len(lose):.0f}%)")
print(f"  重复度>0.3（退化复读）: {deg} ({100*deg/len(lose):.0f}%)")
print(f"  用满预算(>=8000 tok): {cap} ({100*cap/len(lose):.0f}%)")
print(f"\n前 2 道输掉的题，过度 merge 输出的尾部 300 字符：")
for i in lose[:2]:
    o=over[i]; b=best[i]
    print(f"\n--- 题 {i}  gold={b['pred']!r}(对)  过度merge pred={o['pred']!r}  "
          f"长度={o['ntok']} 重复度={o['rep']:.2f}")
    print("   ..." + repr(o["resp"][-300:]))
