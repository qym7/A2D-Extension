#!/usr/bin/env python3
"""Answer length = position of the LEFTMOST EOS in the answer window (that is where scoring stops).
Plus: does the driver ever emit EOS at all? And the schedule question -- at the parent's own
padding positions, does the other model AGREE that EOS belongs there (teacher-forced)?
"""
import json
from collections import Counter
EOS = {151643, 151645}
R = "/mnt/lts4/scratch/home/yqin/DllmAR/dream_study/step2_instruct/outputs/official_eval/stepdiv"

def final_of(prob, key):
    f = {}
    for r in prob["records"]:
        for k, q in enumerate(r["committed"]): f[q] = r[key][k]
    return f

for tag in ("ent-dream-mrg", "ent-dream-aq"):
    d = json.loads(open(f"{R}/{tag}/step_divergence.json").read()); M = d["max_new"]
    lens, noeos = [], 0
    for prob in d["problems"]:
        P = prob["prompt_len"]; f = final_of(prob, "tok_parent")
        e = next((q - P for q in range(P, P + M) if f.get(q) in EOS), None)
        if e is None: noeos += 1; lens.append(M)
        else: lens.append(e)
    lens.sort()
    print(f"\n{tag}  driver = {d['parent'].split('/')[-1]}")
    print(f"  answer length (leftmost EOS): median {lens[len(lens)//2]}, "
          f"p10 {lens[10]}, p90 {lens[90]}, mean {sum(lens)/len(lens):.1f} / budget {M}")
    print(f"  never emitted any EOS: {noeos}/{len(lens)};  "
          f"used the whole budget (len>={M}): {sum(1 for l in lens if l>=M)}/{len(lens)}")

# schedule vs content: at the PARENT's padding positions, does the other model agree EOS belongs there?
print("\n=== teacher-forced agreement at the parent's padding positions ===")
for tag, what in (("ent-dream-ad", "other = 0.4*tau_D only  (NO AR)"),
                  ("ent-dream-mrg", "other = full merge      (0.4*tau_D + 0.6*tau_A)")):
    d = json.loads(open(f"{R}/{tag}/step_divergence.json").read()); M = d["max_new"]
    agree = tot = 0
    for prob in d["problems"]:
        P = prob["prompt_len"]; fp = final_of(prob, "tok_parent")
        nz = [q for q in range(P, P + M) if fp.get(q) not in EOS]
        bnd = (max(nz) + 1) if nz else P
        for r in prob["records"]:
            for k, q in enumerate(r["committed"]):
                if q < bnd: continue
                if fp.get(q) not in EOS: continue
                tot += 1; agree += int(r["tok_merge"][k] in EOS)
    print(f"  {what}: writes EOS at {100*agree/tot:.1f}% of them  (n={tot})")
