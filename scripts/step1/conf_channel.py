#!/usr/bin/env python3
"""Final version. Boundary = 1 + last position where the DRIVER committed a non-EOS token
(the content extent; everything past it is right-hand EOS padding). Robust to interleaved EOS.

Question: does the AR vector's CONTENT effect and its PADDING/termination effect live at
different driver confidences? If yes, one threshold separates them and you can over-merge the
content channel while keeping the parent's termination.
"""
import json, sys
from collections import defaultdict, Counter

EOS = {151643, 151645}

def run(path, label):
    d = json.loads(open(path).read()); M = d["max_new"]
    reg = {"content": defaultdict(lambda: [0, 0]), "pad": defaultdict(lambda: [0, 0])}
    cm = {"content": defaultdict(lambda: [0.0, 0]), "pad": defaultdict(lambda: [0.0, 0])}
    nc = Counter(); lens = []; interleaved = 0; oth_eos = [0, 0]
    for prob in d["problems"]:
        P = prob["prompt_len"]; final = {}
        for r in prob["records"]:
            for k, q in enumerate(r["committed"]): final[q] = r["tok_parent"][k]
        nonpad = [q for q in range(P, P + M) if final.get(q) not in EOS]
        bnd = (max(nonpad) + 1) if nonpad else P
        lens.append(bnd - P)
        interleaved += sum(1 for q in range(P, bnd) if final.get(q) in EOS) > 0
        for r in prob["records"]:
            for k, q in enumerate(r["committed"]):
                tp, tm = r["tok_parent"][k], r["tok_merge"][k]
                b = min(int(r["conf_parent"][k] * 10), 9)
                w = "content" if q < bnd else "pad"
                nc[w] += 1; reg[w][b][0] += 1
                if tp != tm:
                    reg[w][b][1] += 1
                    cm[w][b][0] += r["conf_merge"][k]; cm[w][b][1] += 1
                    if w == "pad":
                        oth_eos[0] += int(tm in EOS); oth_eos[1] += 1
    N = sum(nc.values())
    print(f"\n{'='*94}\n{label}\n  driver={d['parent'].split('/')[-1]}   other={d['merge'].split('/')[-1]}   n={d['n_problems']}\n{'='*94}")
    print(f"driver content extent: mean {sum(lens)/len(lens):.1f} tok, median {sorted(lens)[len(lens)//2]}, "
          f"max {max(lens)}/{M}; filled the whole budget: {sum(1 for l in lens if l>=M)}/{len(lens)}; "
          f"problems with EOS interleaved inside the content: {interleaved}/{len(lens)}")
    for w in ("content", "pad"):
        n = nc[w]; dd = sum(v[1] for v in reg[w].values())
        print(f"\n--- {w.upper()}: {n} commits ({100*n/N:.0f}%), disagreement {100*dd/max(n,1):.2f}% ---")
        print(f"{'driver conf':>13} {'mass':>7} {'disagree%':>10} {'n_dis':>7}")
        for b in range(10):
            c, dq = reg[w][b]
            if c == 0: continue
            print(f"  [{b/10:.1f},{(b+1)/10:.1f})".rjust(13) + f"{100*c/n:6.1f}%" + f"{100*dq/c:10.2f}%" + f"{dq:7d}")
        # the separability number
        lo = sum(reg[w][b][1] for b in range(0, 7)); hi = sum(reg[w][b][1] for b in range(7, 10))
        print(f"  -> {100*lo/max(lo+hi,1):.1f}% of this region's disagreements sit at driver conf < 0.7")
    if oth_eos[1]:
        print(f"\n  in PAD, on disagreement the other model writes EOS {100*oth_eos[0]/oth_eos[1]:.1f}% "
              f"of the time ({100-100*oth_eos[0]/oth_eos[1]:.1f}% it refuses to pad)")

R = "/mnt/lts4/scratch/home/yqin/DllmAR/dream_study/step2_instruct/outputs/official_eval/stepdiv"
for tag, lab in [("ent-dream-aq", "C) driver = 0.4tau_D (no AR) | other = +0.6tau_A   <<< ISOLATES THE AR VECTOR"),
                 ("ent-dream-mrg", "A) driver = Dream-Instruct parent | other = full merge"),
                 ("ent-dream-ad", "B) driver = Dream-Instruct parent | other = 0.4tau_D only (no AR) [control]")]:
    run(f"{R}/{tag}/step_divergence.json", lab)
