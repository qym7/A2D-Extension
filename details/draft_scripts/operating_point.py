#!/usr/bin/env python3
"""Operating-point table for a confidence threshold theta that decides, per committed position,
whether the OVER-MERGED model or the PARENT gets to write.

  conf < theta  -> over-merge writes   (this is where the AR reasoning effect lives)
  conf >= theta -> parent writes       (this is where the EOS padding behaviour lives)

Columns:
  keep_AR     fraction of the AR vector's CONTENT changes that still happen (run C, conf < theta)
  reclaim_EOS fraction of the merge's PADDING damage the parent takes back (run A, conf >= theta)
  content_cost fraction of ALL content commits that get overridden despite the models disagreeing
               there, i.e. AR content changes we throw away (run C, conf >= theta, as % of commits)
  pos_frac    fraction of commits the parent writes

Also a step-profile, to tell whether one time-phased lambda could do the same job with ONE model.
"""
import json
from collections import defaultdict

EOS = {151643, 151645}
R = "/mnt/lts4/scratch/home/yqin/DllmAR/dream_study/step2_instruct/outputs/official_eval/stepdiv"

def load(tag):
    return json.loads(open(f"{R}/{tag}/step_divergence.json").read())

def collect(d, region):
    """yield (conf_parent, disagree, step_frac) for commits in the requested region"""
    M, S = d["max_new"], d["steps"]
    out = []
    for prob in d["problems"]:
        P = prob["prompt_len"]; final = {}
        for r in prob["records"]:
            for k, q in enumerate(r["committed"]): final[q] = r["tok_parent"][k]
        nonpad = [q for q in range(P, P + M) if final.get(q) not in EOS]
        bnd = (max(nonpad) + 1) if nonpad else P
        for r in prob["records"]:
            for k, q in enumerate(r["committed"]):
                w = "content" if q < bnd else "pad"
                if w != region: continue
                out.append((r["conf_parent"][k],
                            int(r["tok_parent"][k] != r["tok_merge"][k]),
                            r["step"] / S))
    return out

C = collect(load("ent-dream-aq"), "content")   # AR vector isolated, content region
A = collect(load("ent-dream-mrg"), "pad")      # real parent vs full merge, padding region

nC, dC = len(C), sum(x[1] for x in C)
nA, dA = len(A), sum(x[1] for x in A)
print(f"content region (AR-vector arm): {nC} commits, {dC} AR-induced changes ({100*dC/nC:.2f}%)")
print(f"padding region (parent vs merge): {nA} commits, {dA} padding disagreements ({100*dA/nA:.2f}%)")

print(f"\n{'theta':>6} {'keep_AR':>9} {'reclaim_EOS':>12} {'content lost':>13} {'parent writes':>14}")
for th in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99):
    keep = sum(x[1] for x in C if x[0] < th)
    lost = dC - keep
    recl = sum(x[1] for x in A if x[0] >= th)
    pfrac = sum(1 for x in C if x[0] >= th) / nC
    print(f"{th:6.2f} {100*keep/dC:8.1f}% {100*recl/dA:11.1f}% "
          f"{lost:6d} tok ({100*lost/nC:.3f}% of commits) {100*pfrac:9.1f}%")

print("\n=== step profile: when in the trajectory does each effect happen? ===")
print(f"{'step decile':>12} {'AR content changes':>20} {'padding disagreements':>23}")
bc = defaultdict(lambda: [0, 0]); ba = defaultdict(lambda: [0, 0])
for cp, dd, sf in C:
    b = min(int(sf * 10), 9); bc[b][0] += 1; bc[b][1] += dd
for cp, dd, sf in A:
    b = min(int(sf * 10), 9); ba[b][0] += 1; ba[b][1] += dd
for b in range(10):
    c1 = f"{100*bc[b][1]/dC:6.1f}% of all" if bc[b][0] else "          -"
    c2 = f"{100*ba[b][1]/dA:6.1f}% of all" if ba[b][0] else "          -"
    print(f"  [{b/10:.1f},{(b+1)/10:.1f})".rjust(12) + f"{c1:>20}{c2:>23}")
