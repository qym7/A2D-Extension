#!/usr/bin/env python3
"""Everything, from OFFICIAL lm-eval runs on MATH-500 @512 whose denoising trajectories were saved.

Each traj record: unmask_step[i] = the step at which generated position i was committed,
final_ids[i] = the token there, sel_prob[i] / entropy[i] = the selection stats.
So first-EOS STEP, EOS share, corr(step,position) and answer length are all offline-computable.
"""
import json, glob, statistics as st
from pathlib import Path

EOS = {151643, 151645}
R = Path("/mnt/lts4/scratch/home/yqin/DllmAR/dream_study/step2_instruct/outputs/official_eval")
ARMS = [("mathins_m500-dream-ins-09aD-00aA-plain", "0.9 tau_D + 0.0 tau_A", 39.6),
        ("mathins_transfer_b09",                   "0.0 tau_D + 0.9 tau_A", 40.2),
        ("mathins_m500-dream-ins-095aD-055aA-plain","0.95 tau_D + 0.55 tau_A", 48.6),
        ("mathins_m500-dream-ins-10aD-11aA-plain", "1.0 tau_D + 1.1 tau_A", 50.0),
        ("mathins_m500-dream-ins-09aD-11aA-plain", "0.9 tau_D + 1.1 tau_A", 51.0)]

def acc_official(a):
    for f in glob.glob(str(R / a / "**" / "results*.json"), recursive=True):
        d = json.load(open(f))
        for k, r in d.get("results", d).items():
            for kk in r:
                if kk.startswith("exact_match,"): return 100 * r[kk]
    return None

print(f"OFFICIAL lm-eval, MATH-500, plain prompt, 512 tokens / 512 steps, n=500\n")
hdr = (f"| {'arm':<24} | acc | 第一个EOS步(中位) | 有EOS的题 | EOS占提交 | "
       f"corr(步,位置) | 答案长度中位 | 顶cap | 第1步位置 | 第1步EOS |")
print(hdr)
print("|" + "|".join("-" * len(c) for c in hdr.split("|")[1:-1]) + "|")
out = {}
for a, lab, acc_known in ARMS:
    f = list((R / a / "traj").glob("*.traj.jsonl"))
    if not f: print(f"| {lab:<24} | no traj |"); continue
    recs = [json.loads(l) for l in f[0].read_text().splitlines() if l.strip()]
    first, lens, eos_n, tot, sweep, nany = [], [], 0, 0, [], 0
    s1pos, s1eos, s1n = 0.0, 0, 0
    for r in recs:
        us, ids, G = r["unmask_step"], r["final_ids"], r["gen_len"]
        e = next((i for i, t in enumerate(ids) if t in EOS), G)
        lens.append(e)
        fe = None
        for i, t in enumerate(ids):
            if us[i] < 0: continue
            tot += 1
            if us[i] == min(x for x in us if x >= 0):
                s1pos += i / G; s1eos += int(t in EOS); s1n += 1
            if t in EOS:
                eos_n += 1; sweep.append((us[i], i / G))
                if fe is None or us[i] < fe: fe = us[i]
        if fe is not None: first.append(fe); nany += 1
    fs, ls = sorted(first), sorted(lens)
    xs = [x for x, _ in sweep]; ys = [y for _, y in sweep]
    mx, my = st.mean(xs), st.mean(ys)
    cr = sum((x-mx)*(y-my) for x, y in sweep) / ((sum((x-mx)**2 for x in xs)*sum((y-my)**2 for y in ys))**0.5)
    acc = acc_official(a)
    print(f"| {lab:<24} | {acc:.1f} | {fs[len(fs)//2]:>17} | {nany:>4}/{len(recs)} | "
          f"{100*eos_n/tot:>8.1f}% | {cr:>+13.3f} | {ls[len(ls)//2]:>12} | "
          f"{100*sum(1 for l in lens if l>=510)/len(lens):>4.1f}% | {s1pos/s1n:>9.2f} | {100*s1eos/s1n:>7.1f}% |")
    out[lab] = (acc, cr, 100*eos_n/tot)
