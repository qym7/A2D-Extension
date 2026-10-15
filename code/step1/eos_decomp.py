#!/usr/bin/env python3
"""What happens at step 1 (and the first few steps), from already-stored artifacts.

Source A: step_divergence JSONs (GSM8K, 256 tok) -- gives parent (driver), best merge lambda=0.6
          (other), and the lambda=0 ablation. conf_* are RAW softmax probabilities.
Source B: eosdiag-a15-rec/records/*.json (MATH-500, 512 tok) -- full per-step trajectory of the
          alpha=1.5 over-merge driving itself. p_driver_raw is the raw probability.
"""
import json
from pathlib import Path
from collections import Counter

EOS = {151643, 151645}
SD = Path("/mnt/lts4/scratch/home/yqin/DllmAR/dream_study/step2_instruct/outputs/official_eval/stepdiv")

print("=" * 92)
print("SOURCE A -- GSM8K, 256-slot canvas, parent drives (teacher-forced)")
print("=" * 92)
for tag, lab in [("ent-dream-mrg", "other = BEST MERGE lambda=0.6"),
                 ("ent-dream-ad",  "other = lambda=0 ablation (0.4*tau_D, no AR)")]:
    d = json.loads((SD / tag / "step_divergence.json").read_text())
    M = d["max_new"]
    for nstep in (1, 2, 3):
        dp = de = 0; mp = me = 0; cd = cm = 0.0; n = 0; relpos = 0.0
        toks = Counter()
        for prob in d["problems"]:
            P = prob["prompt_len"]
            for r in prob["records"]:
                if r["step"] != nstep: continue
                for k, q in enumerate(r["committed"]):
                    n += 1; relpos += (q - P) / M
                    de += int(r["tok_parent"][k] in EOS); cd += r["conf_parent"][k]
                    me += int(r["tok_merge"][k] in EOS); cm += r["conf_merge"][k]
                    if r["tok_merge"][k] not in EOS: toks[r["tok_merge"][k]] += 1
        if n == 0: continue
        if nstep == 1:
            print(f"\n{lab}")
        print(f"  step {nstep}: n={n:3d}  mean rel position {relpos/n:.2f}  |  "
              f"parent writes EOS {100*de/n:5.1f}% (conf {cd/n:.3f})  |  "
              f"other writes EOS {100*me/n:5.1f}% (its conf {cm/n:.3f})")
        if nstep == 1 and toks:
            print(f"           what the other writes instead (top 5 ids): {[t for t,_ in toks.most_common(5)]}")

print()
print("=" * 92)
print("SOURCE B -- MATH-500, 512-slot canvas, alpha=1.5 OVER-MERGE driving ITSELF (free mode)")
print("=" * 92)
REC = Path("/mnt/lts4/scratch/home/yqin/DllmAR/new_ideas/idea1_selective_distill/"
           "outputs/step0/lockstep/math500/eosdiag-a15-rec/records")
files = sorted(REC.glob("*.json"))
print(f"{len(files)} problems")
per_step = {}
first_eos_step, first_eos_pos, n_any_eos, tot_eos, tot_com = [], [], 0, 0, 0
sweep = []
for f in files:
    r = json.loads(f.read_text())
    P = r["prompt_len"]; S = r["steps"]
    fe = None
    for rec in S:
        sf = rec["step"]
        for k, q in enumerate(rec["pos"]):
            t = rec["tok_driver"][k]; rp = q / 512.0
            e = per_step.setdefault(sf, [0, 0, 0.0, 0.0])
            e[0] += 1; e[2] += rp; e[3] += rec["p_driver_raw"][k]
            tot_com += 1
            if t in EOS:
                e[1] += 1; tot_eos += 1; sweep.append((sf, rp))
                if fe is None: fe = (sf, rp)
    if fe: first_eos_step.append(fe[0]); first_eos_pos.append(fe[1]); n_any_eos += 1
print(f"of {tot_com} commits, {100*tot_eos/tot_com:.1f}% are EOS   "
      f"(pure diffusion parent on GSM8K: 92.8%)")
print(f"problems that ever commit an EOS: {n_any_eos}/{len(files)}")
if first_eos_step:
    fs = sorted(first_eos_step); fp = sorted(first_eos_pos)
    print(f"first EOS commit: median at step {fs[len(fs)//2]} of 512, "
          f"relative position {fp[len(fp)//2]:.2f}    (parent: step 1, position 1.00)")
print(f"\n{'step':>6} {'commits':>8} {'% EOS':>7} {'mean rel pos':>13} {'mean raw conf':>14}")
for s in sorted(per_step)[:6]:
    e = per_step[s]
    print(f"{s:>6} {e[0]:>8} {100*e[1]/e[0]:>6.1f}% {e[2]/e[0]:>13.2f} {e[3]/e[0]:>14.3f}")
if sweep:
    import statistics as st
    xs=[a for a,_ in sweep]; ys=[b for _,b in sweep]
    mx,my=st.mean(xs),st.mean(ys)
    num=sum((a-mx)*(b-my) for a,b in sweep)
    den=(sum((a-mx)**2 for a in xs)*sum((b-my)**2 for b in ys))**0.5
    print(f"\ncorr(step, position) over its EOS commits = {num/den:+.3f}   "
          f"(parent on GSM8K: -0.982, i.e. right-to-left)")
