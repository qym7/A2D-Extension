#!/usr/bin/env python3
"""When, in its own denoising trajectory, does a pure diffusion LM commit EOS?

driver of ent-dream-mrg / ent-dream-ad = Dream-v0-Instruct-7B (the real parent, no merge).
driver of ent-dream-aq                 = D_b + 0.4*tau_D (the no-AR reconstruction).
Teacher-forced: the driver commits, so its commits ARE its own trajectory.
"""
import json
from collections import defaultdict

EOS = {151643, 151645}
R = "/mnt/lts4/scratch/home/yqin/DllmAR/dream_study/step2_instruct/outputs/official_eval/stepdiv"

def report(tag, label):
    d = json.loads(open(f"{R}/{tag}/step_divergence.json").read())
    S, M = d["steps"], d["max_new"]
    dec = defaultdict(lambda: {"n": 0, "eos": 0, "ce": 0.0, "cc": 0.0,
                               "pe": 0.0, "pc": 0.0, "nc": 0})
    first_eos_sf, first_eos_pos, cum = [], [], []
    tot_eos = tot_con = 0
    sweep = []   # (step_frac, rel_pos) for EOS commits, to see right-to-left fill
    for prob in d["problems"]:
        P = prob["prompt_len"]
        fe = None; neos = 0
        per = []
        for r in prob["records"]:
            sf = r["step"] / S
            for k, q in enumerate(r["committed"]):
                t = r["tok_parent"][k]; c = r["conf_parent"][k]; rp = (q - P) / M
                b = min(int(sf * 10), 9); D = dec[b]; D["n"] += 1
                if t in EOS:
                    D["eos"] += 1; D["ce"] += c; D["pe"] += rp
                    tot_eos += 1; neos += 1
                    sweep.append((sf, rp))
                    if fe is None: fe = (sf, rp)
                    per.append(sf)
                else:
                    D["cc"] += c; D["pc"] += rp; D["nc"] += 1
                    tot_con += 1
        if fe: first_eos_sf.append(fe[0]); first_eos_pos.append(fe[1])
        if per:
            per.sort()
            cum.append([per[int(f * (len(per) - 1))] for f in (0.25, 0.5, 0.9)])

    print(f"\n{'='*88}\n{label}\n  driver = {d['parent'].split('/')[-1]}   "
          f"n={d['n_problems']}  budget={M} tok / {S} steps\n{'='*88}")
    print(f"of {tot_eos+tot_con} commits: {100*tot_eos/(tot_eos+tot_con):.1f}% are EOS, "
          f"{100*tot_con/(tot_eos+tot_con):.1f}% are content")
    print(f"\n{'step':>11} {'commits':>8} {'% are EOS':>10} | {'EOS conf':>9} {'EOS pos':>8} "
          f"| {'content conf':>13} {'content pos':>12}")
    for b in range(10):
        D = dec[b]
        if D["n"] == 0: continue
        ec = f"{D['ce']/D['eos']:9.3f}" if D["eos"] else " " * 9
        ep = f"{D['pe']/D['eos']:8.2f}" if D["eos"] else " " * 8
        cc = f"{D['cc']/D['nc']:13.3f}" if D["nc"] else " " * 13
        cp = f"{D['pc']/D['nc']:12.2f}" if D["nc"] else " " * 12
        print(f"  [{b/10:.1f},{(b+1)/10:.1f})".rjust(11) + f"{D['n']:8d}"
              + f"{100*D['eos']/D['n']:9.1f}%" + " |" + ec + ep + " |" + cc + cp)
    if first_eos_sf:
        fs = sorted(first_eos_sf); fp = sorted(first_eos_pos)
        print(f"\nfirst EOS commit: median at step frac {fs[len(fs)//2]:.3f} "
              f"(= step {int(fs[len(fs)//2]*S)} of {S}), at relative position {fp[len(fp)//2]:.2f}")
    if cum:
        q = lambda i: sorted(c[i] for c in cum)[len(cum)//2]
        print(f"by what step frac is the EOS padding laid down (median over problems): "
              f"25% done at {q(0):.3f}, 50% at {q(1):.3f}, 90% at {q(2):.3f}")
    if sweep:
        import statistics as st
        xs = [a for a, _ in sweep]; ys = [b for _, b in sweep]
        mx, my = st.mean(xs), st.mean(ys)
        num = sum((a-mx)*(b-my) for a, b in sweep)
        den = (sum((a-mx)**2 for a in xs) * sum((b-my)**2 for b in ys)) ** 0.5
        print(f"corr(step, position) over EOS commits = {num/den:+.3f}   "
              f"(negative => fills right-to-left: far positions first)")

report("ent-dream-mrg", "PURE DIFFUSION (the real parent)")
report("ent-dream-aq",  "NO-AR RECONSTRUCTION  D_b + 0.4*tau_D")
