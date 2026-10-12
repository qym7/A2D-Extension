#!/usr/bin/env python3
"""The complete AR-dose table. Everything that can be computed from artifacts already on disk.

Main block : MATH-500 @512, free mode, SINGLE model, identical code path (lockstep_port run()),
             restricted to the 200 problems ALL seven arms answered.
             Length is the real token count of the post-processed text (trimmed at EOS), so the
             first-EOS POSITION is exactly the answer length.
Traj block : per-step quantities -- these need `--record`, which only alpha=1.5 was run with,
             so every other cell is marked "need run".
"""
import json, os
from pathlib import Path
from math import comb
os.environ.setdefault("HF_HOME", "/mnt/lts4/scratch/home/yqin/DllmAR/dream_study/hf_cache")
from transformers import AutoTokenizer

B = Path("/mnt/lts4/scratch/home/yqin/DllmAR/new_ideas/idea1_selective_distill")
tok = AutoTokenizer.from_pretrained(str(B / "outputs/checkpoints/merge_l04"), trust_remote_code=True)
L = B / "details/logs/lockstep/math500"
CAP = 512

ARMS = [("s0-free-parent", "parent  (tau_A = 0)"), ("e3-mrg04-free", "lambda 0.4"),
        ("s0-free-a2d", "lambda 0.6  <- A2D best"), ("e3-mrg08-free", "lambda 0.8  (over)"),
        ("e3-mrg10-free", "lambda 1.0  (over)"), ("donor-a12-free", "alpha 1.2   (over)"),
        ("donor-a15-free", "alpha 1.5   (over)")]

D = {}
for tag, lab in ARMS:
    p = L / tag / "texts.jsonl"
    if not p.exists(): continue
    per = {}
    for l in p.read_text().splitlines():
        if not l.strip(): continue
        r = json.loads(l); t = r["text"]
        per[r["id"]] = (int(r["ok"]), "\\boxed" in t,
                        len(tok(t, add_special_tokens=False).input_ids))
    D[tag] = per
common = sorted(set.intersection(*[set(v) for v in D.values()]), key=int)

def mcnemar(a, b):
    n01 = n10 = n = 0
    for x, y in zip(a, b):
        if not (x[1] and y[1]): continue
        n += 1
        if x[0] and not y[0]: n10 += 1
        elif y[0] and not x[0]: n01 += 1
    m = n01 + n10
    if m == 0: return n, 0.0, 1.0
    return n, 100 * (n01 - n10) / n, min(1.0, 2 * sum(comb(m, k) for k in range(min(n01, n10) + 1)) / 2 ** m)

V = {t: [D[t][i] for i in common] for t, _ in ARMS if t in D}
ref = V["s0-free-a2d"]

print(f"MATH-500 @512, free mode, single model, same code path, same {len(common)} problems\n")
hdr = (f"| {'arm':<22} | 官方acc | 写完率 | 写完acc | 配对增益vs λ0.6 (p) | "
       f"长度中位 | 写完题长度中位 | 顶cap≥510 | 未收尾=512 |")
print(hdr); print("|" + "|".join(["-" * (len(c)) for c in hdr.split("|")[1:-1]]) + "|")
for tag, lab in ARMS:
    if tag not in V: continue
    v = V[tag]; n = len(v)
    fin = [x for x in v if x[1]]
    ls = sorted(c for _, _, c in v)
    lf = sorted(c for _, _, c in fin)
    nn, dd, pp = mcnemar(ref, v)
    pair = "参考" if tag == "s0-free-a2d" else f"{dd:+.1f} (n={nn}, p={pp:.3f})"
    print(f"| {lab:<22} | {100*sum(a for a,_,_ in v)/n:6.1f} | {100*len(fin)/n:5.1f}% | "
          f"{(100*sum(a for a,_,_ in fin)/len(fin) if fin else float('nan')):6.1f} | {pair:<19} | "
          f"{ls[n//2]:8d} | {(lf[len(lf)//2] if lf else 0):14d} | "
          f"{100*sum(1 for _,_,c in v if c>=510)/n:8.1f}% | {100*sum(1 for _,_,c in v if c>=CAP)/n:9.1f}% |")

# ---- trajectory block
print("\n\nPer-step quantities (need --record; only alpha=1.5 has it)\n")
EOS = {151643, 151645}
REC = L / "eosdiag-a15-rec" / "records"
rows = {}
if REC.exists():
    fe_all, fe_fin, tot, eos_n, sweep = [], [], 0, 0, []
    fin_ids = {i for i, (_, f, _) in D["donor-a15-free"].items() if f}
    for f in sorted(REC.glob("*.json")):
        r = json.loads(f.read_text()); first = None
        for rec in r["steps"]:
            for k, q in enumerate(rec["pos"]):
                tot += 1
                if rec["tok_driver"][k] in EOS:
                    eos_n += 1; sweep.append((rec["step"], q / CAP))
                    if first is None: first = rec["step"]
        if first is not None:
            fe_all.append(first)
            if str(r["id"]) in fin_ids: fe_fin.append(first)
    import statistics as st
    xs=[a for a,_ in sweep]; ys=[b for _,b in sweep]; mx,my=st.mean(xs),st.mean(ys)
    cr=sum((a-mx)*(b-my) for a,b in sweep)/((sum((a-mx)**2 for a in xs)*sum((b-my)**2 for b in ys))**0.5)
    fa=sorted(fe_all); ff=sorted(fe_fin)
    rows["alpha 1.5"] = (f"{fa[len(fa)//2]}", f"{len(fa)}/500",
                         f"{ff[len(ff)//2] if ff else '-'}", f"{100*eos_n/tot:.1f}%", f"{cr:+.3f}")
print(f"| {'arm':<22} | 第一个EOS步(中位) | 有EOS的题 | 写完题的第一个EOS步 | EOS占提交 | corr(步,位置) |")
print("|" + "-"*24 + "|" + "-"*19 + "|" + "-"*11 + "|" + "-"*21 + "|" + "-"*11 + "|" + "-"*15 + "|")
for tag, lab in ARMS:
    key = lab.split()[0] + " " + lab.split()[1]
    if key in rows:
        a,b,c,d,e = rows[key]
        print(f"| {lab:<22} | {a:>17} | {b:>9} | {c:>19} | {d:>9} | {e:>13} |")
    else:
        print(f"| {lab:<22} | {'need run':>17} | {'need run':>9} | {'need run':>19} | {'need run':>9} | {'need run':>13} |")
