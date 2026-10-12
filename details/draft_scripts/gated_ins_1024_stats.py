import json, statistics as st
L = "/mnt/lts4/scratch/home/yqin/DllmAR/new_ideas/idea1_selective_distill/details/logs/"
def tx(p): return {r["id"]: r for r in map(json.loads, open(L + p + "/texts.jsonl"))}
I = tx("lockstep1024/math500/parent-free"); M = tx("lockstep1024/math500/merge-free"); A = tx("lockstep1024/math500/add11-free")
ok = lambda d, i: d[i]["ok"]
R = range(200)
hp = [i for i in R if not ok(I, i) and ok(A, i)]   # AR-help pool: Ins wrong, add1.1 right
xp = [i for i in R if ok(I, i) and not ok(A, i)]   # harm pool: Ins right, add1.1 wrong
print(f"pools vs Instruct: add1.1 fixes {len(hp)}, add1.1 breaks {len(xp)}; both right {sum(ok(I,i) and ok(A,i) for i in R)}, both wrong {sum(not ok(I,i) and not ok(A,i) for i in R)}")
print(f"gen_len mean: Ins {st.mean(I[i]['gen_len'] for i in R):.0f}, merge {st.mean(M[i]['gen_len'] for i in R):.0f}, add1.1 {st.mean(A[i]['gen_len'] for i in R):.0f}")
ARMS = {f"Ins→{n} θ={t[1]}.{t[2]}": f"{d}/math500/{t}" for n, d in [("add1.1", "gated_ins_add-a10-b11_1024"), ("add2.0", "gated_ins_add-a10-b20_1024"), ("merge", "gated_ins_mrg0406_1024")] for t in ["t02", "t04"]}
rows = []
for k, p in ARMS.items():
    G = tx(p); g = {r["id"]: r for r in map(json.loads, open(L + p + "/gate_per_item.jsonl"))}
    s = [g[i]["n_steps_with_sub"] for i in R]
    used = [i for i in R if s[i] > 0]; none = [i for i in R if s[i] == 0]
    flip_used = sum(ok(G, i) != ok(I, i) for i in used); flip_none = sum(ok(G, i) != ok(I, i) for i in none)
    q = sorted(s)
    rows.append(dict(arm=k, acc=sum(ok(G, i) for i in R),
        help=f"{sum(ok(G,i) for i in hp)}/{len(hp)}", harm=f"{sum(not ok(G,i) for i in xp)}/{len(xp)}",
        both_right_lost=sum(ok(I,i) and ok(A,i) and not ok(G,i) for i in R), both_wrong_won=sum(not ok(I,i) and not ok(A,i) and ok(G,i) for i in R),
        steps=f"med {q[100]}, p90 {q[180]}, max {q[-1]}", used=len(used),
        flip=f"{flip_used}/{len(used)} vs {flip_none}/{len(none)}",
        acc_used=f"{sum(ok(G,i) for i in used)}/{len(used)} (Ins {sum(ok(I,i) for i in used)})",
        lowconf_steps=f"{st.mean(g[i]['n_steps_with_lowconf'] for i in R):.1f}",
        gen_len=f"{st.mean(G[i]['gen_len'] for i in R):.0f}"))
cols = ["arm", "acc", "help", "harm", "both_right_lost", "both_wrong_won", "used", "steps", "acc_used", "flip", "lowconf_steps", "gen_len"]
print("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols))
for r in rows: print("| " + " | ".join(str(r[c]) for c in cols) + " |")
