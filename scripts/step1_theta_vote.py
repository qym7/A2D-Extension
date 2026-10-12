"""Within-ONE-system answer vote: same student<=teacher pair, members = different theta (and the two endpoints).
Zero GPU. Same corrected vote rules as weight_vote.py (abstentions dropped, ties scored at expected accuracy).
Also prints an 'agreement' rule: take the answer when members agree, else fall back to a named member.
"""
import json, sys
from pathlib import Path
sys.path.insert(0, "/mnt/lts4/scratch/home/yqin/DllmAR/dream_study/dream_official_eval/eval_instruct/lm_eval/tasks/minerva_math")
import utils as U
L = Path("/mnt/lts4/scratch/home/yqin/DllmAR/new_ideas/idea1_selective_distill/details/logs")
def ans(text):
    b = U.last_boxed_only_string(text)
    return U.normalize_final_answer(U.remove_boxed(b) if b is not None else U.get_unnormalized_answer(text))
def load(d, N):
    rows = {r["id"]: r for r in map(json.loads, open(L / d / "texts.jsonl")) if r["id"] < N}
    assert len(rows) == N, (d, len(rows))
    out = {}
    for i, r in rows.items():
        a = ans(r["text"]); out[i] = (a, int(U.is_equiv(a, r["gold"]))); assert out[i][1] == r["ok"], (d, i)
    return out
def vote(R, members, N):
    tot = 0.0
    for i in range(N):
        groups = []
        for m in members:
            a, ok = R[m][i]
            if a == "[invalidanswer]": continue
            for g in groups:
                if g[0] == a or U.is_equiv(g[0], a): g[1] += 1; break
            else: groups.append([a, 1, ok])
        if not groups: continue
        top = max(g[1] for g in groups); tied = [g for g in groups if g[1] == top]
        tot += sum(g[2] for g in tied) / len(tied)
    return 100 * tot / N
def agree_else(R, a, b, fallback, N):
    tot = 0
    for i in range(N):
        x, okx = R[a][i]; y, oky = R[b][i]
        if x != "[invalidanswer]" and y != "[invalidanswer]" and (x == y or U.is_equiv(x, y)): tot += okx
        else: tot += R[fallback][i][1]
    return 100 * tot / N
SYS = {
 "M(.4,.6)<=A(2.0) (500)": (500, {"M46": "lockstep1024/math500/merge-free", "@0.6": "gated_mrg0406_add-a10-b20_1024/math500/t06-merged",
                                 "@0.8": "gated_mrg0406_add-a10-b20_1024/math500/t08", "A(2.0)": "lockstep1024/math500/add20-free"}),
 "Ins<=A(2.0) (500)": (500, {"Ins": "lockstep1024/math500/parent-free", "@0.6": "gated_ins_add-a10-b20_1024/math500/t06",
                            "@0.8": "gated_ins_add-a10-b20_1024/math500/t08", "A(2.0)": "lockstep1024/math500/add20-free"}),
 "M(.4,.6)<=A(2.0) (200)": (200, {"M46": "lockstep1024/math500/merge-free", "@0.6": "gated_mrg0406_add-a10-b20_1024/math500/t06",
                                 "@0.8": "gated_mrg0406_add-a10-b20_1024/math500/t08", "A(2.0)": "lockstep1024/math500/add20-free"}),
 "Ins<=A(2.0) (200, 4 thetas)": (200, {"Ins": "lockstep1024/math500/parent-free", "@0.2": "gated_ins_add-a10-b20_1024/math500/t02", "@0.4": "gated_ins_add-a10-b20_1024/math500/t04",
                                      "@0.6": "gated_ins_add-a10-b20_1024/math500/t06", "@0.8": "gated_ins_add-a10-b20_1024/math500/t08", "A(2.0)": "lockstep1024/math500/add20-free"}),
}
for name, (N, mem) in SYS.items():
    R = {k: load(d, N) for k, d in mem.items()}
    print(f"\n## {name}   single: " + ", ".join(f"{k} {100*sum(R[k][i][1] for i in range(N))/N:.1f}" for k in R))
    ks = list(R)
    print(f"  vote @0.6+@0.8:           {vote(R, ['@0.6','@0.8'], N):.1f}  (2x)")
    print(f"  vote @0.6+@0.8+A(2.0):    {vote(R, ['@0.6','@0.8','A(2.0)'], N):.1f}  (3x decodes, 2 models)")
    print(f"  vote all {len(ks)}:              {vote(R, ks, N):.1f}")
    inner = [k for k in ks if k.startswith('@')]
    if len(inner) > 2: print(f"  vote thetas only {inner}: {vote(R, inner, N):.1f}")
    print(f"  agree(@0.6,@0.8) else @0.8: {agree_else(R,'@0.6','@0.8','@0.8',N):.1f};  agree(@0.8,A(2.0)) else @0.8: {agree_else(R,'@0.8','A(2.0)','@0.8',N):.1f}")
