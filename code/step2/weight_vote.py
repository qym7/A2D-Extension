"""Answer-level majority vote over models with different merge/addition weights (all entropy-order 'free' decoding,
no l2r), MATH-500 first 200 @1024 lockstep. Zero GPU. Fixes the two errors that sank ENSEMBLE_FINDING.md:
  - abstentions ([invalidanswer]) are dropped, not counted as a vote;
  - ties are scored as the expected accuracy of a uniform tie-break (fraction of tied answer groups that are correct).
Answers are grouped with the official minerva is_equiv. Sanity: per-member re-scoring must reproduce every recorded `ok`.
"""
import itertools, json, sys
from pathlib import Path
sys.path.insert(0, "/mnt/lts4/scratch/home/yqin/DllmAR/dream_study/dream_official_eval/eval_instruct/lm_eval/tasks/minerva_math")
import utils as U
L = Path("/mnt/lts4/scratch/home/yqin/DllmAR/new_ideas/idea1_selective_distill/outputs/step0/lockstep1024/math500")
M = {"Ins": "parent-free", "M(.4,.6)": "merge-free", "M(.2,.8)": "k2-l08", "M(0,1)": "l10-free", "A(1.1)": "add11-free", "A(2.0)": "add20-free"}
N = 200
def ans(text):
    b = U.last_boxed_only_string(text)
    return U.normalize_final_answer(U.remove_boxed(b) if b is not None else U.get_unnormalized_answer(text))
R, gold = {}, {}
for k, d in M.items():
    rows = {r["id"]: r for r in map(json.loads, open(L / d / "texts.jsonl")) if r["id"] < N}
    assert len(rows) == N, (k, len(rows))
    R[k] = {}
    for i, r in rows.items():
        a = ans(r["text"]); R[k][i] = (a, int(U.is_equiv(a, r["gold"]))); gold[i] = r["gold"]
        assert R[k][i][1] == r["ok"], (k, i)
print("sanity: re-scoring reproduces all recorded ok fields")
def vote(members):
    tot = 0.0
    for i in range(N):
        groups = []  # [rep, count, correct]
        for m in members:
            a, ok = R[m][i]
            if a == "[invalidanswer]": continue
            for g in groups:
                if g[0] == a or U.is_equiv(g[0], a): g[1] += 1; break
            else: groups.append([a, 1, ok])
        if not groups: continue
        top = max(g[1] for g in groups); tied = [g for g in groups if g[1] == top]
        tot += sum(g[2] for g in tied) / len(tied)
    return tot
single = {m: sum(R[m][i][1] for i in range(N)) for m in M}
oracle = lambda ms: sum(any(R[m][i][1] for m in ms) for i in range(N))
print("\n| members | vote /200 | best member alone | oracle |\n|---|---:|---:|---:|")
pre = [["M(.4,.6)", "M(.2,.8)", "M(0,1)"], ["Ins", "M(.4,.6)", "M(0,1)"], ["A(1.1)", "A(2.0)", "M(.4,.6)"],
       ["Ins", "M(.4,.6)", "M(0,1)", "A(1.1)", "A(2.0)"], list(M)]
for ms in pre:
    print(f"| {' + '.join(ms)} | {vote(ms):.1f} | {max(single[m] for m in ms)} ({max(ms, key=single.get)}) | {oracle(ms)} |")
print("\nall subsets: vote − best member in subset")
for k in [3, 5]:
    d = [vote(ms) - max(single[m] for m in ms) for ms in itertools.combinations(M, k)]
    print(f"  size {k}: {len(d)} subsets, mean {sum(d)/len(d):+.1f}, min {min(d):+.1f}, max {max(d):+.1f}")
print("singles:", single)
