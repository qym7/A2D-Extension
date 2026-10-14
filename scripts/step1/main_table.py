"""Over distillation main table: MATH-500 first 200, @1024 lockstep decoder. Zero GPU.
Columns fixed (user, 2026-10-10): acc, vs student, vs teacher, sub-rate 1 (positions handed to teacher, student p<θ),
sub-rate 2 (teacher token != student token), avg length (tokens of the post-processed answer text, Dream tokenizer),
finish rate (\\boxed present), compute (forward passes per step), hardware. Unfinished arms (<200) are left blank.
Writes technical_details/main_table.md.
"""
import json, os, sys
from math import comb
from pathlib import Path
os.environ.setdefault("HF_HOME", "/mnt/lts4/scratch/home/yqin/DllmAR/dream_study/hf_cache")
from transformers import AutoTokenizer
TOK = AutoTokenizer.from_pretrained("Dream-org/Dream-v0-Instruct-7B", trust_remote_code=True)
I1 = Path(__file__).resolve().parents[2]
L = I1 / "details/logs"
N = 200
def load(p):
    f = L / p / "texts.jsonl"
    return {r["id"]: r for r in map(json.loads, open(f))} if f.exists() else {}
def done(d): return all(i in d for i in range(N))
def mc(a, b):
    w = sum(a[i]["ok"] and not b[i]["ok"] for i in range(N)); l = sum(b[i]["ok"] and not a[i]["ok"] for i in range(N)); m = w + l
    p = min(1, 2 * sum(comb(m, k) for k in range(min(w, l) + 1)) / 2 ** m) if m else 1
    return f"{w}/{l}, {p:.2f}"
def rates(p):
    f = L / p / "gate_per_item.jsonl"
    if f.exists():
        s = list({x["id"]: x for x in map(json.loads, open(f)) if x["id"] < N}.values())
        if len(s) >= N:
            S = lambda k: sum(x[k] for x in s)
            return f"{100*S('n_lowconf')/S('n_committed'):.2f}%", f"{100*S('n_substituted')/S('n_committed'):.2f}%"
    g = L / p / "gate_stats.json"
    if g.exists():
        s = json.load(open(g)); return "—", f"{100*s['substitution_rate']:.2f}%†"
    return "", ""
BASE = {"Ins": "lockstep1024/math500/parent-free", "M(.4,.6)": "lockstep1024/math500/merge-free", "M(.2,.8)": "lockstep1024/math500/k2-l08",
        "M(0,1)": "lockstep1024/math500/l10-free", "A(1.1)": "lockstep1024/math500/add11-free", "A(2.0)": "lockstep1024/math500/add20-free"}
ROLE = {"Ins": "Instruct (τ_D=1, τ_A=0)", "M(.4,.6)": "**best merge**（convex 线，IFEval 选）", "M(.2,.8)": "convex 线，比 best merge 多 AR",
        "M(0,1)": "**over merge**（convex 终点，τ_D=0）", "A(1.1)": "**best addition**（IFEval 选）", "A(2.0)": "**over addition**"}
HW = {"Ins": "B300", "M(.4,.6)": "B300", "M(.2,.8)": "B300", "M(0,1)": "B300", "A(1.1)": "H200·0.51", "A(2.0)": "H200·0.51"}
B = {k: load(p) for k, p in BASE.items()}
GATED = []
for T, d in [("A(1.1)", "gated_ins_add-a10-b11_1024"), ("A(2.0)", "gated_ins_add-a10-b20_1024"), ("M(.4,.6)", "gated_ins_mrg0406_1024")]:
    for t in ["t02", "t04", "t06", "t08"]: GATED.append(("Ins", T, f"0.{t[2]}", f"{d}/math500/{t}", "H200·0.51"))
GATED.append(("A(1.1)", "A(2.0)", "0.2", "gated_add-a10-b11_add-a10-b20_1024/math500/t02", "H200·0.51"))
for t, hw in [("t03", "B300"), ("t06", "H200·0.51"), ("t07", "B300"), ("t08", "H200·0.51")]:
    GATED.append(("M(.4,.6)", "M(0,1)", f"0.{t[2]}", f"gated1024/math500/e1b-mrg10-{t}", hw))
for t in ["t06", "t08"]:
    GATED.append(("M(.4,.6)", "A(2.0)", f"0.{t[2]}", f"gated_mrg0406_add-a10-b20_1024/math500/{t}", "H200·0.51"))
def length(d): return sum(len(TOK(d[i]["text"])["input_ids"]) for i in range(N)) / N
def fin(d): return 100 * sum("\\boxed" in d[i]["text"] for i in range(N)) / N
acc = lambda d: 100 * sum(d[i]["ok"] for i in range(N)) / N
H = ["| 系统 | 计算 | acc | vs student (赢/输, p) | vs teacher (赢/输, p) | 替换率1 | 替换率2 | 平均长度 (tok) | 写完率 | 硬件 |",
     "|---|---:|---:|---|---|---:|---:|---:|---:|---|"]
out = ["# Over distillation — main table (MATH-500 前 200 题, @1024；acc 为 %)\n", "## 1. Baselines（单独跑，同一解码器）\n",
       "| 模型 | 是什么 | 计算 | acc | 平均长度 (tok) | 写完率 | 硬件 |", "|---|---|---:|---:|---:|---:|---|"]
for k, d in B.items():
    out.append(f"| {k} | {ROLE[k]} | 1× | {acc(d):.1f} | {length(d):.0f} | {fin(d):.1f}% | {HW[k]} |")
out.append("| Qwen2.5-7B-Instruct | **AR 源模型**（参考线：lm-eval 贪心 1024 token，AR 解码器，非同一解码器；全 500 题 = 56.0） | 1× | 57.0 | | | H200 |")
out += ["", "## 2. 门控替换 student ⇐ teacher @θ（token 级）\n"] + H
for S, T, th, p, hw in GATED:
    d = load(p); name = f"{S} ⇐ {T} @{th}"
    if not done(d):
        out.append(f"| {name} | 2× | | | | | | | | {hw} |"); continue
    r1, r2 = rates(p)
    out.append(f"| {name} | 2× | {acc(d):.1f} | {mc(d, B[S])} | {mc(d, B[T])} | {r1} | {r2} | {length(d):.0f} | {fin(d):.1f}% | {hw} |")
# compact matrix: student alone (= @0) | θ columns | teacher alone
TH = ["0.2", "0.3", "0.4", "0.6", "0.7", "0.8"]
mat = ["", "## 0. 一张表对比（acc %，空 = 没跑/没跑完）\n", "@0 = student 单独（已验证逐题完全相同）。@1.0 列放 teacher 单独当 baseline；严格说门控 @1.0 = 每个位置都用 teacher 的 token、但哪个位置先写仍由 student 决定，和 teacher 单独不完全相同。\n",
       "| student ⇐ teacher | student 单独 (@0) | " + " | ".join("@" + t for t in TH) + " | @1.0 位置：teacher 单独 |",
       "|---|---:|" + "---:|" * len(TH) + "---:|"]
pairs = []
for S, T, th, p_, hw in GATED:
    if (S, T) not in pairs: pairs.append((S, T))
for S, T in pairs:
    cells = []
    for t in TH:
        hit = [p_ for S2, T2, th, p_, hw in GATED if (S2, T2, th) == (S, T, t)]
        d = load(hit[0]) if hit else {}
        cells.append(f"{acc(d):.1f}" if hit and done(d) else "")
    mat.append(f"| {S} ⇐ {T} | {acc(B[S]):.1f} | " + " | ".join(cells) + f" | {acc(B[T]):.1f} |")
def acc500(d): return f"{100*sum(d[i]['ok'] for i in range(500))/500:.1f}" if all(i in d for i in range(500)) else ""
B500 = {k: load(p_) for k, p_ in BASE.items()}
mat += ["", "### 同一张表，全部 500 题（验证用；空 = 还没有 500 题的结果）\n",
        "| student ⇐ teacher | student 单独 (@0) | " + " | ".join("@" + t for t in TH) + " | @1.0 位置：teacher 单独 |",
        "|---|---:|" + "---:|" * len(TH) + "---:|"]
for S, T in pairs:
    cells = []
    for t in TH:
        hit = [p_ for S2, T2, th, p_, hw in GATED if (S2, T2, th) == (S, T, t)]
        cells.append(acc500(load(hit[0])) if hit else "")
    mat.append(f"| {S} ⇐ {T} | {acc500(B500[S])} | " + " | ".join(cells) + f" | {acc500(B500[T])} |")
mat += ["", "流程（用户，2026-10-10）：先用前 200 题筛选（选 θ / 选 student-teacher 对），再用全部 500 题验证选中的配置。"]
out[1:1] = mat[1:] + [""]
# answer-level vote (fixed: abstentions dropped, ties = expected accuracy of a uniform tie-break)
sys.path.insert(0, "/mnt/lts4/scratch/home/yqin/DllmAR/dream_study/dream_official_eval/eval_instruct/lm_eval/tasks/minerva_math")
import utils as U
def ans(t):
    b = U.last_boxed_only_string(t)
    return U.normalize_final_answer(U.remove_boxed(b) if b is not None else U.get_unnormalized_answer(t))
A = {k: {i: ans(d[i]["text"]) for i in range(N)} for k, d in B.items()}
for k, d in B.items():
    assert all(int(U.is_equiv(A[k][i], d[i]["gold"])) == d[i]["ok"] for i in range(N)), k
def vote(ms):
    tot = 0.0
    for i in range(N):
        gs = []
        for m in ms:
            a = A[m][i]
            if a == "[invalidanswer]": continue
            for g in gs:
                if g[0] == a or U.is_equiv(g[0], a): g[1] += 1; break
            else: gs.append([a, 1, B[m][i]["ok"]])
        if gs:
            top = max(g[1] for g in gs); tied = [g for g in gs if g[1] == top]; tot += sum(g[2] for g in tied) / len(tied)
    return tot
out += ["", "## 3. 不同权重模型的答案投票（答案级；弃权剔除，平票按期望计分）\n",
        "| 成员 | 计算 | acc | 组内最好单模型 | oracle（任一对） |", "|---|---:|---:|---:|---:|"]
for ms in [["M(.4,.6)", "M(0,1)"], ["A(1.1)", "A(2.0)"], ["M(.4,.6)", "M(.2,.8)", "M(0,1)"], ["A(1.1)", "A(2.0)", "M(.4,.6)"],
           ["Ins", "M(.4,.6)", "M(0,1)", "A(1.1)", "A(2.0)"], list(B)]:
    best = max(ms, key=lambda m: acc(B[m]))
    orc = sum(any(B[m][i]["ok"] for m in ms) for i in range(N))
    out.append(f"| {' + '.join(ms)} | {len(ms)}× | {100*vote(ms)/N:.1f} | {acc(B[best]):.1f} ({best}) | {100*orc/N:.1f} |")
# ---- 4. controls + 5. first200 / last300 / all500 split (2026-10-12; sharded runs merged by merge_shards.py) ----
import math as _m
def loadany(p):
    f = L / p / "texts.jsonl"
    return {r["id"]: r for r in map(json.loads, open(f))} if f.exists() else {}
def mcr(a, b, ids):
    w = sum(a[i]["ok"] and not b[i]["ok"] for i in ids); l = sum(b[i]["ok"] and not a[i]["ok"] for i in ids); m = w + l
    pv = min(1, 2 * sum(_m.comb(m, k) for k in range(min(w, l) + 1)) / 2 ** m) if m else 1
    return f"{w}/{l}, {pv:.2f}"
def accr(d, ids): return f"{100*sum(d[i]['ok'] for i in ids)/len(ids):.1f}" if ids and all(i in d for i in ids) else ""
def rate_last300(p):
    f = L / p / "gate_per_item.jsonl"
    if not f.exists(): return "", ""
    s_ = {x["id"]: x for x in map(json.loads, open(f)) if 200 <= x["id"] < 500}
    if len(s_) < 300: return "", ""
    T = lambda k: sum(x[k] for x in s_.values())
    return f"{100*T('n_lowconf')/T('n_committed'):.1f}%", f"{100*T('n_substituted')/T('n_committed'):.2f}%"
R200, R300, R500 = range(200), range(200, 500), range(500)
ROWS = [  # (label, path, student-key or None, teacher-key or None)
 ("Ins 单独", BASE["Ins"], None, None), ("M(.4,.6) 单独", BASE["M(.4,.6)"], None, None), ("M(0,1) 单独", BASE["M(0,1)"], None, None),
 ("A(1.1) 单独", BASE["A(1.1)"], None, None), ("A(2.0) 单独", BASE["A(2.0)"], None, None),
 ("A(3.0) 单独（崩）", "lockstep1024/math500/add30-free-merged", None, None),
 ("(0.7,1.3) 单独 = M(.4,.6)/A(2.0) 权重中点", "lockstep1024/math500/add-a07-b13-free-merged", None, None),
 ("M(.4,.6) ⇐ M(0,1) @0.3 †", "gated1024/math500/e1b-mrg10-t03", "M(.4,.6)", "M(0,1)"),
 ("M(.4,.6) ⇐ M(0,1) @0.7 †", "gated1024/math500/e1b-mrg10-t07", "M(.4,.6)", "M(0,1)"),
 ("Ins ⇐ M(.4,.6) @0.6", "gated_ins_mrg0406_1024/math500/t06", "Ins", "M(.4,.6)"), ("Ins ⇐ M(.4,.6) @0.8", "gated_ins_mrg0406_1024/math500/t08", "Ins", "M(.4,.6)"),
 ("Ins ⇐ A(2.0) @0.6", "gated_ins_add-a10-b20_1024/math500/t06", "Ins", "A(2.0)"), ("Ins ⇐ A(2.0) @0.8", "gated_ins_add-a10-b20_1024/math500/t08", "Ins", "A(2.0)"),
 ("M(.4,.6) ⇐ A(2.0) @0.4", "gated_mrg0406_add-a10-b20_1024/math500/t04-merged", "M(.4,.6)", "A(2.0)"),
 ("M(.4,.6) ⇐ A(2.0) @0.6", "gated_mrg0406_add-a10-b20_1024/math500/t06-merged", "M(.4,.6)", "A(2.0)"),
 ("**M(.4,.6) ⇐ A(2.0) @0.8**", "gated_mrg0406_add-a10-b20_1024/math500/t08", "M(.4,.6)", "A(2.0)"),
 ("M(.4,.6) ⇐ A(2.0) @1.0（teacher 内容 + student 顺序）", "gated_mrg0406_add-a10-b20_1024/math500/t10-merged", "M(.4,.6)", "A(2.0)"),
 ("随机位置替换，剂量匹配 @0.4（分歧位置的 26.2%）", "gated_mrg0406_add-a10-b20_1024/math500/rand0262-merged", "M(.4,.6)", "A(2.0)"),
 ("**随机位置替换**，剂量匹配 @0.8（分歧位置的 85.4%）", "gated_mrg0406_add-a10-b20_1024/math500/rand0854-merged", "M(.4,.6)", "A(2.0)"),
 ("M(.4,.6) ⇐ A(3.0) @0.8", "gated_mrg0406_add-a10-b30_1024/math500/t08-merged", "M(.4,.6)", "A(3.0)"),
]
BX = dict(B); BX["A(3.0)"] = loadany("lockstep1024/math500/add30-free-merged")
out += ["", "## 4. 对照与机制（后 300 题 = id 200–499；配对 = 赢/输, p）\n",
        "| 系统 | 后 300 acc | vs student | vs teacher | vs 门控 @0.8 | 替换率1 | 替换率2 |", "|---|---:|---|---|---|---:|---:|"]
G08 = loadany("gated_mrg0406_add-a10-b20_1024/math500/t08")
for lab, p, S_, T_ in ROWS[5:7] + ROWS[13:]:  # controls block
    d = loadany(p); ok = all(i in d for i in R300)
    if not ok: out.append(f"| {lab} | | | | | | |"); continue
    r1, r2 = rate_last300(p)
    vs_s = mcr(d, BX[S_], R300) if S_ else ""; vs_t = mcr(d, BX[T_], R300) if T_ and all(i in BX[T_] for i in R300) else ""
    vs_g = mcr(d, G08, R300) if p != "gated_mrg0406_add-a10-b20_1024/math500/t08" else "—"
    out.append(f"| {lab} | {accr(d, R300)} | {vs_s} | {vs_t} | {vs_g} | {r1} | {r2} |")
out += ["", "读法：随机位置 ≈ 门控 → 置信度不携带信息（SPEC kill 条件）；@1.0 ≈ teacher 单独 → 解码顺序不是机制；A(3.0) 崩、门控只救回 student → 剂量有最优点。",
        "", "## 5. 前 200 / 后 300 / 全 500 分开看（acc %；配对 = 赢/输, p）\n",
        "| 系统 | 前 200 | 后 300 | 全 500 | 后 300：vs student / vs teacher | 全 500：vs student / vs teacher |", "|---|---:|---:|---:|---|---|"]
for lab, p, S_, T_ in ROWS:
    d = loadany(p)
    c = lambda ids: accr(d, ids)
    pair = lambda ids: (f"{mcr(d, BX[S_], ids)} / {mcr(d, BX[T_], ids)}" if S_ and all(i in d for i in ids) and all(i in BX[T_] for i in ids) else "")
    out.append(f"| {lab} | {c(R200)} | {c(R300)} | {c(R500)} | {pair(R300)} | {pair(R500)} |")
out += ["", "AR 参考线 Qwen2.5-7B-Instruct（lm-eval 贪心 1024）：前 200 = 57.0，后 300 = 55.3，全 500 = 56.0。",
        "† 9/30 B300 旧 run；其余门控 H200·0.51。Ins / M(.4,.6) / M(0,1) 单独 B300，A(·) 单独 H200·0.51。",
        "分片合并：details/draft_scripts/merge_shards.py（*-merged 目录）。",
        "", "注：替换率1 = 交给 teacher 的位置占比（student p<θ）；替换率2 = teacher token ≠ student token 的占比；均按已提交 token。",
        "† 9/30 旧 run 只有全 500 题的汇总（gate_stats.json），替换率1 当时未记录。",
        "平均长度 = 后处理后答案文本的 token 数（截在 EOS）；写完率 = 文本含 \\boxed。",
        "硬件：9/30 的 run 是 B300 整卡，10/09 之后是 H200·0.51；同表比较时注意。",
        "生成脚本：details/draft_scripts/main_table.py"]
(I1 / "technical_details/main_table.md").write_text("\n".join(out) + "\n"); print("\n".join(out))
