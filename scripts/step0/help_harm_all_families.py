"""AR-help / AR-harm of direct addition vs convex merge, relative to the diffusion parent, for every family.

Zero GPU: reads saved per-item results only. File map located 2026-10-09 (see technical_details/figs/help_harm_all_families.md).
D = diffusion parent correct, A = AR parent correct, M = merged model correct.
Writes technical_details/figs/help_harm_all_families.{json,png}.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGS = Path(__file__).resolve().parents[2] / "technical_details" / "figs"
DS = "/mnt/lts4/scratch/home/yqin/DllmAR/dream_study/"
OE = DS + "step2_instruct/outputs/official_eval/"
EP = DS + "dream_coder_official_eval/instruct/code_eval/evalplus/evalplus_official/humaneval/"
CC = OE + "coding_community/mbpp/"
S8 = DS + "step8_rl_vector/outputs/"
M5 = DS + "step7_reasoning_merge/outputs/math500/"
CK = "--lts4--scratch--home--yqin--DllmAR--dream_study--step2_instruct--outputs--checkpoints--"
NO = "/mnt/lts4/scratch/home/abdelrah/ARiFF/TaskVectorDLLM/step5_cross_arch/outputs"


def jl(p):
    return [json.loads(l) for l in open(p)]


def lm_eval(d, key):
    f = sorted(Path(OE + d).glob("*/samples_*.jsonl"))[-1]
    return {r["doc_id"]: int(bool(r[key])) for r in jl(f)}


def evalplus(p, plus=False):
    ev = json.load(open(p))["eval"]
    ok = lambda s: s["base_status"] == "pass" and (not plus or s["plus_status"] == "pass")
    return {t: int(ok(s[0])) for t, s in ev.items()}


def reasoner(p):
    return {r["i"]: int(bool(r["ok"])) for r in json.load(open(p))["per_problem"]}


def qwen3(p):  # sampled avg@4 -> majority correct
    return {i: int(q["frac"] >= 0.5) for i, q in enumerate(json.load(open(p))["results"][0]["per_q"])}


def nemo(d, b, field, key):
    rows = jl(f"{d}/eval/eval-results/{b}/output-rs0.jsonl")
    get = (lambda r: r["strict_eval"]["follow_all_instructions"]) if field == "ifeval" else (lambda r: r[field])
    return {r[key]: int(bool(get(r))) for r in rows}


QC = lambda b: S8 + f"coding_ar/qwencoder-inst-tok2048/{b}/samples-sanitized/eval_results.json"
CELLS = {}
CELLS["Dream-v0 · IFEval"] = dict(
    D=lambda: lm_eval("dream_instruct_full541", "prompt_level_strict_acc"),
    A=lambda: lm_eval("ifeval_qwen_inst_ar", "prompt_level_strict_acc"),
    merge=lambda: lm_eval("merge_0.4_0.6", "prompt_level_strict_acc"),
    add=lambda: lm_eval("ifeval_ins-10aD-11aA", "prompt_level_strict_acc"),
    lab=("0.4/0.6", "β=1.1"))
CELLS["Dream-v0 · MATH-500"] = dict(
    D=lambda: lm_eval("mathins_dreamins_m500", "exact_match"),
    A=lambda: lm_eval("mathins_m500-ar-plain", "exact_match"),
    merge=lambda: lm_eval("mathins_m500-merge-plain", "exact_match"),
    add=lambda: lm_eval("mathins_m500-dream-ins-10aD-11aA-plain", "exact_match"),
    lab=("0.4/0.6", "β=1.1"))
CELLS["Dream-Coder · HumanEval"] = dict(
    D=lambda: evalplus(EP + "Dream-org--Dream-Coder-v0-Instruct-7B_dllm_temp_0.1_len_512.eval_results.json"),
    A=lambda: evalplus(QC("humaneval")),
    merge=lambda: evalplus(EP + "mnt" + CK + "dreamcoderbase_plus_0.8vdreamcoder_0.2vqwencoder_dllm_temp_0.1_len_512.eval_results.json"),
    add=lambda: evalplus(OE + "coding_full/dcoder-10aD-04aA/humaneval_eval_results_20260908T223859.json"),
    lab=("0.8/0.2", "β=0.4"))
CELLS["Dream-Coder · MBPP"] = dict(
    D=lambda: evalplus(CC + "parent_full/mbpp/eval_results.json"),
    A=lambda: evalplus(QC("mbpp")),
    merge=lambda: evalplus(CC + "merge8020_full/mbpp/eval_results.json"),
    add=lambda: evalplus(CC + "custom-dcoder-10aD-04aA_full/mbpp/eval_results.json"),
    lab=("0.8/0.2", "β=0.4"))
CELLS["DiffuCoder-SFT · HumanEval"] = dict(
    D=lambda: evalplus(S8 + "coding/inst-t02/humaneval_T0.2_eval_results.json"),
    A=lambda: evalplus(QC("humaneval")),
    merge=lambda: evalplus(S8 + "coding/cvx-b020-he-t02/humaneval_T0.2_eval_results.json"),
    add=lambda: evalplus(S8 + "coding/cvx-addonly-a060/humaneval_T0.2_eval_results.json"),
    lab=("0.8/0.2", "β=0.6"))
CELLS["DiffuCoder-SFT · MBPP"] = dict(
    D=lambda: evalplus(S8 + "coding/inst-mbpp-t02/mbpp_T0.2_eval_results.json"),
    A=lambda: evalplus(QC("mbpp")),
    merge=lambda: evalplus(S8 + "coding/cvx-b020-mbpp-t02/mbpp_T0.2_eval_results.json"),
    add=lambda: evalplus(S8 + "coding/cvx-addonly-a060/mbpp_T0.2_eval_results.json"),
    lab=("0.8/0.2", "β=0.6"))
CELLS["DiffuCoder-RL · MBPP+"] = dict(
    D=lambda: evalplus(S8 + "coding/grpo-mbpp-t02/mbpp_T0.2_eval_results.json", True),
    A=lambda: evalplus(S8 + "coding_ar/acecoder-rule-tok2048/mbpp/samples-sanitized/eval_results.json", True),
    merge=lambda: evalplus(S8 + "coding/rlrl1-b050-mbpp-t02/mbpp_T0.2_eval_results.json", True),
    add=lambda: evalplus(S8 + "coding/rl-dadd-b04-mbpp-t02/mbpp_T0.2_eval_results.json", True),
    lab=("0.5/0.5", "β=0.4"))
CELLS["DreamReasoner · MATH-500 (8k)"] = dict(
    D=lambda: reasoner(M5 + "parent-m500-8k/results.json"),
    A=lambda: qwen3(M5 + "qwen3-m500-tok8192/results.json"),
    merge=lambda: reasoner(M5 + "rmg-b020-m500-8k/results.json"),
    add=lambda: reasoner(M5 + "rmg-dadd-b05-m500-8k-b300/results.json"),
    lab=("0.8/0.2", "β=0.5"))
_P = {b: f"{NO}/famC_paper_reproduction/evaluations/famc-paper-repro-parent-dlm-{b}" for b in ["math-500", "human-eval", "ifeval"]}
_P["mbpp"] = f"{NO}/famC_paper_extension/evaluations/famc-paper-ext-v1-parent-dlm-mbpp"
_AR = lambda b: f"{NO}/famC_ar_donor_ns_native_sampled_pixi_v1/3b/seed0/sm_103a/{b}"
_T = f"{NO}/famC_tokenizer_protected_sweeps/3b/"
_MG = lambda b: _T + ("evaluations/famc-tokenizer-protected-3b-merge-dlm-ifeval-coef030" if b == "ifeval" else f"downstream_evaluations/famc-tokenizer-protected-3b-merge-coef030-dlm-{b}")
_AD = lambda b: _T + ("evaluations/famc-tokenizer-protected-3b-ar-addition-dlm-ifeval-coef040" if b == "ifeval" else f"downstream_evaluations/famc-tokenizer-protected-3b-ar-addition-coef040-dlm-{b}")
_F = {"math-500": ("symbolic_correct", "unique_id"), "human-eval": ("is_correct", "task_id"),
      "mbpp": ("is_correct", "task_id"), "ifeval": ("ifeval", "key")}
_N = {"math-500": "MATH-500", "human-eval": "HumanEval", "mbpp": "MBPP", "ifeval": "IFEval"}
for b, (fld, key) in _F.items():
    CELLS[f"Nemotron-3B · {_N[b]}"] = dict(
        D=(lambda b=b, fld=fld, key=key: nemo(_P[b], b, fld, key)),
        A=(lambda b=b, fld=fld, key=key: nemo(_AR(b), b, fld, key)),
        merge=(lambda b=b, fld=fld, key=key: nemo(_MG(b), b, fld, key)),
        add=(lambda b=b, fld=fld, key=key: nemo(_AD(b), b, fld, key)),
        lab=("D_p−0.3a_D+0.3a_A", "D_p+0.4a_A"))

out = {}
for cell, spec in CELLS.items():
    X = {k: spec[k]() for k in ["D", "A", "merge", "add"]}
    ids = sorted(set.intersection(*[set(v) for v in X.values()]), key=str)
    D, A = X["D"], X["A"]
    acc = {k: round(100 * sum(v[i] for i in ids) / len(ids), 1) for k, v in X.items()}
    out[cell] = dict(n=len(ids), acc=acc, labels=spec["lab"],
                     help_pool=sum(1 for i in ids if not D[i] and A[i]),
                     harm_pool=sum(1 for i in ids if D[i] and not A[i]))
    for m in ["merge", "add"]:
        M = X[m]
        c = lambda d, a, mm: sum(1 for i in ids if D[i] == d and A[i] == a and M[i] == mm)
        out[cell][m] = dict(help_inh=c(0, 1, 1), synergy=c(0, 0, 1), harm_inh=c(1, 0, 0), own_loss=c(1, 1, 0))
    print(f"{cell:32s} n={len(ids):4d} acc={acc} merge={out[cell]['merge']} add={out[cell]['add']}")
(FIGS / "help_harm_all_families.json").write_text(json.dumps(out, indent=1))

# ---- plot: one row per (cell, mode), % of items, counts printed in segments
INK, INK2, MUTED, GRID, BASE, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SEG = {"own_loss": ("#4a3aa7", "Lost, AR also right (D=1,A=1,M=0)"),
       "harm_inh": ("#eb6834", "AR-harm inherited (D=1,A=0,M=0)"),
       "help_inh": ("#2a78d6", "AR-help inherited (D=0,A=1,M=1)"),
       "synergy": ("#1baf7a", "New solve, AR also wrong (D=0,A=0,M=1)")}
plt.rcParams.update({"font.size": 9, "axes.edgecolor": BASE, "text.color": INK,
                     "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": INK})
cells = list(out)
fig, ax = plt.subplots(figsize=(11, 0.62 * 2 * len(cells) + 1.2))
y, yt, yl = 0, [], []
for cell in cells:
    r = out[cell]
    n = r["n"]
    for mode, name in [("add", "addition"), ("merge", "merge")]:
        s = r[mode]
        left = right = 0.0
        for key in ["harm_inh", "own_loss"]:
            v = 100 * s[key] / n
            ax.barh(y, -v, left=-left, height=0.72, color=SEG[key][0], edgecolor=SURF, linewidth=1.5)
            if s[key] >= 4:
                ax.text(-left - v / 2, y, str(s[key]), ha="center", va="center", color="white", fontsize=7.5, weight="bold")
            left += v
        for key in ["help_inh", "synergy"]:
            v = 100 * s[key] / n
            ax.barh(y, v, left=right, height=0.72, color=SEG[key][0], edgecolor=SURF, linewidth=1.5)
            if s[key] >= 4:
                ax.text(right + v / 2, y, str(s[key]), ha="center", va="center", color="white", fontsize=7.5, weight="bold")
            right += v
        lab = r["labels"][1] if mode == "add" else r["labels"][0]
        net = sum(s[k] for k in ["help_inh", "synergy"]) - sum(s[k] for k in ["harm_inh", "own_loss"])
        ax.text(right + 0.6, y, f"{r['acc'][mode]:.1f} (net {net:+d})", va="center", color=INK2, fontsize=8)
        yt.append(y)
        yl.append(f"{name}  {lab}")
        y += 1
    ax.text(-30.5, y - 2.55, f"{cell}   D={r['acc']['D']:.1f}  AR={r['acc']['A']:.1f}  n={n}  "
            f"help pool {r['help_pool']} · harm pool {r['harm_pool']}", fontsize=8.5, weight="bold", color=INK)
    y += 0.9
ax.set_yticks(yt, yl, fontsize=8)
ax.invert_yaxis()
ax.axvline(0, color=INK2, linewidth=1)
ax.set_xlim(-31, 36)
ax.grid(axis="x", color=GRID, linewidth=0.6)
ax.set_axisbelow(True)
for s in ["top", "right", "left"]:
    ax.spines[s].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.set_xlabel("← % of items lost vs diffusion parent        % of items gained vs diffusion parent →   (numbers in bars = item counts; right label = accuracy)")
handles = [plt.Rectangle((0, 0), 1, 1, color=SEG[k][0]) for k in ["harm_inh", "own_loss", "help_inh", "synergy"]]
fig.legend(handles, [SEG[k][1] for k in ["harm_inh", "own_loss", "help_inh", "synergy"]],
           loc="upper center", ncol=4, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 1.0))
fig.patch.set_facecolor(SURF)
fig.tight_layout(rect=(0, 0, 1, 0.975))
p = FIGS / "help_harm_all_families.png"
fig.savefig(p, dpi=170, bbox_inches="tight", facecolor=SURF)
print(p)
