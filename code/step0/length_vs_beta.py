"""Output length vs beta for addition (and the best merge) in every family. Zero GPU.

Length unit per family: Nemotron = num_generated_tokens; Dream-v0 / DreamReasoner = response characters;
code (EvalPlus) = characters of the sanitized solution (raw generation not in these files -> a proxy).
Writes technical_details/figs/length_vs_beta.json and prints a markdown table.
"""
import glob
import json
import statistics as _st
from pathlib import Path

class st:
    mean = staticmethod(lambda v: None if v is None else _st.mean(v))


FIGS = Path(__file__).resolve().parents[2] / "technical_details" / "figs"
S = json.loads((FIGS / "sweeps_file_map.json").read_text())
OE = "/mnt/lts4/scratch/home/yqin/DllmAR/dream_study/step2_instruct/outputs/official_eval/"


def lengths(loader, p):
    if loader.startswith("evalplus"):
        return [len(v[0]["solution"]) for v in json.load(open(p))["eval"].values()]
    if loader == "reasoner":
        pp = json.load(open(p))["per_problem"]
        return [len(r["resp"]) for r in pp] if "resp" in pp[0] else None  # older runs did not save the response
    if loader.startswith("nemo_"):
        f = glob.glob(p + "/eval/eval-results/*/output-rs0.jsonl")[0]
        return [json.loads(l)["num_generated_tokens"] for l in open(f)]
    raise ValueError(loader)


def lm(d):
    f = sorted(glob.glob(OE + d + "/*/samples_*.jsonl"))[-1]
    out = []
    for l in open(f):
        r = json.loads(l)["resps"][0]
        out.append(len(r[0] if isinstance(r, list) else r))
    return out


rows = {}
DV = {"Dream-v0 · IFEval": ("dream_instruct_full541", "merge_0.4_0.6", 0.6,
                            {b: f"ifeval_ins-10aD-{int(round(b*10)):02d}aA" for b in [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.6]}
                            | {0.2: "ifeval_if1002", 0.4: "ifeval_if1004", 0.6: "ifeval_if1006"}),
      "Dream-v0 · MATH-500": ("mathins_dreamins_m500", "mathins_m500-merge-plain", 0.6,
                              {1.1: "mathins_m500-dream-ins-10aD-11aA-plain"})}
for name, (par, mrg, mb, add) in DV.items():
    rows[name] = dict(unit="chars", parent=st.mean(lm(par)), merge=(mb, st.mean(lm(mrg))),
                      addition=[(b, st.mean(lm(d))) for b, d in sorted(add.items())])
for name, s in S.items():
    if len(s["addition"]) < 2:
        continue
    unit = "tokens" if s["loader"].startswith("nemo_") else ("solution chars" if s["loader"].startswith("evalplus") else "chars")
    best_m = max(s["merge"], key=lambda p: p["acc"]) if s["merge"] else None
    rows[name] = dict(unit=unit, parent=st.mean(lengths(s["loader"], s["parent"])),
                      merge=(best_m["beta"], st.mean(lengths(s["loader"], best_m["path"]))) if best_m else None,
                      addition=[(p["beta"], st.mean(lengths(s["loader"], p["path"]))) for p in sorted(s["addition"], key=lambda p: p["beta"])])
(FIGS / "length_vs_beta.json").write_text(json.dumps(rows, indent=1))
print("| series | unit | parent | best merge (β) | addition β → length (× parent) |\n|---|---|---:|---|---|")
for name, r in rows.items():
    pa = r["parent"]
    f = lambda v: "n/a" if v is None else (f"{v:.0f}" if not pa else f"{v:.0f} (×{v/pa:.2f})")
    m = f"{f(r['merge'][1])} β{r['merge'][0]}" if r["merge"] else "—"
    a = " · ".join(f"{b}: {f(v)}" for b, v in r["addition"])
    print(f"| {name} | {r['unit']} | {'n/a' if pa is None else round(pa)} | {m} | {a} |")
