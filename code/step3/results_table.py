"""STEP3 results table: one row per arm from outputs/step3/step3-<arm>/{math500/free-last300-merged/summary.json, *.done} plus the fixed
reference rows (Ins alone 36.7, gate Ins <= A(1.5) @0.8 48.3, last 300, H200·0.51). McNemar vs Ins and vs C2 on the shared 300 ids.
Writes a Markdown table; guard columns stay '—' until the official-harness .done files carry a number. Zero GPU."""
import argparse, json, math
from pathlib import Path

REF = {"Ins (C0, untouched)": "step1/gated_ins_add-a10-b20_1024/math500/t08-merged"}   # only used for the Ins ok-set if the free run is missing
INS_OK_FILE = "step0/lockstep1024/math500/parent-free/texts.jsonl"


def mcnemar(a, b):
    n01 = len(b - a); n10 = len(a - b)
    if n01 + n10 == 0:
        return "—"
    k = min(n01, n10); n = n01 + n10
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return f"{n10}/{n01}, p={p:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    L = Path(a.logs)
    rows = {}
    for arm in a.arms:
        f = L / f"step3/step3-{arm}" / "math500" / "free-last300-merged" / "summary.json"
        if f.exists():
            rows[arm] = json.load(open(f))
    ins = L / INS_OK_FILE
    ins_ok = {json.loads(l)["id"] for l in open(ins) if json.loads(l)["ok"]} if ins.exists() else None

    def guard(arm, g):
        f = L / f"step3/step3-{arm}" / f"{g}.done"
        if not f.exists():
            return "—"
        try:
            return json.load(open(f)).get("score", "done")
        except Exception:
            return "done"

    lines = ["# STEP3 results (MATH-500 last 300, lockstep @1024, student alone at 1×; H200·0.51)", "",
             "| arm | acc | finish | acc on finished | vs Ins | vs C2 | GSM8K | HumanEval+ | IFEval |", "|---|---:|---:|---:|---|---|---:|---:|---:|",
             "| Ins (C0) | 36.7 | 76 % | — | — | — | ref | ref | ref |", "| gate Ins ⇐ A(1.5) @0.8 (1.14×, reference ceiling) | 48.3 | 95 % | 51.1 | — | — | — | — | — |"]
    c2 = set(rows["C2"]["ok_ids"]) if "C2" in rows else None
    for arm in a.arms:
        if arm not in rows:
            lines.append(f"| {arm} | not run | | | | | | | |"); continue
        r = rows[arm]; ok = set(r["ok_ids"])
        lines.append(f"| {arm} | {r['acc']:.1f} | {r['finish']:.0f} % | {r['acc_on_finished']:.1f} | {mcnemar(ok, ins_ok) if ins_ok else '—'} | "
                     f"{mcnemar(ok, c2) if (c2 and arm != 'C2') else '—'} | {guard(arm, 'gsm8k')} | {guard(arm, 'humaneval')} | {guard(arm, 'ifeval')} |")
    Path(a.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
