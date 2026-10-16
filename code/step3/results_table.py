"""STEP3 results table: reference rows (the untouched student, the two end-point baselines best merge M(.4,.6) and over addition A(2.0), the
two decode-time gates) + one row per trained arm, all scored on the SAME MATH-500 problem ids (last 300 ∩ what every row has), with
finish rate (\\boxed present), accuracy on finished problems, exact McNemar vs Ins / C2 / M(.4,.6) / A(2.0), and a hardware column.
  python results_table.py --logs outputs --run step3-t1k --arms M C2 --out technical_details/step3_results_t1k.md
Reference runs are the step0/step1 lockstep@1024 outputs (texts.jsonl rows {id, text, ok, gold}). Guard columns are filled from
outputs/step3/<run>-<arm>/{ifeval,humaneval,gsm8k}.done when present. Zero GPU."""
import argparse, json, math
from pathlib import Path

# name -> (path under --logs, hardware)
REFS = {"Ins (student, untouched)": ("step0/lockstep1024/math500/parent-free", "B300"),
        "M(.4,.6) best merge": ("step0/lockstep1024/math500/merge-free", "B300"),
        "A(2.0) over addition (= the teacher)": ("step0/lockstep1024/math500/add20-free", "H200"),
        "gate Ins ⇐ A(2.0) @0.8 (2×)": ("step1/gated_ins_add-a10-b20_1024/math500/t08", "H200"),
        "gate M(.4,.6) ⇐ A(2.0) @0.8 (2×)": ("step1/gated_mrg0406_add-a10-b20_1024/math500/t08", "H200")}
LAST300 = set(range(200, 500))


def load(path):
    rows = {}
    f = Path(path) / "texts.jsonl"
    if f.exists():
        for l in open(f):
            r = json.loads(l); rows[r["id"]] = r
    return rows


def mcnemar(ok_a, ok_b, ids):
    """exact two-sided McNemar: a-wins/b-wins, p — on ids"""
    n10 = sum(1 for i in ids if ok_a[i] and not ok_b[i]); n01 = sum(1 for i in ids if ok_b[i] and not ok_a[i])
    n = n10 + n01
    if n == 0:
        return "0/0"
    p = min(1.0, 2 * sum(math.comb(n, k) for k in range(min(n10, n01) + 1)) / 2 ** n)
    return f"{n10}/{n01}, p={p:.2f}"


def guard(logs, run, arm, g):
    f = Path(logs) / "step3" / f"{run}-{arm}" / f"{g}.done"
    if not f.exists():
        return "—"
    try:
        return f"{json.load(open(f)).get('score', '?')}"
    except Exception:
        return "done"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True, help="idea1 outputs dir")
    ap.add_argument("--run", default="step3-t1k")
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm-hw", default="H200")
    a = ap.parse_args()
    L = Path(a.logs)
    rows = {name: (load(L / p), hw) for name, (p, hw) in REFS.items()}
    for arm in a.arms:
        rows[arm] = (load(L / "step3" / f"{a.run}-{arm}" / "math500" / "free-last300-merged"), a.arm_hw)
    rows = {k: v for k, v in rows.items() if v[0]}
    ids = sorted(LAST300.intersection(*[set(r) for r, _ in rows.values()]))
    ok = {k: {i: bool(r[i]["ok"]) for i in ids} for k, (r, _) in rows.items()}
    fin = {k: {i: ("\\boxed" in r[i]["text"]) for i in ids} for k, (r, _) in rows.items()}
    cmp_to = [c for c in ["Ins (student, untouched)", "C2", "M(.4,.6) best merge", "A(2.0) over addition (= the teacher)"] if c in rows]
    hdr = ["row", "acc", "finish", "acc\\|finished"] + [f"vs {c.split(' ')[0]}" for c in cmp_to] + ["GSM8K", "HumanEval+", "IFEval", "HW"]
    lines = [f"# STEP3 results — run `{a.run}` (MATH-500, {len(ids)} shared problems of the last 300, lockstep @1024, every row decoded by ONE model except the two gates)",
             "", "| " + " | ".join(hdr) + " |", "|" + "|".join(["---"] + ["---:"] * 3 + ["---"] * len(cmp_to) + ["---:"] * 3 + ["---"]) + "|"]
    for k, (r, hw) in rows.items():
        acc = 100 * sum(ok[k].values()) / len(ids); nf = sum(fin[k].values())
        accf = 100 * sum(ok[k][i] for i in ids if fin[k][i]) / max(1, nf)
        cells = [f"**{k}**" if k in a.arms else k, f"{acc:.1f}", f"{100 * nf / len(ids):.0f}%", f"{accf:.1f}"]
        cells += ["—" if c == k else mcnemar(ok[k], ok[c], ids) for c in cmp_to]
        cells += [guard(L, a.run, k, g) if k in a.arms else "—" for g in ["gsm8k", "humaneval", "ifeval"]] + [hw]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "McNemar cells: wins-of-row / wins-of-column, exact two-sided p. Reference rows come from `outputs/step0` (single models) and "
              "`outputs/step1` (gates). Ins and M(.4,.6) were decoded on B300, everything else on H200·0.51 — one hardware column, no cross-pool claim "
              "without an H200 re-run of the B300 rows."]
    Path(a.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
