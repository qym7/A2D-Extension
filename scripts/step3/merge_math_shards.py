"""Merge the 3 last-300 MATH-500 shards of one STEP3 arm (lockstep @1024, student alone) into <out>/{texts.jsonl,summary.json}.
Generic version of scripts/step1/merge_shards.py (that one has a hard-coded arm map). Zero GPU.
texts.jsonl rows: {id, text, ok, gen_len, gold}; finished = gen_len < 1024 (the decoder's cap)."""
import argparse, json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cap", type=int, default=1024)
    a = ap.parse_args()
    rows = {}
    for s in a.shards:
        f = Path(s) / "texts.jsonl"
        if f.exists():
            for l in open(f):
                r = json.loads(l); rows[r["id"]] = r
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    with open(out / "texts.jsonl", "w") as f:
        for i in sorted(rows):
            f.write(json.dumps(rows[i], ensure_ascii=False) + "\n")
    ids = sorted(rows)
    fin = [i for i in ids if rows[i]["gen_len"] < a.cap]
    summ = dict(n=len(ids), ids=[ids[0], ids[-1]] if ids else None, acc=100 * sum(rows[i]["ok"] for i in ids) / max(1, len(ids)),
                finish=100 * len(fin) / max(1, len(ids)), acc_on_finished=100 * sum(rows[i]["ok"] for i in fin) / max(1, len(fin)),
                ok_ids=[i for i in ids if rows[i]["ok"]])
    json.dump(summ, open(out / "summary.json", "w"), indent=1)
    print(f"{out}: n={summ['n']} acc={summ['acc']:.1f} finish={summ['finish']:.0f}% acc_fin={summ['acc_on_finished']:.1f}")
    if len(ids) != 300:
        raise SystemExit(f"expected 300 ids, got {len(ids)}")


if __name__ == "__main__":
    main()
