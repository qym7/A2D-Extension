#!/usr/bin/env python3
"""Build the MATH validation pool (user 2026-10-17: "找一个有真正 val set 的" — MATH-500 ids 0–199 are not clean, the rand100 test column touches them).

Pool = the MATH (Hendrycks) TEST split, 5 000 problems, MINUS the 500 that are MATH-500 (matched by problem text). MATH-500 is itself a subset
of that test split (PRM800K's held-out 500), so the remaining 4 500 are same-distribution, never reported anywhere by us, and disjoint from the
rollout pool (MATH *train* 6 000). Output = a lm_eval-samples-shaped jsonl the lockstep decoder reads as `--benchmark math500 --prompts FILE`:
  {doc_id, doc{problem, solution, answer, subject, level, unique_id}, arguments{gen_args_0{arg_0: <exact official chat-templated prompt>}}}
gold = remove_boxed(last_boxed_only_string(solution)) with the official minerva utils (verified below against MATH-500's own `answer` field).
doc_id = 0..N-1 in a fixed random order (seed 0), so `--n K` takes a random K of the pool (never the first/last K of the source).
  python build_math_val.py --n 500 --seed 0 --out val_math_test_minus500_rand500_seed0.jsonl
"""
import argparse, json, random, sys
from pathlib import Path
sys.path.insert(0, "/mnt/lts4/scratch/home/yqin/DllmAR/dream_study/dream_official_eval/eval_instruct")
from lm_eval.tasks.minerva_math.utils import last_boxed_only_string, remove_boxed  # noqa: E402
from datasets import load_dataset  # noqa: E402

SUBJECTS = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory", "prealgebra", "precalculus"]
TEMPLATE = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nProblem:\n{problem}\n\nSolution:<|im_end|>\n<|im_start|>assistant\n"


def gold(sol):
    b = last_boxed_only_string(sol)
    return remove_boxed(b) if b is not None else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--official-samples", default=None, help="a MATH-500 lm_eval samples log: checks the prompt template byte-for-byte")
    a = ap.parse_args()
    m500 = load_dataset("HuggingFaceH4/MATH-500", split="test")
    m500_text = {r["problem"].strip(): r["answer"] for r in m500}
    assert len(m500_text) == 500
    pool, hit, gold_ok = [], 0, 0
    for s in SUBJECTS:
        for r in load_dataset("EleutherAI/hendrycks_math", s, split="test"):
            g = gold(r["solution"])
            if r["problem"].strip() in m500_text:
                hit += 1; gold_ok += int(g == m500_text[r["problem"].strip()]); continue
            if g is None:
                continue
            pool.append(dict(problem=r["problem"], solution=r["solution"], answer=g, subject=s, level=r["level"], unique_id=f"{s}/{r['problem'][:40]}"))
    print(f"[pool] test split: MATH-500 matched {hit}/500 (gold extraction agrees on {gold_ok}); remaining pool {len(pool)}")
    assert hit == 500, hit
    rng = random.Random(a.seed); rng.shuffle(pool); pick = pool[: a.n]
    if a.official_samples:
        r = json.loads(open(a.official_samples).readline())
        assert r["arguments"]["gen_args_0"]["arg_0"] == TEMPLATE.format(problem=r["doc"]["problem"]), "template mismatch"
        print("[template] matches the official MATH-500 samples log")
    with open(a.out, "w") as f:
        for i, d in enumerate(pick):
            f.write(json.dumps({"doc_id": i, "doc": d, "arguments": {"gen_args_0": {"arg_0": TEMPLATE.format(problem=d["problem"])}}}, ensure_ascii=False) + "\n")
    import collections
    print(f"[out] {a.out}: {len(pick)} problems; levels {dict(collections.Counter(d['level'] for d in pick))}; subjects {dict(collections.Counter(d['subject'] for d in pick))}")


if __name__ == "__main__":
    main()
