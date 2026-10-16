"""STEP3 prompt pool. DEFAULT (user 2026-10-16, "the most standard recipe"): --mix tulu3-native = Tulu-3 SFT mixture only, sampled uniformly
(= native sub-source proportions: ~36 % math, ~15 % code, ~40 % chat/general, ~12 % safety, 3 % IF), nothing excluded, prompt <= 512 tokens,
one shuffled draw of --n rows -> a run that uses the first N ids (pipeline n_prompts) gets a native-proportion sample of size N.
Kept for the record: --mix mix17k = the 2026-10-15 pool (Tulu-3 stratified 8k + MATH train 6k + Magicoder 2k + no_robots 1k) used by the mixpool-1k test;
it was redundant (Tulu-3 already contains math / code / no_robots) and non-standard (equal-share stratification), see step3_distillation.md.

STEP3 prompt set (plan §1, `technical_details/step3_plan_frontier_kd.md`): label-free prompts for the student rollouts.
Writes outputs/step3/prompts.jsonl, one line per prompt: {id, source, prompt, ref} — `prompt` is the user turn as the student will see it
(MATH gets the official MATH-500 wrapper "Problem:\n…\n\nSolution:"), `ref` the dataset's own response, kept only for the self-SFT control C2.
Mix (2026-10-15, GSM8K train deliberately absent — it stays held-out):
  Tulu-3 8 000 (Ke Wang's 100k sample, read-only; only safety-refusal / hard-coded sources excluded, stratified by source)
  MATH train 6 000 (EleutherAI/hendrycks_math, all 7 subjects)      Magicoder-OSS-Instruct 2 000      no_robots 1 000
Prompts > 512 Qwen tokens (chat-templated) are dropped before sampling. Deterministic (seed 20261015). Zero GPU; run under the login-node
memory cap (`systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0`)."""
import argparse, json, random, collections
from pathlib import Path

I1 = Path(__file__).resolve().parents[2]
TULU = "/mnt/lts4/scratch/home/kewang/sft/DFT/verl/data/tulu3_sft_100k_seed20260919_allassistant_v2/train.parquet"
TULU_EXCLUDE = ("wildjailbreak", "wildguard", "hard_coded")   # substring match on the source name; math-style sources KEPT (user 2026-10-15: standard Tulu-3 -> GSM8K/MATH-500 practice)
MATH_WRAP = "Problem:\n{q}\n\nSolution:"   # = lm_eval math500 arg_0 for Dream-Instruct (samples_math_500_*.jsonl)
QUOTA = {"tulu3": 8000, "math": 6000, "magicoder": 2000, "no_robots": 1000}


def tulu_rows():
    import pyarrow.parquet as pq
    t = pq.read_table(TULU, columns=["source", "source_id", "messages"])
    by_src = collections.defaultdict(list)
    for src, sid, msgs in zip(t.column("source").to_pylist(), t.column("source_id").to_pylist(), t.column("messages").to_pylist()):
        if any(k in src.lower() for k in TULU_EXCLUDE):
            continue
        if not msgs or msgs[0]["role"] != "user":
            continue
        asst = [m["content"] for m in msgs if m["role"] == "assistant"]
        if len(msgs) < 2 or not asst:
            continue
        by_src[src].append(dict(source=f"tulu3:{src.split('/')[-1]}", prompt=msgs[0]["content"], ref=asst[0], sid=str(sid)))
    return by_src


def hf_rows(name, config=None, split="train"):
    from datasets import load_dataset
    return load_dataset(name, config, split=split)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(I1 / "outputs/step3/prompts.jsonl"))
    ap.add_argument("--tokenizer", default="Dream-org/Dream-v0-Instruct-7B")
    ap.add_argument("--max-prompt-tokens", type=int, default=512)
    ap.add_argument("--seed", type=int, default=20261015)
    ap.add_argument("--mix", choices=["mix17k", "tulu3-native"], default="tulu3-native",
                    help="mix17k = Tulu-3 stratified 8k + MATH train 6k + Magicoder 2k + no_robots 1k (mixpool-1k test, 2026-10-15); "
                         "tulu3-native = ONLY Tulu-3 at its native sub-source proportions, nothing excluded (= Dream-Instruct's own SFT distribution; user 2026-10-16: the most standard recipe)")
    ap.add_argument("--n", type=int, default=17000, help="tulu3-native: number of prompts")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer, trust_remote_code=True)

    def fits(p):
        ids = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True)
        return len(ids) <= a.max_prompt_tokens

    if a.mix == "tulu3-native":
        global TULU_EXCLUDE
        TULU_EXCLUDE = ()
        allrows = [r for rows in tulu_rows().values() for r in rows]
        rng.shuffle(allrows)
        rows = []
        for r in allrows:
            if len(rows) >= a.n:
                break
            if fits(r["prompt"]):
                rows.append(r)
        write(rows, a.out); return

    pools = {}
    # Tulu-3: stratified by source (equal share per source, remainder filled from the largest pools)
    by_src = tulu_rows()
    for k in by_src:
        rng.shuffle(by_src[k])
    tulu = []
    per = QUOTA["tulu3"] // len(by_src)
    for k, rows in by_src.items():
        tulu += [r for r in rows[: per * 3] if fits(r["prompt"])][:per]
    rest = [r for k, rows in by_src.items() for r in rows[per * 3:]]
    rng.shuffle(rest)
    for r in rest:
        if len(tulu) >= QUOTA["tulu3"]:
            break
        if fits(r["prompt"]):
            tulu.append(r)
    pools["tulu3"] = tulu
    # MATH train, all subjects
    math = []
    for cfg in ["algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory", "prealgebra", "precalculus"]:
        for i, r in enumerate(hf_rows("EleutherAI/hendrycks_math", cfg)):
            math.append(dict(source=f"math:{cfg}", prompt=MATH_WRAP.format(q=r["problem"]), ref=r["solution"], sid=f"{cfg}/{i}"))
    rng.shuffle(math)
    pools["math"] = [r for r in math if fits(r["prompt"])][: QUOTA["math"]]
    # Magicoder OSS-Instruct
    mc = [dict(source="magicoder", prompt=r["problem"], ref=r["solution"], sid=str(i)) for i, r in enumerate(hf_rows("ise-uiuc/Magicoder-OSS-Instruct-75K"))]
    rng.shuffle(mc)
    pools["magicoder"] = [r for r in mc if fits(r["prompt"])][: QUOTA["magicoder"]]
    # no_robots (train split; single-turn user->assistant only)
    nr = []
    for i, r in enumerate(hf_rows("HuggingFaceH4/no_robots", split="train")):
        m = r["messages"]
        if len(m) >= 2 and m[0]["role"] == "user" and m[1]["role"] == "assistant":
            nr.append(dict(source=f"no_robots:{r.get('category', '')}", prompt=m[0]["content"], ref=m[1]["content"], sid=str(r.get("prompt_id", i))))
    rng.shuffle(nr)
    pools["no_robots"] = [r for r in nr if fits(r["prompt"])][: QUOTA["no_robots"]]

    rows = [r for k in ["tulu3", "math", "magicoder", "no_robots"] for r in pools[k]]
    rng.shuffle(rows)   # shards get a uniform mix (rollout wall time is similar across shards)
    write(rows, a.out)


def write(rows, out):
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for i, r in enumerate(rows):
            f.write(json.dumps(dict(id=i, source=r["source"], sid=r["sid"], prompt=r["prompt"], ref=r["ref"]), ensure_ascii=False) + "\n")
    cnt = collections.Counter(r["source"].split(":")[0] for r in rows)
    print("wrote", out, len(rows), dict(cnt))
    for k, v in sorted(collections.Counter(r["source"] for r in rows).items()):
        print(f"  {v:6d}  {k}")


if __name__ == "__main__":
    main()
