"""STEP3 preliminary study (2026-10-14, user): does the training-time "low-confidence position" match the decode-time one,
and how often do several teachers agree there?

At decode time the gate looks at ONE position per step (the student's most confident mask, 97-99 % the leftmost one) and hands it
to the teacher when the student's raw p < theta. At training time a masked-diffusion loss corrupts the whole response at once and
scores every masked position, so "p < theta" is evaluated under a different state (many masks, no left-to-right prefix). This script
measures the gap on the 50 recorded problems (ids 200-249, Ins <= A(2.0) @0.8, details/logs/arness50/math500/ins-to-a20-t08/records):

  corruption S(t)  "suffix"  = the exact decode-time state at step t (positions committed before t visible, the rest masked);
  corruption R(r)  "random"  = standard SFT corruption: each response position masked i.i.d. with prob r.

For every corrupted input and every model m in {Ins (student), A(1.5), A(2.0), A(2.5), M(0,1)} we record, at every masked position,
the raw softmax argmax and its probability. Metrics (all per corruption type / parameter, pooled over problems):
  1. share of masked positions with student p < theta ("training low-conf set" L_train);
  2. recall / precision of L_train against the decode-time low-conf set L_dec (positions committed at a step with p_driver_raw < theta)
     restricted to positions that are masked in this corruption; same for the decode-time SUBSTITUTED set (teacher token != student token);
  3. at L_train positions: agreement among the three A(beta) teachers, between A(2.0) and M(0,1), among all four; and whether the
     agreed token equals the final text token (what the gate actually wrote) and the student's own argmax;
  4. the student's confidence quantiles at masked positions (to decide whether theta must depend on the mask ratio);
  5. determinism check: for S(t), the student's p at the leftmost mask must reproduce the recorded p_driver_raw.
Writes <out>/signal_check.json and prints a summary table. One GPU, models loaded one at a time (5 x 15 GB does not fit at 0.51)."""
import argparse, json, glob, os, random
from pathlib import Path
import torch

MASK = None


def shifted_logits(model, x):
    logits = model(x, "full", None).logits
    return torch.cat([logits[:, :1], logits[:, :-1]], dim=1)


@torch.no_grad()
def predict(model, x, mask_index):
    """raw (T=1, no top-p) argmax and its probability at the masked positions of x (1 x L)."""
    lg = shifted_logits(model, x)[mask_index].float()
    p = torch.softmax(lg, dim=-1)
    conf, tok = p.max(dim=-1)
    return tok.cpu(), conf.cpu()


def build_corruptions(rec, theta, n_suffix, ratios, rng):
    """Returns a list of (name, param, x_ids(list[int]), masked_positions(sorted list[int])).
    rec["final_ids"] holds the GENERATED tokens only (lockstep_port writes gen = x[0, P:]); rec["prompt_ids"] is attached by main()."""
    P = rec["prompt_len"]; ids = list(rec["prompt_ids"]) + list(rec["final_ids"]); L = len(ids)
    assert len(rec["prompt_ids"]) == P, (len(rec["prompt_ids"]), P)
    # response region for random masking: prompt_len .. first <|im_end|> (151645) inclusive, else the whole canvas
    ends = [ids.index(e, P) + 1 for e in (151645, 151643) if e in ids[P:]]   # first <|im_end|> or <|endoftext|> after the prompt
    end = min(ends) if ends else L
    order = [s["pos"][0] for s in rec["steps"]]          # one committed position per step, in decode order
    low_steps = [k for k, s in enumerate(rec["steps"]) if s["p_driver_raw"][0] < theta]
    outs = []
    # suffix: decode-time state at n_suffix steps: half of them low-confidence steps, half uniform over the trajectory
    picks = set()
    if low_steps:
        picks.update(rng.sample(low_steps, min(len(low_steps), n_suffix // 2)))
    while len(picks) < n_suffix:
        picks.add(rng.randrange(len(order)))
    for t in sorted(picks):
        x = list(ids); masked = set(order[t:])          # everything not yet committed before step t
        for q in masked: x[q] = MASK
        outs.append(("suffix", t, x, sorted(masked)))
    for r in ratios:
        x = list(ids); masked = [q for q in range(P, end) if rng.random() < r]
        for q in masked: x[q] = MASK
        if masked: outs.append(("random", r, x, masked))
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--prompts", required=True, help="the math500 samples jsonl used by the decoder (prompt text = arguments.gen_args_0.arg_0)")
    ap.add_argument("--student", default="Dream-org/Dream-v0-Instruct-7B")
    ap.add_argument("--teachers", required=True, help="name=path,name=path,...")
    ap.add_argument("--theta", type=float, default=0.8)
    ap.add_argument("--n-suffix", type=int, default=8)
    ap.add_argument("--ratios", default="0.1,0.3,0.5,0.7,0.9")
    ap.add_argument("--n-problems", type=int, default=50)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    global MASK
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.student, trust_remote_code=True)
    MASK = tok.convert_tokens_to_ids("<|mask|>")
    ratios = [float(r) for r in a.ratios.split(",")]
    rng = random.Random(a.seed)
    ptext = {}
    for line in open(a.prompts):
        if line.strip():
            d = json.loads(line); ptext[d["doc_id"]] = d["arguments"]["gen_args_0"]["arg_0"]
    recs = []
    for f in sorted(glob.glob(os.path.join(a.records, "*.json")))[: a.n_problems]:
        rec = json.load(open(f)); rec["prompt_ids"] = tok(ptext[rec["id"]]).input_ids; recs.append(rec)
    # corruptions are built once (same inputs for every model)
    corr = {rec["id"]: build_corruptions(rec, a.theta, a.n_suffix, ratios, rng) for rec in recs}
    models = [("student", a.student)] + [tuple(s.split("=", 1)) for s in a.teachers.split(",")]
    preds = {}   # (model, id, k) -> (tok, conf) tensors over the masked positions of corruption k
    for name, path in models:
        print(f"[load] {name} <- {path}", flush=True)
        m = AutoModel.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda().eval()
        for rec in recs:
            for k, (cname, par, x, masked) in enumerate(corr[rec["id"]]):
                xt = torch.tensor([x], device="cuda")
                preds[(name, rec["id"], k)] = predict(m, xt, xt == MASK)
        del m; torch.cuda.empty_cache()
        print(f"[done] {name}", flush=True)
    tnames = [n for n, _ in models[1:]]
    # ---- metrics
    agg = {}
    def bucket(cname, par):
        key = f"{cname}:{par}"
        return agg.setdefault(key, dict(n_masked=0, n_low=0, n_dec_low_masked=0, n_dec_low_hit=0, n_dec_sub_masked=0, n_dec_sub_hit=0,
                                        n_low_agree_A=0, n_low_agree_A20M01=0, n_low_agree_all=0, n_low_agree_A_eq_final=0,
                                        n_low_agree_A_eq_student=0, n_low_student_eq_final=0, n_low_A20_eq_final=0,
                                        conf=[], n_leftmost=0, n_leftmost_match=0))
    for rec in recs:
        order = [s["pos"][0] for s in rec["steps"]]
        dec_low = {s["pos"][0] for s in rec["steps"] if s["p_driver_raw"][0] < a.theta}
        dec_sub = {s["pos"][0] for s in rec["steps"] if s["p_driver_raw"][0] < a.theta and s["tok_driver"][0] != s["tok_other"][0]}
        p_at_step = {k: s["p_driver_raw"][0] for k, s in enumerate(rec["steps"])}
        for k, (cname, par, x, masked) in enumerate(corr[rec["id"]]):
            st, sc = preds[("student", rec["id"], k)]
            # suffix states are scored in frontier buckets: the first 1 / 4 / 16 masked positions and all of them (2026-10-14 rerun:
            # pooled over all masks the far-right positions dominate, which the gate never looks at)
            buckets = [bucket(cname, par)] if cname == "random" else [bucket("suffix", f"first{K}") for K in (1, 4, 16, 10**6)]
            for b, K in zip(buckets, ([None] if cname == "random" else [1, 4, 16, 10**6])):
              masked_b = masked if K is None else masked[:K]
              if cname == "suffix" and K == 10**6: masked_b = masked
              tt = {n: preds[(n, rec["id"], k)][0] for n in tnames}
              pos2row = {q: r for r, q in enumerate(masked)}
              b["n_masked"] += len(masked_b)
              b["conf"].extend(round(float(sc[pos2row[q]]), 3) for q in masked_b[:: max(1, len(masked_b) // 64)])
              if cname == "suffix" and order[par] == masked[0]:   # determinism: only where the decoder committed the leftmost mask
                  left = masked[0]; b["n_leftmost"] += 1
                  b["n_leftmost_match"] += abs(float(sc[pos2row[left]]) - p_at_step[par]) < 0.02
              for q in masked_b:
                  r = pos2row[q]; low = float(sc[r]) < a.theta
                  if q in dec_low:
                      b["n_dec_low_masked"] += 1; b["n_dec_low_hit"] += low
                  if q in dec_sub:
                      b["n_dec_sub_masked"] += 1; b["n_dec_sub_hit"] += low
                  if not low: continue
                  b["n_low"] += 1
                  final = (list(rec["prompt_ids"]) + list(rec["final_ids"]))[q]; s_tok = int(st[r])
                  A = [int(tt[n][r]) for n in tnames if n.startswith("A")]
                  a20 = int(tt["A2.0"][r]) if "A2.0" in tt else None
                  m01 = int(tt["M01"][r]) if "M01" in tt else None
                  agA = len(set(A)) == 1 if A else False
                  b["n_low_agree_A"] += agA
                  b["n_low_agree_A20M01"] += (a20 is not None and m01 is not None and a20 == m01)
                  b["n_low_agree_all"] += (agA and m01 is not None and A[0] == m01)
                  b["n_low_agree_A_eq_final"] += (agA and A[0] == final)
                  b["n_low_agree_A_eq_student"] += (agA and A[0] == s_tok)
                  b["n_low_student_eq_final"] += (s_tok == final)
                  b["n_low_A20_eq_final"] += (a20 == final)
    # ---- summary
    rows = []
    for key, b in agg.items():
        q = sorted(b["conf"]); qs = [q[int(len(q) * f)] for f in (0.1, 0.25, 0.5, 0.75)] if q else []
        n_low = max(1, b["n_low"])
        rows.append(dict(corruption=key, masked=b["n_masked"], low_share=b["n_low"] / max(1, b["n_masked"]),
                         dec_low_recall=b["n_dec_low_hit"] / max(1, b["n_dec_low_masked"]), dec_sub_recall=b["n_dec_sub_hit"] / max(1, b["n_dec_sub_masked"]),
                         low_precision_vs_dec_low=b["n_dec_low_hit"] / n_low,
                         agree_A=b["n_low_agree_A"] / n_low, agree_A20_M01=b["n_low_agree_A20M01"] / n_low, agree_all=b["n_low_agree_all"] / n_low,
                         agreeA_eq_final=b["n_low_agree_A_eq_final"] / max(1, b["n_low_agree_A"]), agreeA_eq_student=b["n_low_agree_A_eq_student"] / max(1, b["n_low_agree_A"]),
                         student_eq_final=b["n_low_student_eq_final"] / n_low, A20_eq_final=b["n_low_A20_eq_final"] / n_low,
                         conf_q10_25_50_75=qs, leftmost_det=f'{b["n_leftmost_match"]}/{b["n_leftmost"]}'))
    Path(a.out).mkdir(parents=True, exist_ok=True)
    json.dump(dict(args=vars(a), rows=rows), open(os.path.join(a.out, "signal_check.json"), "w"), indent=1)
    hdr = ["corruption", "masked", "low_share", "dec_low_recall", "dec_sub_recall", "low_precision_vs_dec_low", "agree_A", "agree_A20_M01", "agree_all", "agreeA_eq_final", "agreeA_eq_student", "student_eq_final", "A20_eq_final", "conf_q10_25_50_75", "leftmost_det"]
    print("| " + " | ".join(hdr) + " |"); print("|" + "---|" * len(hdr))
    for r in rows:
        print("| " + " | ".join((f"{r[h]:.3f}" if isinstance(r[h], float) else str(r[h])) for h in hdr) + " |")


if __name__ == "__main__":
    main()
