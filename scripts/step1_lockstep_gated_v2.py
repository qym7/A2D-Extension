#!/usr/bin/env python3
"""Confidence-gated content-port: idea 1's E1/S2 instrument.

Wraps ``dream_study/step2_instruct/scripts/lockstep_port.py`` instead of copying it, so the
prompt loading, post-processing, scorer, resume logic and the official Dream ``_sample``
protocol are byte-identical to the script that produced A2D Table 2. The ONLY new logic is the
decode loop's gate, below.

Gate
----
At each denoising step the driver (host) proposes (position, token). We gate on the driver's
**RAW** probability for that token -- ``model_step``'s 5th return value ``praw``, the un-sharpened
belief (softmax at T=1, no top-p) -- NOT the protocol-sharpened ``pmax``.

    This is load-bearing. ``iclr/overleaf/plots/lockstep/README.md`` records that the sharpened
    confidence "saturates at 1.0 everywhere" and "cannot rank tokens by confidence"; that is exactly
    why ``praw`` was added to the instrument on 2026-09-21. A gate on the sharpened value would put
    almost every position above any threshold below 1.0, so the sweep would be degenerate: identical
    to free mode for every theta < 1 and identical to content mode at theta >= 1. An earlier draft of
    this file made that mistake.

For ``--mode content_gated`` we take the OTHER model's (donor's) token at a committed position only
when the driver's raw ``p < theta``; otherwise we keep the driver's. So:

    theta = 0    -> never substitute            == lockstep_port.py --mode free   (plumbing check)
    theta >= 1   -> always substitute where they differ == --mode content         (plumbing check)

Both endpoints are asserted-by-construction reproductions of already-published numbers, which is
why the sweep contains them.

``--mode schedule_gated`` gates at the STEP level instead: if the driver's mean confidence over the
positions it was about to commit is below theta, that step uses the other model's position ranking.
Position choice is not a per-position decision, so a per-position gate is not well defined there.

Everything the gate needs (``praw``, the driver's raw committed-token probabilities) is already
computed by ``lockstep_port.model_step``; no extra forward pass.

Side output: ``gate_stats.json`` in --out-dir, with the substitution rate and the positional
distribution of substitutions (idea 1 ANALYSIS.md asks whether substitutions cluster late).
"""
from __future__ import annotations
# v2 (2026-10-13): + mode content_mean (mean of several teachers' raw distributions at the student's low-confidence steps;
# --with-student adds the student to the mean = soft mixing). Every other mode is byte-identical to step1_lockstep_gated.py.

import json
import math
import random
import sys
from pathlib import Path

import torch

SRC = Path("/mnt/lts4/scratch/home/yqin/DllmAR/dream_study/step2_instruct/scripts")
sys.path.insert(0, str(SRC))
import lockstep_port as LP  # noqa: E402  (path must be set first)

def model_step_ext(model, x, mask_index, top_p, conf_temp, ref_tokens=None):
    """Byte-identical to ``LP.model_step`` (verified by ``test_objection_gate.py``), but also returns
    the RAW (T=1, no top-p) probability this model assigns to ``ref_tokens`` -- used to ask how much
    the donor objects to the HOST's token, which ``model_step`` cannot tell us because it only ever
    gathers the model's own argmax."""
    logits = model(x, "full", None).logits
    logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
    ml = logits[mask_index]
    raw = torch.softmax(ml.float(), dim=-1)
    if conf_temp and conf_temp > 0:
        ml = ml / conf_temp
    if top_p is not None and top_p < 1:
        ml = LP.top_p_logits(ml, top_p)
    probs = torch.softmax(ml, dim=-1)
    pmax, x0 = probs.max(dim=-1)
    negent = torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
    praw = raw.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
    p_ref = None
    if ref_tokens is not None:
        p_ref = raw.gather(-1, ref_tokens.unsqueeze(-1)).squeeze(-1)
    return x0, negent, pmax, logits.dtype, praw, p_ref


def model_step_raw(model, x, mask_index, top_p, conf_temp):
    """model_step_ext without ref_tokens, plus the full RAW (T=1, no top-p) distribution over the masked positions."""
    logits = model(x, "full", None).logits
    logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
    ml = logits[mask_index]
    raw = torch.softmax(ml.float(), dim=-1)
    if conf_temp and conf_temp > 0:
        ml = ml / conf_temp
    if top_p is not None and top_p < 1:
        ml = LP.top_p_logits(ml, top_p)
    probs = torch.softmax(ml, dim=-1)
    pmax, x0 = probs.max(dim=-1)
    negent = torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
    praw = raw.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
    return x0, negent, pmax, logits.dtype, praw, raw


# --- module state the wrapper injects -------------------------------------------------------
THETA: float | None = None
MARGIN: float = 0.0                    # content_relgated: donor must beat host by this much
TAU: float | None = None               # content_objection: donor's log-odds preference threshold
RAND_FRAC: float | None = None         # content_random: P(substitute | the two models differ)
SEED: int = 0
EXTRA_OTHERS: list = []                # content_mean: teacher paths beyond --other (loaded lazily on the GPU)
EXTRA_MODELS: list | None = None
WITH_STUDENT: bool = False             # content_mean: include the student's distribution in the mean (soft mixing)
GATED_MODE: str | None = None          # "content_gated" | "content_relgated" | "schedule_gated"
STATS: dict = {"n_committed": 0, "n_differ": 0, "n_substituted": 0,
               "n_steps": 0, "n_steps_sched_ported": 0, "sub_positions": [],
               "raw_conf_hist": [0] * 20,      # driver raw p(argmax) over committed positions
               # --- diagnostics for "is the donor actually better where the host is unsure?"
               # accumulated in EVERY content mode, so any run answers the question. Restricted to
               # committed positions where the two models' argmax DIFFERS (the only ones that matter).
               "joint_hist_differ": [[0] * 20 for _ in range(20)],   # [driver bin][donor bin]
               "n_differ_donor_more_conf": 0,
               # content_objection diagnostics: donor's log p(own token) - log p(host's token),
               # accumulated over DIFFERING committed positions. Bins are [0,0.5),[0.5,1),...,[9.5,inf).
               "objection_hist": [0] * 20,
               "n_differ_scored": 0,
               "n_lowconf_differ": 0,          # driver raw p < theta AND differs
               "n_lowconf_differ_donor_more": 0,
               "n_steps_with_sub": 0,
               "n_mean_lowconf": 0, "n_vote_differs": 0, "n_mean_eq_last_teacher": 0, "n_mean_eq_student": 0}   # content_mean diagnostics          # steps where >= 1 committed token came from the donor
START: int = 0                         # --start S: skip math500 ids < S (2026-10-12, user: run only the last 300); decode is argmax-deterministic, so ids >= S are identical to a full run
OUTDIR: Path | None = None             # per-problem gate counts go to OUTDIR/gate_per_item.jsonl (2026-10-09)


@torch.no_grad()
def run_gated(driver, other, x, mask_id, *, mode, steps, eps, top_p, conf_temp, record, rng):
    """Copy of LP.run with a confidence gate. Kept line-aligned with the original on purpose."""
    if GATED_MODE is None:
        return LP._run_orig(driver, other, x, mask_id, mode=mode, steps=steps, eps=eps,
                            top_p=top_p, conf_temp=conf_temp, record=record, rng=rng)
    theta = 0.0 if THETA is None else float(THETA)
    global RND
    # BUG FIXED 2026-09-29. This used to be random.Random(SEED), re-seeded identically at every call,
    # i.e. once per problem -- so every problem drew the SAME number sequence and the "random" control
    # substituted the same positional pattern everywhere. It also biased the realised rate far below
    # rand_frac when few positions are eligible per problem (measured 0.103 at frac=0.27; a 10-draw
    # simulation with Random(0) reproduces exactly 0.100). Seeding from the input canvas makes the
    # stream problem-dependent AND stable across resumes (x holds the prompt ids at this point, and
    # resume re-enters with the same canvas for a given problem).
    RND = random.Random((SEED * 1000003) ^ int(x.sum().item()))
    dev = x.device
    B = x.size(0)
    assert B == 1
    timesteps = torch.linspace(1, eps, steps + 1, device=dev)
    recs = []
    # per-problem counters (2026-10-09, user: record how many steps used the donor). math500 ids are 0..N-1 and
    # LP.main runs them in order, so this problem's id = smallest id not yet in texts.jsonl.
    item = dict(n_steps=0, n_steps_with_sub=0, n_steps_with_lowconf=0, n_committed=0, n_lowconf=0,
                n_differ=0, n_substituted=0)
    for i in range(steps):
        mask_index = (x == mask_id)
        if mask_index.sum() == 0:
            break
        global EXTRA_MODELS
        if GATED_MODE == "content_mean" and EXTRA_OTHERS and EXTRA_MODELS is None:
            from transformers import AutoModel
            EXTRA_MODELS = [AutoModel.from_pretrained(p, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda().eval() for p in EXTRA_OTHERS]
        rawd = rawo = None; raws_extra = []; x0_extra = []
        if GATED_MODE == "content_mean":
            x0d, ned, pd, dt, prd, rawd = model_step_raw(driver, x, mask_index, top_p, conf_temp)
            x0o, neo, po, _, pro, rawo = model_step_raw(other, x, mask_index, top_p, conf_temp)
            for m_ in (EXTRA_MODELS or []):
                x0e, _, _, _, _, rawe = model_step_raw(m_, x, mask_index, top_p, conf_temp)
                x0_extra.append(x0e); raws_extra.append(rawe)
            po_at_host = None
        else:
            x0d, ned, pd, dt, prd = LP.model_step(driver, x, mask_index, top_p, conf_temp)
        if GATED_MODE == "content_mean":
            pass
        elif GATED_MODE == "content_objection":
            x0o, neo, po, _, pro, po_at_host = model_step_ext(other, x, mask_index, top_p,
                                                              conf_temp, ref_tokens=x0d)
        else:
            x0o, neo, po, _, pro = LP.model_step(other, x, mask_index, top_p, conf_temp)
            po_at_host = None
        t, s = timesteps[i], timesteps[i + 1]
        num_mask = mask_index.sum() / mask_index.shape[0]
        k = int(num_mask * (1 - s / t)) if i < steps - 1 else int(num_mask)
        if k <= 0:
            continue
        mpos = mask_index[0].nonzero(as_tuple=True)[0]
        pos2row = {int(p): r for r, p in enumerate(mpos.tolist())}
        STATS["n_steps"] += 1
        item["n_steps"] += 1
        sub_here = low_here = False

        # --- which positions
        commit = LP.topk_positions(x, mask_index, ned, k, dt)
        ported = False
        if GATED_MODE in ("schedule_gated", "both_gated"):
            conf_sel = torch.tensor([float(prd[pos2row[int(j)]]) for j in commit[0].tolist()])
            if float(conf_sel.mean()) < theta:
                commit = LP.topk_positions(x, mask_index, neo, k, dt)
                STATS["n_steps_sched_ported"] += 1
                ported = True
        # both_gated (2026-10-12, user): at a low-confidence step hand BOTH the position (teacher's top-k by its own
        # confidence) and the token at that position to the teacher; otherwise the student's position and token.
        # Fires on exactly the same steps as content_gated (k = 1 here, so conf_sel.mean() == the gated p).

        # --- which tokens
        xa = torch.zeros_like(x) + mask_id
        xa[mask_index] = x0d
        if GATED_MODE in ("content_gated", "content_relgated", "content_objection", "content_random", "content_random_uncond", "both_gated", "content_mean"):
            xo = torch.zeros_like(x) + mask_id
            xo[mask_index] = x0o
            for j in commit[0].tolist():
                j = int(j)
                r = pos2row[j]
                STATS["n_committed"] += 1
                item["n_committed"] += 1
                cd, co = float(prd[r]), float(pro[r])   # RAW confidences, not the sharpened pd/po
                b = min(19, int(cd * 20))
                STATS["raw_conf_hist"][b] += 1
                differ = int(x0d[r]) != int(x0o[r])
                if cd < theta:
                    low_here = True
                    item["n_lowconf"] += 1
                if differ:
                    STATS["n_differ"] += 1
                    item["n_differ"] += 1
                    STATS["joint_hist_differ"][b][min(19, int(co * 20))] += 1
                    if co > cd:
                        STATS["n_differ_donor_more_conf"] += 1
                    if cd < theta:
                        STATS["n_lowconf_differ"] += 1
                        if co > cd:
                            STATS["n_lowconf_differ_donor_more"] += 1
                # --- the decision
                if GATED_MODE == "content_relgated":
                    # host unsure AND donor strictly more sure than the host by MARGIN
                    do_sub = cd < theta and co > cd + MARGIN
                elif GATED_MODE == "content_objection":
                    # how strongly does the DONOR prefer its own token over the HOST's?
                    # log p_donor(donor tok) - log p_donor(host tok), >= 0 by construction.
                    if differ:
                        ph = float(po_at_host[r])
                        obj = math.log(max(co, 1e-12)) - math.log(max(ph, 1e-12))
                        STATS["objection_hist"][min(19, int(obj * 2))] += 1
                        STATS["n_differ_scored"] += 1
                        do_sub = obj > TAU
                    else:
                        do_sub = False
                elif GATED_MODE == "content_random":
                    # matched-dose control: pick positions at random among the differing ones
                    do_sub = differ and RND.random() < RAND_FRAC
                elif GATED_MODE == "both_gated":
                    do_sub = ported
                elif GATED_MODE == "content_mean":
                    # mean of the teachers' RAW distributions (+ the student's with --with-student) at this position;
                    # theta >= 1 => every step mixes (plain probability ensemble control).
                    do_sub = cd < theta or theta >= 1.0
                    if do_sub:
                        dists = [rawo[r]] + [rw[r] for rw in raws_extra] + ([rawd[r]] if WITH_STUDENT else [])
                        tok = int(torch.stack(dists).mean(0).argmax())
                        toks = [int(x0o[r])] + [int(x0e[r]) for x0e in x0_extra]
                        STATS["n_mean_lowconf"] += 1
                        if len(toks) >= 2:
                            from collections import Counter
                            top, cnt = Counter(toks).most_common(1)[0]
                            vote = top if cnt * 2 > len(toks) else tok
                            if vote != tok:
                                STATS["n_vote_differs"] += 1
                        if tok == toks[-1]:
                            STATS["n_mean_eq_last_teacher"] += 1
                        if tok == int(x0d[r]):
                            STATS["n_mean_eq_student"] += 1
                        xo[0, j] = tok
                        differ = tok != int(x0d[r])
                elif GATED_MODE == "content_random_uncond":
                    # 2026-10-12 (user): UNCONDITIONAL random control. Every committed position (= every step here:
                    # the lockstep decoder commits one token per step) takes the donor's token with prob RAND_FRAC,
                    # whether or not the two models differ. Match RAND_FRAC to the gate's rate 1 (share of positions
                    # handed to the donor), not to the token-changed rate: conditioning on disagreement gives the
                    # control the gate's "find the disagreement" information for free.
                    do_sub = RND.random() < RAND_FRAC
                else:
                    do_sub = cd < theta
                if do_sub:
                    xa[0, j] = xo[0, j]
                    if differ:
                        STATS["n_substituted"] += 1
                        STATS["sub_positions"].append(j)
                        item["n_substituted"] += 1
                        sub_here = True
            if sub_here:
                STATS["n_steps_with_sub"] += 1
                item["n_steps_with_sub"] += 1
            if low_here:
                item["n_steps_with_lowconf"] += 1
        else:
            STATS["n_committed"] += len(commit[0])

        if record:
            cl = sorted(int(j) for j in commit[0].tolist())
            recs.append({"step": i, "n_masked": int(mask_index.sum()), "k": k, "pos": cl,
                         "tok_driver": [int(x0d[pos2row[j]]) for j in cl],
                         "tok_other": [int(x0o[pos2row[j]]) for j in cl],
                         "p_driver": [round(float(pd[pos2row[j]]), 5) for j in cl],
                         "p_driver_raw": [round(float(prd[pos2row[j]]), 6) for j in cl],
                         "p_other": [round(float(po[pos2row[j]]), 5) for j in cl],
                         "p_other_raw": [round(float(pro[pos2row[j]]), 6) for j in cl]})
        rows = torch.arange(B, device=dev).unsqueeze(1).expand_as(commit)
        x[rows, commit] = xa[rows, commit]
    if OUTDIR is not None:
        tf = OUTDIR / "texts.jsonl"
        done = {json.loads(l)["id"] for l in open(tf)} if tf.exists() else set()
        pf = OUTDIR / "gate_per_item.jsonl"
        item["id"] = min(k for k in range(START, 10**6) if k not in done)  # a crashed problem may log twice: keep the last
        with open(pf, "a") as f:
            f.write(json.dumps(item) + "\n")
    return x, recs


def main(argv=None):
    global THETA, GATED_MODE, MARGIN, TAU, RAND_FRAC, SEED
    argv = list(sys.argv[1:] if argv is None else argv)

    # pull our extra flags out of argv, then hand the rest to the original parser
    if "--theta" in argv:
        i = argv.index("--theta")
        THETA = float(argv[i + 1])
        del argv[i:i + 2]
    for flag, cast, name in (("--tau", float, "TAU"), ("--rand-frac", float, "RAND_FRAC"),
                             ("--seed", int, "SEED")):
        if flag in argv:
            i = argv.index(flag)
            globals()[name] = cast(argv[i + 1])
            del argv[i:i + 2]
    global EXTRA_OTHERS, WITH_STUDENT
    if "--with-student" in argv:
        WITH_STUDENT = True
        argv.remove("--with-student")
    if "--other" in argv:
        i = argv.index("--other")
        parts = argv[i + 1].split(",")
        if len(parts) > 1:
            argv[i + 1] = parts[0]
            EXTRA_OTHERS = parts[1:]
    if "--margin" in argv:
        i = argv.index("--margin")
        MARGIN = float(argv[i + 1])
        del argv[i:i + 2]
    if "--mode" in argv:
        i = argv.index("--mode")
        if argv[i + 1] in ("content_gated", "content_relgated", "content_objection",
                           "content_random", "content_random_uncond", "schedule_gated", "both_gated", "content_mean"):
            GATED_MODE = argv[i + 1]
            # the original parser only knows content/schedule; both require --other, which is
            # exactly the validation we want, so map onto the corresponding base mode.
            argv[i + 1] = "schedule" if GATED_MODE == "schedule_gated" else "content"
    if GATED_MODE in ("content_gated", "content_relgated", "schedule_gated", "both_gated", "content_mean") and THETA is None:
        raise SystemExit("--theta is required with --mode content_gated/content_relgated/schedule_gated")
    if GATED_MODE == "content_objection" and TAU is None:
        raise SystemExit("--tau is required with --mode content_objection")
    if GATED_MODE in ("content_random", "content_random_uncond") and RAND_FRAC is None:
        raise SystemExit("--rand-frac is required with --mode content_random")

    global OUTDIR, START
    if "--start" in argv:
        i = argv.index("--start")
        START = int(argv[i + 1])
        del argv[i:i + 2]
        _lp = LP.load_prompts
        LP.load_prompts = lambda b, p, n: [it for it in _lp(b, p, n) if it["id"] >= START]
    if "--out-dir" in argv:
        OUTDIR = Path(argv[argv.index("--out-dir") + 1])
    LP._run_orig = LP.run
    LP.run = run_gated
    try:
        LP.main(argv)
    finally:
        out = None
        if "--out-dir" in argv:
            out = Path(argv[argv.index("--out-dir") + 1])
        if out is not None and out.exists():
            n_c = STATS["n_committed"] or 1
            stats = dict(STATS)
            stats["theta"] = THETA
            stats["margin"] = MARGIN
            stats["tau"] = TAU
            stats["rand_frac"] = RAND_FRAC
            stats["seed"] = SEED
            stats["gated_mode"] = GATED_MODE
            stats["extra_others"] = EXTRA_OTHERS
            stats["with_student"] = WITH_STUDENT
            nd = STATS["n_differ"] or 1
            nl = STATS["n_lowconf_differ"] or 1
            stats["donor_more_conf_rate_where_differ"] = STATS["n_differ_donor_more_conf"] / nd
            stats["donor_more_conf_rate_where_lowconf_differ"] = STATS["n_lowconf_differ_donor_more"] / nl
            stats["substitution_rate"] = STATS["n_substituted"] / n_c
            stats["disagreement_rate"] = STATS["n_differ"] / n_c
            if STATS["n_steps"]:
                stats["sched_port_step_rate"] = STATS["n_steps_sched_ported"] / STATS["n_steps"]
            # keep the histogram, drop the raw list if it is huge
            if len(stats["sub_positions"]) > 20000:
                stats["sub_positions"] = stats["sub_positions"][:20000]
                stats["sub_positions_truncated"] = True
            (out / "gate_stats.json").write_text(json.dumps(stats, indent=1))
            print(f"[gate] theta={THETA} mode={GATED_MODE} "
                  f"sub_rate={stats['substitution_rate']:.4f} "
                  f"disagree_rate={stats['disagreement_rate']:.4f}", flush=True)


if __name__ == "__main__":
    main()
