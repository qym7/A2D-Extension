#!/usr/bin/env python3
"""Synthetic test of the content_relgated decision rule. No model, no GPU.

Stubs lockstep_port.model_step so the (token, raw confidence) of both models is controlled
exactly, then checks which positions the gate substitutes.
"""
import sys, torch
sys.path.insert(0, ".")
import lockstep_gated as G
import lockstep_port as LP

MASK = 99
# 4 masked positions. driver tokens 10..13, donor tokens 20..23 (all DIFFER).
# raw confidences chosen to hit every case at theta=0.5:
#   pos0: host 0.30 donor 0.80 -> host unsure, donor surer  -> relgate YES, old gate YES
#   pos1: host 0.30 donor 0.20 -> host unsure, donor worse  -> relgate NO,  old gate YES
#   pos2: host 0.90 donor 0.99 -> host sure                 -> relgate NO,  old gate NO
#   pos3: host 0.10 donor 0.11 -> host unsure, donor barely surer -> margin test
CD = [0.30, 0.30, 0.90, 0.10]
CO = [0.80, 0.20, 0.99, 0.11]

def make_step(tokens, craw):
    def f(model, x, mask_index, top_p, conf_temp):
        n = int(mask_index.sum())
        x0 = torch.tensor(tokens[:n])
        praw = torch.tensor(craw[:n])
        # confidence used for ORDER: make it identical for both models so the schedule is fixed
        negent = torch.tensor([-1.0, -2.0, -3.0, -4.0][:n])
        return x0, negent, torch.ones(n), torch.float32, praw
    return f

def run_case(mode, theta, margin=0.0):
    G.STATS.update({"n_committed": 0, "n_differ": 0, "n_substituted": 0, "n_steps": 0,
                    "n_steps_sched_ported": 0, "sub_positions": [], "raw_conf_hist": [0]*20,
                    "joint_hist_differ": [[0]*20 for _ in range(20)],
                    "n_differ_donor_more_conf": 0, "n_lowconf_differ": 0,
                    "n_lowconf_differ_donor_more": 0})
    G.GATED_MODE, G.THETA, G.MARGIN = mode, theta, margin
    x = torch.full((1, 4), MASK)
    orig = LP.model_step
    calls = {"n": 0}
    def dispatch(model, *a, **k):
        calls["n"] += 1
        return (make_step([10,11,12,13], CD) if model == "driver" else make_step([20,21,22,23], CO))(model, *a, **k)
    LP.model_step = dispatch
    try:
        out, _ = G.run_gated("driver", "other", x, MASK, mode="content", steps=1, eps=1e-3,
                             top_p=0.9, conf_temp=0.1, record=False, rng=None)
    finally:
        LP.model_step = orig
    return out[0].tolist(), sorted(G.STATS["sub_positions"]), dict(G.STATS)

fails = []
def check(name, got, want):
    ok = got == want
    print(f"  [{'ok' if ok else 'FAIL'}] {name}: got {got} want {want}")
    if not ok: fails.append(name)

print("theta=0.5, old rule (content_gated): substitutes wherever host raw p < 0.5")
toks, subs, st = run_case("content_gated", 0.5)
check("tokens", toks, [20, 21, 12, 23])          # 0,1,3 swapped; 2 kept
check("sub_positions", subs, [0, 1, 3])

print("theta=0.5, new rule (content_relgated, margin=0): also needs donor raw p > host raw p")
toks, subs, st = run_case("content_relgated", 0.5)
check("tokens", toks, [20, 11, 12, 23])          # pos1 no longer swapped (donor worse)
check("sub_positions", subs, [0, 3])

print("theta=0.5, new rule with margin=0.05: pos3 (0.11 vs 0.10) must drop out")
toks, subs, st = run_case("content_relgated", 0.5, margin=0.05)
check("tokens", toks, [20, 11, 12, 13])
check("sub_positions", subs, [0])

print("theta=0 must never substitute, in both modes")
for m in ("content_gated", "content_relgated"):
    toks, subs, _ = run_case(m, 0.0)
    check(f"{m} theta=0 tokens", toks, [10, 11, 12, 13])
    check(f"{m} theta=0 subs", subs, [])

print("diagnostics are accumulated regardless of the decision (theta=0.5, relgated)")
_, _, st = run_case("content_relgated", 0.5)
check("n_differ", st["n_differ"], 4)
check("n_differ_donor_more_conf", st["n_differ_donor_more_conf"], 3)   # pos0,2,3
check("n_lowconf_differ", st["n_lowconf_differ"], 3)                    # pos0,1,3
check("n_lowconf_differ_donor_more", st["n_lowconf_differ_donor_more"], 2)  # pos0,3

print()
print("FAILED: " + ", ".join(fails) if fails else "ALL PASS")
sys.exit(1 if fails else 0)
