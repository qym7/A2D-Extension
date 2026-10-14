#!/usr/bin/env python3
"""Tests for the two new gates. No GPU, no real model.

1. model_step_ext must be BYTE-IDENTICAL to lockstep_port.model_step on its four shared returns --
   it is a re-implementation, so this is the guard against protocol drift.
2. content_objection must fire exactly when  log p_donor(donor tok) - log p_donor(host tok) > tau.
3. content_random must substitute only where the two models differ, at about the requested rate.
"""
import math, sys, torch
sys.path.insert(0, ".")
import lockstep_gated as G
import lockstep_port as LP

MASK = 7          # vocab 0..7, MASK is the last id
V, L = 8, 4

class Fake:
    """model(x, 'full', None).logits -> preset (1, L, V)."""
    def __init__(self, lg): self.lg = lg
    def __call__(self, x, mode, attn):
        class O: pass
        o = O(); o.logits = self.lg
        return o

def logits_from_probs(rows):
    """rows[j] = {token: prob} in INPUT space; missing mass spread over the remaining tokens.

    model_step applies  out[0] = in[0],  out[i] = in[i-1],  which is NOT invertible (out[0] and
    out[1] are forced to share in[0]), so the fixture is written in input space and the expected
    per-position values are derived with SRC below instead of being pre-shifted.
    """
    lg = torch.zeros(1, L, V)
    for j, row in enumerate(rows):
        p = torch.full((V,), (1.0 - sum(row.values())) / (V - len(row)))
        for t, v in row.items(): p[t] = v
        lg[0, j] = torch.log(p)
    return lg


SRC = [0] + list(range(L - 1))      # output position i reads input row SRC[i]

fails = []
def check(name, got, want, tol=None):
    ok = (abs(got - want) <= tol) if tol is not None else (got == want)
    print(f"  [{'ok' if ok else 'FAIL'}] {name}: got {got} want {want}")
    if not ok: fails.append(name)

# ---------------------------------------------------------------- 1. bit-identity
print("1. model_step_ext == LP.model_step (4 shared returns), 5 random cases")
torch.manual_seed(0)
for case in range(5):
    lg = torch.randn(1, L, V) * 3
    m = Fake(lg)
    x = torch.full((1, L), MASK)
    mi = (x == MASK)
    a = LP.model_step(m, x, mi, 0.9, 0.1)
    b = G.model_step_ext(m, x, mi, 0.9, 0.1, ref_tokens=torch.zeros(L, dtype=torch.long))
    same = (torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
            and torch.equal(a[2], b[2]) and a[3] == b[3] and torch.equal(a[4], b[4]))
    print(f"  [{'ok' if same else 'FAIL'}] case {case}")
    if not same: fails.append(f"bit-identity case {case}")

# ---------------------------------------------------------------- 2. objection gate
# host argmax = token 1 everywhere; donor argmax = token 2 everywhere.
HOST = logits_from_probs([{1: 0.90, 2: 0.02}] * L)
DONOR_ROWS = [{2: 0.50, 1: 0.40},    # obj = ln 1.25 = 0.2231
              {2: 0.60, 1: 0.20},    # obj = ln 3    = 1.0986
              {2: 0.90, 1: 0.01},    # obj = ln 90   = 4.4998
              {2: 0.40, 1: 0.30}]    # unused (dropped by the shift)
DONOR = logits_from_probs(DONOR_ROWS)
OBJ = [math.log(DONOR_ROWS[SRC[i]][2] / DONOR_ROWS[SRC[i]][1]) for i in range(L)]
print(f"\n2. content_objection; per-position objection scores are {[round(o,4) for o in OBJ]}")

def run(mode, **kw):
    G.STATS.update({"n_committed": 0, "n_differ": 0, "n_substituted": 0, "n_steps": 0,
                    "n_steps_sched_ported": 0, "sub_positions": [], "raw_conf_hist": [0]*20,
                    "joint_hist_differ": [[0]*20 for _ in range(20)],
                    "n_differ_donor_more_conf": 0, "n_lowconf_differ": 0,
                    "n_lowconf_differ_donor_more": 0, "objection_hist": [0]*20,
                    "n_differ_scored": 0})
    G.GATED_MODE, G.THETA, G.MARGIN = mode, kw.get("theta", 1.0), kw.get("margin", 0.0)
    G.TAU, G.RAND_FRAC, G.SEED = kw.get("tau"), kw.get("rand_frac"), kw.get("seed", 0)
    x = torch.full((1, L), MASK)
    orig = LP.model_step
    LP.model_step = lambda m, *a, **k: LP.__dict__["model_step"] if False else _ms(m, *a, **k)
    def _ms(m, *a, **k): return orig(m, *a, **k)
    LP.model_step = _ms
    try:
        out, _ = G.run_gated(Fake(HOST), Fake(DONOR), x, MASK, mode="content", steps=1,
                             eps=1e-3, top_p=0.9, conf_temp=0.1, record=False, rng=None)
    finally:
        LP.model_step = orig
    return out[0].tolist(), sorted(G.STATS["sub_positions"]), dict(G.STATS)

for tau, want in ((1.0, [2, 3]), (0.2, [0, 1, 2, 3]), (5.0, [])):
    toks, subs, st = run("content_objection", tau=tau)
    check(f"tau={tau} substituted positions", subs, want)
    check(f"tau={tau} tokens", toks, [2 if i in want else 1 for i in range(L)])
_, _, st = run("content_objection", tau=1.0)
check("all 4 differing positions scored", st["n_differ_scored"], 4)

# ---------------------------------------------------------------- 3. random control
print("\n3. content_random: only at differing positions, at about the requested rate")
for frac in (0.0, 1.0):
    toks, subs, st = run("content_random", rand_frac=frac)
    check(f"rand_frac={frac} n_substituted", st["n_substituted"], 0 if frac == 0 else L)
# with an all-agree donor nothing may be substituted even at frac=1
AGREE = logits_from_probs([{1: 0.90, 2: 0.02}] * L)
G.STATS["n_substituted"] = 0
G.GATED_MODE, G.RAND_FRAC, G.SEED, G.THETA = "content_random", 1.0, 0, 1.0
x = torch.full((1, L), MASK)
out, _ = G.run_gated(Fake(HOST), Fake(AGREE), x, MASK, mode="content", steps=1, eps=1e-3,
                     top_p=0.9, conf_temp=0.1, record=False, rng=None)
check("identical donor -> zero substitutions", G.STATS["n_substituted"], 0)

# rate check over many draws
G.STATS.update({"n_differ": 0, "n_substituted": 0})
hit = 0; trials = 400
import random as _r
G.SEED = 1
for t in range(trials):
    G.STATS["n_substituted"] = 0
    G.SEED = t
    x = torch.full((1, L), MASK)
    G.GATED_MODE, G.RAND_FRAC = "content_random", 0.25
    G.run_gated(Fake(HOST), Fake(DONOR), x, MASK, mode="content", steps=1, eps=1e-3,
                top_p=0.9, conf_temp=0.1, record=False, rng=None)
    hit += G.STATS["n_substituted"]
rate = hit / (trials * L)
check("empirical rate at rand_frac=0.25", round(rate, 2), 0.25, tol=0.05)

print()
print("FAILED: " + ", ".join(fails) if fails else "ALL PASS")
sys.exit(1 if fails else 0)
