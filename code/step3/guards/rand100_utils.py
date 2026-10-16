"""ifeval_rand100: the official IFEval scoring (re-exported from lm_eval.tasks.ifeval.utils) on a fixed random subset of 100 prompts.
IDS = sorted(random.Random(0).sample(range(541), 100)) — the same positions in google/IFEval train for every model; also used by
score_ifeval_subset.py to read the same 100 out of an existing full-541 samples file (doc_id). 2026-10-16."""
import random
from lm_eval.tasks.ifeval.utils import process_results, agg_inst_level_acc  # noqa: F401  (official scorer, unchanged)

N_TOTAL, N_SUB, SEED = 541, 100, 0
IDS = sorted(random.Random(SEED).sample(range(N_TOTAL), N_SUB))


def select_rand100(dataset):
    assert len(dataset) == N_TOTAL, len(dataset)
    return dataset.select(IDS)

# validation twin (2026-10-16): random 100 of the 441 prompts not in IDS, seed 1 — file val_ifeval_rand100_seed1_disjoint.json next to this module
import json as _json, os as _os
IDS_VAL = _json.load(open(_os.path.join(_os.path.dirname(__file__), "val_ifeval_rand100_seed1_disjoint.json")))
assert not set(IDS_VAL) & set(IDS) and len(IDS_VAL) == N_SUB


def select_val100(dataset):
    assert len(dataset) == N_TOTAL, len(dataset)
    return dataset.select(IDS_VAL)
