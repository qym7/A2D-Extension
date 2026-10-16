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
