"""Merge sharded runs (same arm, disjoint id ranges) into <arm>-merged/{texts.jsonl,gate_per_item.jsonl}. Zero GPU.
Originals are untouched. Shards were produced by run_gated_shard*.sh (2026-10-12); decode is argmax-deterministic, so
an id's result does not depend on which shard produced it."""
import glob, json, sys
from pathlib import Path
L = Path(__file__).resolve().parents[1] / "details" / "logs"
ARMS = {  # merged dir : glob of source dirs
 "lockstep1024/math500/add30-free-merged": "lockstep1024/math500/add30-free-last300*",
 "lockstep1024/math500/add30-free-merged": "lockstep1024/math500/add30-free-[ls]*",
 "lockstep1024/math500/add-a07-b13-free-merged": "lockstep1024/math500/add-a07-b13-free-s*",
 "gated1024/math500/e1b-mrg10-t06-merged": "gated1024/math500/e1b-mrg10-t06-last300-s*",
 "gated1024/math500/e1b-mrg10-t08-merged": "gated1024/math500/e1b-mrg10-t08-last300-s*",
 "gated1024/math500/e1b-mrg10-t09-merged": "gated1024/math500/e1b-mrg10-t09-last300-s*",
 "gated_mrg0406_add-a10-b30_1024/math500/t08-merged": "gated_mrg0406_add-a10-b30_1024/math500/t08-last300*",
 "gated_mrg0406_add-a10-b20_1024/math500/rand0854-merged": "gated_mrg0406_add-a10-b20_1024/math500/rand0854-last300*",
 "gated_mrg0406_add-a10-b20_1024/math500/rand0262-merged": "gated_mrg0406_add-a10-b20_1024/math500/rand0262-last300*",
 "gated_mrg0406_add-a10-b20_1024/math500/rand0854-seed1-merged": "gated_mrg0406_add-a10-b20_1024/math500/rand0854-seed1-last300-s*",
 "gated_mrg0406_add-a10-b20_1024/math500/rand0854-seed2-merged": "gated_mrg0406_add-a10-b20_1024/math500/rand0854-seed2-last300-s*",
 "gated_mrg0406_add-a10-b20_1024/math500/randu0028-merged": "gated_mrg0406_add-a10-b20_1024/math500/randu0028-last300-s*",
 "gated_mrg0406_add-a10-b20_1024/math500/randu0074-merged": "gated_mrg0406_add-a10-b20_1024/math500/randu0074-last300-s*",
 "gated_mrg0406_add-a10-b20_1024/math500/randu0149-merged": "gated_mrg0406_add-a10-b20_1024/math500/randu0149-last300-s*",
 "gated_ins_add-a10-b20_1024/math500/randu0053-merged": "gated_ins_add-a10-b20_1024/math500/randu0053-last300-s*",
 "gated_ins_add-a10-b20_1024/math500/randu0135-merged": "gated_ins_add-a10-b20_1024/math500/randu0135-last300-s*",
 "gated_ins_add-a10-b20_1024/math500/both08-merged": "gated_ins_add-a10-b20_1024/math500/both08-last300-s*",
 "gated1024/math500/e1b-mrg10-both08-merged": "gated1024/math500/e1b-mrg10-both08-last300-s*",
 "gated_ins_add-a10-b20_1024/math500/sched08-merged": "gated_ins_add-a10-b20_1024/math500/sched08-last300-s*",
 "gated1024/math500/e1b-mrg10-sched08-merged": "gated1024/math500/e1b-mrg10-sched08-last300-s*",
 "gated_mrg0406_add-a10-b20_1024/math500/t10-merged": "gated_mrg0406_add-a10-b20_1024/math500/t10-last300*",
 "gated_mrg0406_add-a10-b20_1024/math500/t04-merged": "gated_mrg0406_add-a10-b20_1024/math500/t04-last300*",
 "gated_mrg0406_add-a10-b20_1024/math500/t06-merged": "gated_mrg0406_add-a10-b20_1024/math500/t06*",
}
for dst, pat in ARMS.items():
    srcs = [Path(p) for p in sorted(glob.glob(str(L / pat))) if not p.endswith("-merged")]
    for fn in ["texts.jsonl", "gate_per_item.jsonl"]:
        rows = {}
        for s in srcs:
            f = s / fn
            if f.exists():
                for l in open(f):
                    r = json.loads(l); rows[r["id"]] = r   # later shard wins on duplicates (crash re-logs)
        if rows:
            (L / dst).mkdir(parents=True, exist_ok=True)
            with open(L / dst / fn, "w") as f:
                for i in sorted(rows): f.write(json.dumps(rows[i], ensure_ascii=False) + "\n")
    ids = sorted({json.loads(l)["id"] for l in open(L / dst / "texts.jsonl")}) if (L / dst / "texts.jsonl").exists() else []
    print(f"{dst}: {len(ids)} ids [{ids[0] if ids else '-'}..{ids[-1] if ids else '-'}] from {len(srcs)} dirs")
