"""compare_rollouts.py A.jsonl B.jsonl — same prompts decoded two ways (e.g. batch 1 vs batch 16 at temperature 0): identical texts? finish rate?
first differing position? Prints a per-prompt line and a summary. CPU only."""
import json, sys
A = {json.loads(l)["id"]: json.loads(l) for l in open(sys.argv[1])}
B = {json.loads(l)["id"]: json.loads(l) for l in open(sys.argv[2])}
ids = sorted(set(A) & set(B)); same = 0; fin = [0, 0]; first_diff = []
for i in ids:
    a, b = A[i]["resp_ids"], B[i]["resp_ids"]
    fin[0] += A[i]["finished"]; fin[1] += B[i]["finished"]
    k = next((j for j in range(min(len(a), len(b))) if a[j] != b[j]), None)
    if k is None and len(a) == len(b):
        same += 1; print(f"{i:5d} identical  len {len(a)}")
    else:
        k = min(len(a), len(b)) if k is None else k
        first_diff.append(k); print(f"{i:5d} differ at {k:4d}  len {len(a)} vs {len(b)}  finished {A[i]['finished']} vs {B[i]['finished']}")
print(f"\n{same}/{len(ids)} identical; finished {fin[0]}/{len(ids)} vs {fin[1]}/{len(ids)}; "
      f"first diff position median {sorted(first_diff)[len(first_diff) // 2] if first_diff else '-'}")
