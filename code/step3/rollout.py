"""STEP3 on-policy prefixes (plan §1): the student decodes every prompt once with the official Dream sampler
(MATH-500 settings: max_new_tokens 512, steps 512, temperature 0.1, top_p 0.9, alg=entropy) and we keep the canvas + commit order.
Training states are cut from these canvases (suffix corruption); the per-position commit step lets the trainer sample cuts along the
decode order if wanted. One line per prompt in <out>/rollouts.jsonl:
  {id, source, prompt_len, resp_ids (up to and incl. the first <|im_end|>/<|endoftext|>), resp_len, finished,
   commit_step[per response position: the sampler step that wrote it], commit_conf[raw T=1 probability of the written token at that step],
   text}
The trainer replays frames of this trajectory (state after step t = tokens with commit_step <= t, the rest masked) and samples frames
preferentially right before a low-confidence commit (commit_conf < theta at step t+1).
Batched with left padding (Dream's generate accepts attention_mask); batch=1 reproduces the official per-prompt run exactly.
Shardable: --start/--end over prompt ids. GPU job only (never on the login node)."""
import argparse, json, time
from pathlib import Path
import torch, transformers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=10**9)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--steps", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    done = set()
    fo_path = out / "rollouts.jsonl"
    if fo_path.exists():
        done = {json.loads(l)["id"] for l in open(fo_path)}
    rows = [json.loads(l) for l in open(a.prompts)]
    rows = [r for r in rows if a.start <= r["id"] < a.end and r["id"] not in done]
    print(f"[rollout] {len(rows)} prompts to do (skipping {len(done)} done)", flush=True)

    tok = transformers.AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    m = transformers.AutoModel.from_pretrained(a.model, torch_dtype=torch.bfloat16, trust_remote_code=True).eval().cuda()
    MASK = m.generation_config.mask_token_id if m.generation_config.mask_token_id is not None else m.config.mask_token_id
    IM_END = tok.convert_tokens_to_ids("<|im_end|>"); EOS = tok.eos_token_id
    fo = open(fo_path, "a")
    t0 = time.time(); n = 0
    for b in range(0, len(rows), a.batch):
        batch = rows[b:b + a.batch]
        texts = [tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], add_generation_prompt=True, tokenize=False) for r in batch]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
        ids, am = enc.input_ids.cuda(), enc.attention_mask.cuda()
        P = ids.shape[1]
        # raw (T = 1) probability of each committed token at the step it was written = the gate's confidence signal; captured through the
        # sampler's token hook (called after step i with the step's shifted raw logits). Used by the trainer to sample "frames before a
        # low-confidence commit".
        conf_at = torch.zeros(ids.shape[0], a.max_new, device="cuda"); prev = {"x": None}

        def hook(step, x, logits):
            if step is None:
                prev["x"] = x.clone(); return x
            new = (prev["x"] == MASK) & (x != MASK)
            if new.any():
                with torch.no_grad():
                    p = torch.softmax(logits.float(), -1).gather(-1, x.unsqueeze(-1)).squeeze(-1)   # [B, L]
                nr = new[:, P:]
                conf_at[nr] = p[:, P:][nr]
            prev["x"] = x.clone(); return x

        with torch.no_grad():
            o = m.diffusion_generate(ids, attention_mask=am, max_new_tokens=a.max_new, output_history=True, return_dict_in_generate=True,
                                     steps=a.steps, temperature=a.temperature, top_p=a.top_p, alg="entropy", alg_temp=0.0,
                                     generation_tokens_hook_func=hook)
        seq = o.sequences[:, P:].cpu()                                   # [B, max_new]
        hist = torch.stack([h[:, P:].cpu() for h in o.history])          # [S, B, max_new]  (state after step 0 … S-1)
        was_mask = hist == MASK
        # commit step of each response position: number of recorded states in which it is still MASK = index of the step that wrote it
        # (history[k] is the state AFTER step k, so a token written at step k is masked in history[0..k-1] -> commit = k)
        commit = was_mask.sum(0)
        conf_cpu = conf_at.cpu()
        for i, r in enumerate(batch):
            resp = seq[i]
            ends = [(resp == t).nonzero() for t in (IM_END, EOS)]
            ends = [e[0].item() for e in ends if len(e)]
            finished = len(ends) > 0
            L = (min(ends) + 1) if finished else resp.shape[0]
            resp_ids = resp[:L].tolist()
            rec = dict(id=r["id"], source=r["source"], prompt_len=int(am[i].sum().item()), resp_ids=resp_ids, resp_len=L, finished=finished,
                       commit_step=commit[i, :L].tolist(), commit_conf=[round(c, 4) for c in conf_cpu[i, :L].tolist()],
                       text=tok.decode(resp[:L], skip_special_tokens=True))
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fo.flush(); n += len(batch)
        if (b // a.batch) % 10 == 0:
            el = time.time() - t0
            print(f"[rollout] {n}/{len(rows)}  {el / n:.1f} s/prompt  eta {(len(rows) - n) * el / n / 3600:.1f} h", flush=True)
    print(f"[rollout] done {n} in {(time.time() - t0) / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
