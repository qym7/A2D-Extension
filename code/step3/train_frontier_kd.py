"""STEP3 trainer — frontier over-distillation (plan §2–3, `technical_details/step3_plan_frontier_kd.md`). DRAFT 2026-10-15, not yet smoked.
Single GPU (H200·0.51): student (bf16 + LoRA, grad-ckpt) and a frozen teacher side by side; micro-batch = 1 state (no padding), gradient
accumulation to `--batch` states.

One training state = a rollout canvas  prompt ‖ x_<c ‖ [MASK]^(L−c)  at a random cut c (suffix corruption; `--corruption random` = i.i.d. masking
of the response, control C4). The loss lives on the FRONTIER = the first K masked positions:
  M  : KL(p_T ‖ p_S) at frontier positions where max p_S < θ                (+ λ·CE on the rollout token)
  C1 : KL at every frontier position (no θ)                                  (selection control)
  C2 : CE on the rollout token at every frontier position, no teacher        (self-SFT control)
  C3 : = M with another teacher (pass --teacher add-a10-b10)                 (over control)
  C4 : = M with --corruption random                                          (suffix control)
Dream logit convention: the token at position p is predicted from hidden p−1 (shift as in scripts/step3/signal_check.py).
Writes <out>/lora (adapter), <out>/merged (bf16 full checkpoint loadable by every eval harness), <out>/train_log.jsonl."""
import argparse, json, random, time, shutil, os
from pathlib import Path
import torch, torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

ARMS = {"M": dict(target="kd", select=True), "C1": dict(target="kd", select=False), "C2": dict(target="ce", select=False),
        "C3": dict(target="kd", select=True), "C4": dict(target="kd", select=True)}


def shifted_logits(model, x):
    logits = model(x, "full", None).logits
    return torch.cat([logits[:, :1], logits[:, :-1]], dim=1)


def make_state(rec, mask_id, corruption, rng, max_len):
    """Returns (ids, frontier_positions, targets) for one rollout record; ids already corrupted."""
    P = rec["prompt_len"]; resp = rec["resp_ids"]; L = len(resp)
    ids = list(rec["prompt_ids"]) + resp
    if len(ids) > max_len:
        return None
    if corruption == "suffix":
        c = rng.randint(1, L - 1) if L > 2 else 1
        masked = list(range(P + c, P + L))
    else:   # random i.i.d. masking at ratio r ~ U(0,1), control C4
        r = rng.random()
        masked = [P + j for j in range(L) if rng.random() < r] or [P + rng.randint(0, L - 1)]
    for q in masked:
        ids[q] = mask_id
    return ids, masked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=list(ARMS), required=True)
    ap.add_argument("--student", required=True)
    ap.add_argument("--teacher", default=None, help="checkpoint dir; required for kd arms")
    ap.add_argument("--rollouts", required=True, help="rollouts.jsonl (with prompt_ids attached, see --prompts)")
    ap.add_argument("--prompts", required=True, help="prompts.jsonl (prompt text -> prompt_ids via the chat template)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--corruption", choices=["suffix", "random"], default="suffix")
    ap.add_argument("--theta", type=float, default=0.8)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--lambda-ce", type=float, default=0.1)
    ap.add_argument("--cuts-per-canvas", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=32, help="states per optimizer step (micro-batch is 1)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=128)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--max-steps", type=int, default=0, help="smoke: stop after this many optimizer steps")
    ap.add_argument("--n-canvases", type=int, default=0, help="smoke: use only the first n rollouts")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    arm = ARMS[a.arm]
    if arm["target"] == "kd" and not a.teacher:
        raise SystemExit("kd arm needs --teacher")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed); torch.manual_seed(a.seed)

    tok = AutoTokenizer.from_pretrained(a.student, trust_remote_code=True)
    MASK = tok.convert_tokens_to_ids("<|mask|>")
    ptext = {json.loads(l)["id"]: json.loads(l)["prompt"] for l in open(a.prompts)}
    recs = []
    for l in open(a.rollouts):
        r = json.loads(l)
        if r["resp_len"] < 4:
            continue
        r["prompt_ids"] = tok.apply_chat_template([{"role": "user", "content": ptext[r["id"]]}], add_generation_prompt=True)
        assert len(r["prompt_ids"]) == r["prompt_len"], (r["id"], len(r["prompt_ids"]), r["prompt_len"])
        recs.append(r)
        if a.n_canvases and len(recs) >= a.n_canvases:
            break
    print(f"[data] {len(recs)} canvases, {a.cuts_per_canvas} cuts each, {a.epochs} epochs", flush=True)

    from peft import LoraConfig, get_peft_model
    student = AutoModel.from_pretrained(a.student, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
    try:
        student.gradient_checkpointing_enable()
    except Exception as e:
        print("[warn] no gradient checkpointing:", e)
    lcfg = LoraConfig(r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=0.0, bias="none",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    student = get_peft_model(student, lcfg); student.print_trainable_parameters()
    teacher = None
    if arm["target"] == "kd":
        teacher = AutoModel.from_pretrained(a.teacher, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda().eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
    opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
    n_states = len(recs) * a.cuts_per_canvas * a.epochs
    total_steps = max(1, n_states // a.batch)
    if a.max_steps:
        total_steps = min(total_steps, a.max_steps)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, int(0.1 * total_steps))) * max(0.0, 1 - s / total_steps))
    log = open(out / "train_log.jsonl", "a")
    step = 0; acc = dict(n=0, n_sel=0, n_front=0, n_empty=0, kd=0.0, ce=0.0); t0 = time.time()
    student.train()
    done = False
    for ep in range(a.epochs):
        order = [(i, k) for i in range(len(recs)) for k in range(a.cuts_per_canvas)]
        rng.shuffle(order)
        for i, _ in order:
            st = make_state(recs[i], MASK, a.corruption, rng, a.max_len)
            if st is None:
                continue
            ids, masked = st
            front = masked[: a.K] if a.corruption == "suffix" else masked
            x = torch.tensor([ids], device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ls = shifted_logits(student, x)[0, front].float()
            logp_s = F.log_softmax(ls, -1)
            gold = torch.tensor([recs[i]["prompt_ids"] + recs[i]["resp_ids"]], device="cuda")[0, front]
            ce = F.nll_loss(logp_s, gold)
            if arm["target"] == "kd":
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    lt = shifted_logits(teacher, x)[0, front].float()
                p_t = F.softmax(lt, -1)
                conf_s = logp_s.detach().exp().max(-1).values
                sel = conf_s < a.theta if arm["select"] else torch.ones_like(conf_s, dtype=torch.bool)
                acc["n_sel"] += int(sel.sum()); acc["n_front"] += len(front)
                if sel.any():
                    kd = (p_t * (torch.log(p_t + 1e-12) - logp_s)).sum(-1)[sel].mean()
                else:
                    kd = ls.sum() * 0.0; acc["n_empty"] += 1
                loss = kd + a.lambda_ce * ce
                acc["kd"] += float(kd)
            else:
                loss = ce
                acc["n_front"] += len(front)
            acc["ce"] += float(ce); acc["n"] += 1
            (loss / a.batch).backward()
            if acc["n"] % a.batch == 0:
                torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad], 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
                rec = dict(step=step, epoch=ep, kd=acc["kd"] / a.batch, ce=acc["ce"] / a.batch, sel_rate=acc["n_sel"] / max(1, acc["n_front"]),
                           empty_states=acc["n_empty"], lr=sched.get_last_lr()[0], s_per_state=(time.time() - t0) / (step * a.batch))
                log.write(json.dumps(rec) + "\n"); log.flush()
                if step % 10 == 1 or step == total_steps:
                    print("[train]", json.dumps(rec), flush=True)
                acc = dict(n=0, n_sel=0, n_front=0, n_empty=0, kd=0.0, ce=0.0)
                if step >= total_steps:
                    done = True; break
        if done:
            break
    student.save_pretrained(out / "lora")
    merged = student.merge_and_unload()
    merged.save_pretrained(out / "merged", safe_serialization=True)
    tok.save_pretrained(out / "merged")
    # Dream's remote code: copy the modeling/config python files next to the weights so trust_remote_code loads from the merged dir
    from transformers.utils.hub import cached_file
    src_dir = Path(cached_file(a.student, "config.json")).parent
    for f in src_dir.glob("*.py"):
        shutil.copy(f, out / "merged" / f.name)
    json.dump(vars(a), open(out / "args.json", "w"), indent=1)
    print(f"[done] {step} steps in {(time.time() - t0) / 3600:.2f} h -> {out / 'merged'}", flush=True)


if __name__ == "__main__":
    main()
