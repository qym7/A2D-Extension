"""STEP3 trainer — distil the gate into one model by replaying the student's own denoising trajectory. DRAFT 2026-10-15, not yet smoked.
Single GPU (H200·0.51): student (bf16 + LoRA, grad-ckpt) and a frozen teacher side by side; micro-batch = 1 frame (no padding), gradient
accumulation to `--batch` frames.

A training frame = the student's rollout (code/step3/rollout.py) rewound to step t: tokens with commit_step <= t are visible, the rest are
[MASK] (`--corruption replay`, the default = exactly the state the sampler was in). NEXT(t) = the positions the sampler wrote at step t+1.
Frames are sampled per canvas: `--low-frames` of them from steps where some NEXT token was written with raw confidence < theta
("the moment before a low-confidence commit"), the rest uniformly over steps.
Loss on NEXT(t), split by the student's raw confidence c_j = max p_S(.|frame, j):
  c_j <  theta  (unsure) : D( p_S(j) , p_T(j) )              teacher distribution, --kl reverse|forward|jsd        ("learn the teacher")
  c_j >= theta  (sure)   : lambda * KL( p_S(j) || p_S0(j) )  p_S0 = the student before training (LoRA disabled)  ("do not drift")
so the per-frame target is the gate itself: teacher where the student is unsure, the original student elsewhere.
Arms (plan §3): M = the above; C1 = no theta split (every NEXT position -> teacher); C2 = CE on the student's own rollout token (self-SFT, no
teacher); C3 = non-over teacher; C4 = random-mask corruption; C5 = forward KL; `--anchor none|self|ce` selects what the sure positions get.
Dream logit convention: the token at position p is predicted from hidden p-1 (shift as in code/step3/signal_check.py).
Writes <out>/lora (adapter), <out>/merged (bf16 full checkpoint loadable by every eval harness), <out>/train_log.jsonl."""
import argparse, json, random, time, shutil, math
from pathlib import Path
import torch, torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

ARMS = {"M": dict(target="kd", select=True), "C1": dict(target="kd", select=False), "C2": dict(target="ce", select=False),
        "C3": dict(target="kd", select=True), "C4": dict(target="kd", select=True), "C5": dict(target="kd", select=True)}


def shifted_logits(model, x):
    logits = model(x, "full", None).logits
    return torch.cat([logits[:, :1], logits[:, :-1]], dim=1)


def divergence(ls, lt, kind, top_k, temp):
    """Mean over positions of a divergence between student logits ls [n, V] (grad) and teacher logits lt [n, V] (no grad).
    top_k > 0 restricts both distributions to the teacher's top-k support (renormalised) — the long tail is noise (BiLD / TOPD use k = 8–10)."""
    ls = ls / temp; lt = lt / temp
    if top_k:
        idx = lt.topk(top_k, dim=-1).indices
        ls = ls.gather(-1, idx); lt = lt.gather(-1, idx)
    logp_s = F.log_softmax(ls, -1); logp_t = F.log_softmax(lt, -1)
    p_s, p_t = logp_s.exp(), logp_t.exp()
    if kind == "forward":      # KL(p_T || p_S): mode covering
        d = (p_t * (logp_t - logp_s)).sum(-1)
    elif kind == "reverse":    # KL(p_S || p_T): mode seeking (student concentrates on teacher-preferred modes at this state)
        d = (p_s * (logp_s - logp_t)).sum(-1)
    else:                      # JSD
        logm = torch.logsumexp(torch.stack([logp_s, logp_t]), 0) - math.log(2)
        d = 0.5 * (p_s * (logp_s - logm)).sum(-1) + 0.5 * (p_t * (logp_t - logm)).sum(-1)
    return d.mean()


def frames_of(rec, theta, n_low, n_uniform, rng):
    """Sample frame steps t for one canvas. A step t is 'low' if some token written at step t+1 had commit_conf < theta."""
    cs, cc = rec["commit_step"], rec["commit_conf"]
    steps = sorted(set(cs))
    low = sorted({s - 1 for s, c in zip(cs, cc) if c < theta and s - 1 >= 0})
    cand = [s for s in steps if s < max(steps)]        # a frame must have a next step
    out = []
    if low:
        out += [rng.choice(low) for _ in range(n_low)]
    out += [rng.choice(cand) for _ in range(n_uniform + (0 if low else n_low))]
    return out


def make_state(rec, mask_id, corruption, t, rng, max_len):
    """Returns (ids, loss_positions) for one frame. replay: rewind the trajectory to step t, loss on what the sampler wrote at the next step."""
    P = rec["prompt_len"]; resp = rec["resp_ids"]; L = len(resp)
    ids = list(rec["prompt_ids"]) + resp
    if len(ids) > max_len:
        return None
    cs = rec["commit_step"]
    if corruption == "replay":
        nxt = min(s for s in cs if s > t)
        masked = [P + j for j in range(L) if cs[j] > t]
        pos = [P + j for j in range(L) if cs[j] == nxt]
    elif corruption == "suffix":    # the STEP3-prelim corruption: prefix visible, everything to the right masked, loss on the first 4 masks
        c = rng.randint(1, L - 1) if L > 2 else 1
        masked = list(range(P + c, P + L)); pos = masked[:4]
    else:                           # random i.i.d. masking at ratio r ~ U(0,1) (control C4); loss on the 4 masked positions the student is most sure of
        r = rng.random()
        masked = [P + j for j in range(L) if rng.random() < r] or [P + rng.randint(0, L - 1)]
        pos = None                  # chosen after the forward (needs the student's confidence)
    for q in masked:
        ids[q] = mask_id
    return ids, masked, pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=list(ARMS), required=True)
    ap.add_argument("--student", required=True, help="init checkpoint: Dream-org/Dream-v0-Instruct-7B or the best merge M(.4,.6)")
    ap.add_argument("--teacher", default=None, help="checkpoint dir; required for kd arms")
    ap.add_argument("--rollouts", required=True, help="rollouts.jsonl from code/step3/rollout.py")
    ap.add_argument("--prompts", required=True, help="prompts.jsonl (prompt text -> prompt_ids via the chat template)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--corruption", choices=["replay", "suffix", "random"], default="replay")
    ap.add_argument("--anchor", choices=["self", "ce", "none"], default="self", help="loss on the SURE next positions: KL to the untouched student, CE on its own token, or nothing")
    ap.add_argument("--kl", choices=["reverse", "forward", "jsd"], default="reverse",
                    help="reverse = KL(p_S||p_T) (TOPD's ablation: reverse > JSD > forward for dLLM distillation); forward = KL(p_T||p_S)")
    ap.add_argument("--top-k", type=int, default=0, help="truncate the teacher distribution to its top-k and renormalise (0 = full vocabulary)")
    ap.add_argument("--kd-temp", type=float, default=1.0, help="temperature applied to both logits before the divergence")
    ap.add_argument("--theta", type=float, default=0.8)
    ap.add_argument("--theta-end", type=float, default=None, help="linear schedule of the selection threshold from --theta to this value over training (user 2026-10-15: distil less and less); default = constant")
    ap.add_argument("--lambda-anchor", type=float, default=0.3, help="weight of the sure-position anchor term (means are taken separately)")
    ap.add_argument("--low-frames", type=int, default=2, help="frames per canvas per epoch sampled right before a low-confidence commit")
    ap.add_argument("--uniform-frames", type=int, default=1, help="frames per canvas per epoch sampled uniformly over steps")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=32, help="frames per optimizer step (micro-batch is 1)")
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
        if r["resp_len"] < 4 or len(set(r["commit_step"])) < 2:
            continue
        r["prompt_ids"] = tok.apply_chat_template([{"role": "user", "content": ptext[r["id"]]}], add_generation_prompt=True)
        assert len(r["prompt_ids"]) == r["prompt_len"], (r["id"], len(r["prompt_ids"]), r["prompt_len"])
        recs.append(r)
        if a.n_canvases and len(recs) >= a.n_canvases:
            break
    fpc = a.low_frames + a.uniform_frames
    print(f"[data] {len(recs)} canvases, {fpc} frames each ({a.low_frames} before a low-confidence commit), {a.epochs} epochs", flush=True)

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
    n_frames = len(recs) * fpc * a.epochs
    total_steps = max(1, n_frames // a.batch)
    if a.max_steps:
        total_steps = min(total_steps, a.max_steps)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, int(0.1 * total_steps))) * max(0.0, 1 - s / total_steps))
    log = open(out / "train_log.jsonl", "a")
    zero = lambda: dict(n=0, n_pos=0, n_unsure=0, n_empty=0, kd=0.0, anchor=0.0, ce=0.0)
    step = 0; acc = zero(); t0 = time.time(); student.train(); done = False
    for ep in range(a.epochs):
        order = [(i, t) for i, r in enumerate(recs) for t in frames_of(r, a.theta, a.low_frames, a.uniform_frames, rng)]
        rng.shuffle(order)
        for i, t in order:
            st = make_state(recs[i], MASK, a.corruption, t, rng, a.max_len)
            if st is None:
                continue
            ids, masked, pos = st
            x = torch.tensor([ids], device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ls_all = shifted_logits(student, x)[0].float()
            if pos is None:   # random-mask control: the 4 masked positions the student is most confident about = what it would write next
                conf_m = F.softmax(ls_all[masked].detach(), -1).max(-1).values
                pos = [masked[k] for k in conf_m.topk(min(4, len(masked))).indices.tolist()]
            ls = ls_all[pos]
            logp_s = F.log_softmax(ls, -1)
            gold = torch.tensor(recs[i]["prompt_ids"] + recs[i]["resp_ids"], device="cuda")[pos]
            ce = F.nll_loss(logp_s, gold)
            acc["n_pos"] += len(pos); acc["ce"] += float(ce)
            if arm["target"] == "ce":
                loss = ce
            else:
                conf_s = logp_s.detach().exp().max(-1).values           # raw T=1 confidence = the gate's signal
                th = a.theta + (a.theta_end - a.theta) * min(1.0, step / total_steps) if a.theta_end is not None else a.theta
                unsure = conf_s < th if arm["select"] else torch.ones_like(conf_s, dtype=torch.bool)
                acc["n_unsure"] += int(unsure.sum())
                loss = ls.sum() * 0.0
                if unsure.any():
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        lt = shifted_logits(teacher, x)[0, pos].float()
                    kd = divergence(ls[unsure], lt[unsure], a.kl, a.top_k, a.kd_temp)
                    loss = loss + kd; acc["kd"] += float(kd)
                else:
                    acc["n_empty"] += 1
                sure = ~unsure
                if sure.any() and a.anchor != "none":
                    if a.anchor == "self":
                        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16), student.disable_adapter():
                            l0 = shifted_logits(student, x)[0, pos].float()
                        an = divergence(ls[sure], l0[sure], "reverse", 0, 1.0)
                    else:
                        an = F.nll_loss(logp_s[sure], gold[sure])
                    loss = loss + a.lambda_anchor * an; acc["anchor"] += float(an)
            acc["n"] += 1
            (loss / a.batch).backward()
            if acc["n"] % a.batch == 0:
                torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad], 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
                rec = dict(step=step, epoch=ep, kd=acc["kd"] / a.batch, anchor=acc["anchor"] / a.batch, ce=acc["ce"] / a.batch,
                           pos_per_frame=acc["n_pos"] / a.batch, unsure_rate=acc["n_unsure"] / max(1, acc["n_pos"]), empty_frames=acc["n_empty"],
                           theta=(a.theta + (a.theta_end - a.theta) * min(1.0, step / total_steps) if a.theta_end is not None else a.theta),
                           lr=sched.get_last_lr()[0], s_per_frame=(time.time() - t0) / (step * a.batch))
                log.write(json.dumps(rec) + "\n"); log.flush()
                if step % 10 == 1 or step == total_steps:
                    print("[train]", json.dumps(rec), flush=True)
                acc = zero()
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
