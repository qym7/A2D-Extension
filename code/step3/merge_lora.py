"""merge_lora.py — fold a LoRA adapter into a Dream checkpoint and save a plain bf16 checkpoint loadable by every eval harness.
Used by train_frontier_kd.py at the end of training, and standalone to recover `merged/` from `lora/` (e.g. after a failed save):
  python merge_lora.py --student <base ckpt> --lora <out>/lora --out <out>/merged
Dream's DreamGenerationConfig.validate() has no `strict` kwarg, which transformers' save_pretrained calls → neutralised here, and the original
generation_config.json + remote-code *.py files are copied from the base snapshot instead. GPU/CPU job only (loads 7B weights)."""
import argparse, shutil
from pathlib import Path


def save_merged(model, tok, base, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    from transformers.utils.hub import cached_file
    src_dir = Path(cached_file(base, "config.json")).parent
    gc = model.generation_config
    if gc is not None:
        gc.validate = lambda *a, **k: None          # DreamGenerationConfig.validate(is_init) vs transformers' validate(strict=True)
    model.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    for f in list(src_dir.glob("*.py")) + [src_dir / "generation_config.json"]:
        if f.exists():
            shutil.copy(f, out / f.name)
    print(f"[merge] saved {out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True); ap.add_argument("--lora", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch
    from transformers import AutoModel, AutoTokenizer
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(a.student, trust_remote_code=True)
    m = AutoModel.from_pretrained(a.student, trust_remote_code=True, torch_dtype=torch.bfloat16)
    m = PeftModel.from_pretrained(m, a.lora).merge_and_unload()
    save_merged(m, tok, a.student, a.out)
