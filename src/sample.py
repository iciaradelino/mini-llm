#!/usr/bin/env python3
"""
src/sample.py  (step 13)

Generate text from a checkpoint saved by train.py --save_checkpoint.

The model predicts a probability for each of the 8,192 tokens; we pick one,
append it, and repeat (autoregressive generation). Temperature and top-k control
how adventurous the picks are.

Usage
    python src/sample.py --checkpoint checkpoints/final.pt
    python src/sample.py --checkpoint checkpoints/final.pt --prompt "Sancho dijo" \
        --num_samples 3 --max_new_tokens 300 --temperature 0.8 --top_k 200
"""

import argparse
import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from train import GPT, GPTConfig

BASE_DIR = Path(__file__).resolve().parent.parent


@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature, top_k, eot_id):
    """Extend idx (shape 1 x T) one token at a time."""
    block = model.cfg.block_size
    for _ in range(max_new_tokens):
        context = idx[:, -block:]                     # the model only sees the last 256 tokens
        logits, _ = model(context)
        logits = logits[:, -1, :]                     # scores for the NEXT token only
        if temperature == 0:                          # greedy: always the most likely token
            next_id = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k:                                 # keep only the k best candidates
                kth_best = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
                logits[logits < kth_best] = float("-inf")
            probs = F.softmax(logits, dim=-1)         # scores -> probabilities
            next_id = torch.multinomial(probs, num_samples=1)  # draw one at random
        idx = torch.cat([idx, next_id], dim=1)
        if next_id.item() == eot_id:                  # the model "ended the book"
            break
    return idx


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help=".pt file from train.py --save_checkpoint")
    p.add_argument("--tokenizer", default=str(BASE_DIR / "data" / "tokenized" / "tokenizer.json"))
    p.add_argument("--prompt", default="En un lugar de la Mancha")
    p.add_argument("--num_samples", type=int, default=3)
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8, help="0 = greedy")
    p.add_argument("--top_k", type=int, default=200, help="0 = no limit")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()

    ckpt = torch.load(a.checkpoint, map_location=a.device)
    model = GPT(GPTConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model"])
    model.to(a.device).eval()

    tok = Tokenizer.from_file(a.tokenizer)
    tok_sha = hashlib.sha256(Path(a.tokenizer).read_bytes()).hexdigest()
    if ckpt.get("tokenizer_sha256") and ckpt["tokenizer_sha256"] != tok_sha:
        print("[WARN] this tokenizer is not the one the model was trained with; output will be garbage")
    eot_id = tok.token_to_id("<|endoftext|>")

    print(f"[INFO] checkpoint step {ckpt['step']}, val loss {ckpt['val_loss']:.3f} | "
          f"temperature {a.temperature} | top_k {a.top_k} | seed {a.seed}")
    torch.manual_seed(a.seed)
    prompt_ids = torch.tensor([tok.encode(a.prompt).ids], device=a.device)
    for i in range(a.num_samples):
        out = generate(model, prompt_ids, a.max_new_tokens, a.temperature, a.top_k, eot_id)
        text = tok.decode(out[0].tolist())
        print(f"\n---------------- sample {i + 1} ----------------\n{text}")


if __name__ == "__main__":
    main()