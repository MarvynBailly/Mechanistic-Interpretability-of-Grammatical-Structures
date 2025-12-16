"""
Full IFR (Indirect Effect via Residual) head activation frequencies.

For each example and head:
  baseline = logit_diff(clean prompt)
  IFR-ablated = logit_diff(clean prompt with head residual contribution removed)
  delta = baseline - IFR-ablated

Active(head) = delta > tau   (or |delta| > tau if --abs)
Frequency = fraction of examples where Active(head)

How to run this code, with an example below:
python3 correlation_analysis/activation_frequencies_ifr.py \
    --model bigscience/bloom-560m \ #model name, could be gpt2-small as well
    --dataset data_generation/output/english_ioi_pairs_small.json \ # path to the generated dataset. Look at Token helpers to see which fields are being loaded from the dataset.
    --out results/correlation/actfreq_en_ifr_bloom_mps.npy \ #output npy path
    --abs \ #use absolute value of delta, on if this flag is provided
    --tau 0.03 \ # threshold value on delta
    --device cuda # device to run the model on, could be cpu as well

Example usages:

python3 correlation_analysis/activation_frequencies_ifr.py --model bigscience/bloom-560m --dataset data_generation/output/english_ioi_pairs_small.json --out results/correlation/actfreq_en_ifr_bloom_mps.npy --abs --tau 0.03 --device cuda

python3 correlation_analysis/activation_frequencies_ifr.py --model bigscience/bloom-560m --dataset data_generation/output/chinese_ioi_pairs_small.json --out results/correlation/actfreq_zh_ifr_bloom_mps.npy --abs --tau 0.03 --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformer_lens import HookedTransformer


# -------------------------
# Dataset loading
# -------------------------

def load_dataset(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict) and "pairs" in obj:
        return obj["pairs"]
    if isinstance(obj, list):
        return obj
    raise ValueError("Unsupported dataset format")


# -------------------------
# Token helpers
# -------------------------

def get_token_id(model: HookedTransformer, text: str) -> int:
    toks = model.to_tokens(text, prepend_bos=False)[0]
    return int(toks[-1])


def get_io_s_ids(model: HookedTransformer, ex: Dict[str, Any]) -> Tuple[int, int]:
    if "io_token" in ex and "s_name" in ex:
        io = get_token_id(model, ex["io_token"])
        s = get_token_id(model, " " + ex["s_name"])
        return io, s
    raise ValueError("Missing IO/S tokens")


# -------------------------
# IFR hook
# -------------------------

def make_ifr_hook(
    layer: int,
    head: int,
    head_resid: torch.Tensor,
):
    """
    Subtract a head's residual contribution at resid_post.
    head_resid: [B, d_model]
    """
    def hook(resid_post: torch.Tensor, hook):
        return resid_post - head_resid
    return hook


# -------------------------
# Core computation
# -------------------------

@dataclass
class IFRConfig:
    model: str
    dataset: str
    out: str
    meta: Optional[str]
    tau: float
    use_abs: bool
    batch_size: int
    device: str


@torch.no_grad()
def compute_ifr_frequencies(cfg: IFRConfig):
    model = HookedTransformer.from_pretrained(cfg.model, device=cfg.device)
    model.eval()

    data = load_dataset(cfg.dataset)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    d_model = model.cfg.d_model

    active = torch.zeros((n_layers, n_heads), device=cfg.device)
    total = 0

    for ex in data:
        toks = model.to_tokens(ex["clean"], prepend_bos=True)
        io_id, s_id = get_io_s_ids(model, ex)

        # Baseline
        logits, cache = model.run_with_cache(toks)
        pos = toks.shape[1] - 1
        baseline = logits[0, pos, io_id] - logits[0, pos, s_id]

        # Extract head residuals
        for L in range(n_layers):
            z = cache[f"blocks.{L}.attn.hook_z"]        # [1,S,H,d_head]
            W_O = model.W_O[L]                           # [H,d_head,d_model]
            head_resid = torch.einsum("shd,hdm->shm", z[0], W_O)  # [S,H,d_model]
            head_resid = head_resid[pos]                 # [H,d_model]

            for h in range(n_heads):
                hook = make_ifr_hook(L, h, head_resid[h:h+1])
                ablated = model.run_with_hooks(
                    toks,
                    fwd_hooks=[(f"blocks.{L}.hook_resid_post", hook)],
                    return_type="logits",
                )
                delta = baseline - (ablated[0, pos, io_id] - ablated[0, pos, s_id])

                if cfg.use_abs:
                    is_active = delta.abs() > cfg.tau
                else:
                    is_active = delta > cfg.tau

                active[L, h] += float(is_active)

        total += 1
        if total % 10 == 0:
            print(f"Processed {total}/{len(data)}")

    freq = (active / total).cpu().numpy()
    return freq


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gpt2-small")
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--meta", default=None)
    p.add_argument("--tau", type=float, default=0.1)
    p.add_argument("--abs", action="store_true")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    cfg = IFRConfig(
        model=args.model,
        dataset=args.dataset,
        out=args.out,
        meta=args.meta,
        tau=args.tau,
        use_abs=args.abs,
        batch_size=args.batch_size,
        device=args.device,
    )

    freq = compute_ifr_frequencies(cfg)
    np.save(args.out, freq)

    if args.meta:
        with open(args.meta, "w") as f:
            json.dump(vars(cfg), f, indent=2)

    print("Saved IFR activation frequencies:", args.out)


if __name__ == "__main__":
    main()
