from dataclasses import dataclass
from typing import List, Tuple, Dict
import random
import numpy as np
import torch
from transformer_lens import HookedTransformer


@dataclass
class IOIExample:
    prompt: str     # up to the "to" — the next token should be IO
    s_name: str
    io_name: str


# ------------- Utils -------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sample_name_pair(names: List[str]) -> Tuple[str, str]:
    """Return (S, IO) as two different names."""
    s, io = random.sample(names, 2)
    return s, io


def build_dataset(n: int) -> List[IOIExample]:
    data: List[IOIExample] = []
    for _ in range(n):
        s, io = sample_name_pair(ONE_TOKEN_NAMES)
        template = random.choice(TEMPLATES)
        # IO is first name in template, S is second
        prompt = template.format(S=s, IO=io)
        data.append(IOIExample(prompt=prompt, s_name=s, io_name=io))
    return data

def get_logits_for_next_token(
    model: HookedTransformer,
    texts: List[str],
    batch_size: int = 64,
) -> torch.Tensor:
    """
    Returns final logits for the next token after each text.
    Shape: [batch, vocab]

    We do this in batches to avoid GPU/MPS OOM.
    """
    all_logits = []

    model_device = model.cfg.device

    with torch.inference_mode():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            tokens = model.to_tokens(batch_texts, prepend_bos=False).to(model_device)
            logits = model(tokens)          # [b, seq, vocab]
            next_logits = logits[:, -1, :]  # [b, vocab]
            all_logits.append(next_logits)

    return torch.cat(all_logits, dim=0)     # [B, vocab]


def evaluate(
    model: HookedTransformer,
    dataset: List[IOIExample],
    batch_size: int = 64,
) -> Dict[str, float]:
    """
    Batched evaluation to avoid OOM.
    """
    prompts = [ex.prompt for ex in dataset]
    next_logits = get_logits_for_next_token(model, prompts, batch_size=batch_size)  # [B, V]

    # Convert names to token ids per example
    io_ids = torch.tensor(
        [model.to_single_token(ex.io_name) for ex in dataset],
        device=next_logits.device,
    )
    s_ids = torch.tensor(
        [model.to_single_token(ex.s_name) for ex in dataset],
        device=next_logits.device,
    )

    # Sanity check: all names should be single-token
    assert io_ids.min().item() >= 0
    assert s_ids.min().item() >= 0

    batch_index = torch.arange(len(dataset), device=next_logits.device)
    io_logits = next_logits[batch_index, io_ids]
    s_logits = next_logits[batch_index, s_ids]

    acc = (io_logits > s_logits).float().mean().item()
    zero_rank = (next_logits.argmax(dim=-1) == io_ids).float().mean().item()
    logit_diff = (io_logits - s_logits).mean().item()
    s_beats_io = (s_logits > io_logits).float().mean().item()

    print(f"Mean logit diff (IO - S): {logit_diff:.3f}")
    print(f"Proportion S logit > IO : {s_beats_io:.3f}")

    return {
        "accuracy": acc,
        "zero_rank": zero_rank,
        "logit_diff": logit_diff,
        "s_beats_io": s_beats_io,
    }

def logit_diff_io_s_from_tokens(
    model: HookedTransformer,
    tokens: torch.Tensor,
    s_toks: torch.Tensor,
    io_toks: torch.Tensor,
) -> float:
    """
    Compute mean logit(IO) - logit(S) at END position for a batch of tokens.
    
    tokens : [B, seq]
    s_toks : [B]   token ids for S for each example
    io_toks: [B]   token ids for IO for each example
    """
    logits = model(tokens)            # [B, seq, vocab]
    next_logits = logits[:, -1, :]    # [B, vocab]

    batch_idx = torch.arange(tokens.size(0), device=tokens.device)

    io_logits = next_logits[batch_idx, io_toks]   # shape [B]
    s_logits  = next_logits[batch_idx, s_toks]    # shape [B]

    return (io_logits - s_logits).mean().item()
