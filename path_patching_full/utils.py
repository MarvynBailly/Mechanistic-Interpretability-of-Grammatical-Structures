"""
Utility functions for IOI (Indirect Object Identification) experiments.

This module provides:
- Data structures for IOI examples
- Dataset generation utilities
- Model evaluation functions
- Helper functions for logit computation
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict
import random
import numpy as np
import torch
from transformer_lens import HookedTransformer


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class IOIExample:
    """
    Represents a single IOI example.
    
    Attributes:
        prompt: Text prompt up to "to" — next token should be IO name
        s_name: The subject name (S)
        io_name: The indirect object name (IO)
    """
    prompt: str
    s_name: str
    io_name: str


# ============================================================================
# Dataset Generation
# ============================================================================

def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sample_name_pair(names: List[str]) -> Tuple[str, str]:
    """
    Sample two different names from the list.
    
    Args:
        names: List of available names
        
    Returns:
        Tuple of (S, IO) as two different names
    """
    s, io = random.sample(names, 2)
    return s, io


def build_dataset(
    n: int,
    names: List[str],
    templates: List[str]
) -> List[IOIExample]:
    """
    Build a dataset of n IOI examples.
    
    Args:
        n: Number of examples to generate
        names: List of single-token names to sample from
        templates: List of template strings with {S}, {IO} placeholders
        
    Returns:
        List of IOIExample instances
        
    Note:
        Templates should have {IO} as first name and {S} as second name.
    """
    data: List[IOIExample] = []
    for _ in range(n):
        s, io = sample_name_pair(names)
        template = random.choice(templates)
        # IO is first name in template, S is second
        prompt = template.format(S=s, IO=io)
        data.append(IOIExample(prompt=prompt, s_name=s, io_name=io))
    return data


# ============================================================================
# Model Evaluation
# ============================================================================

def get_logits_for_next_token(
    model: HookedTransformer,
    texts: List[str],
    batch_size: int = 64,
) -> torch.Tensor:
    """
    Get logits for the next token after each text prompt.
    
    Args:
        model: The transformer model
        texts: List of text prompts
        batch_size: Batch size for processing (to avoid OOM)
        
    Returns:
        Tensor of shape [len(texts), vocab_size] with logits for next token
        
    Note:
        Uses batching to avoid GPU/CPU memory issues.
    """
    all_logits = []
    model_device = model.cfg.device

    with torch.inference_mode():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            tokens = model.to_tokens(batch_texts, prepend_bos=False).to(model_device)
            logits = model(tokens)          # [batch, seq, vocab]
            next_logits = logits[:, -1, :]  # [batch, vocab]
            all_logits.append(next_logits)

    return torch.cat(all_logits, dim=0)     # [total_batch, vocab]


def evaluate(
    model: HookedTransformer,
    dataset: List[IOIExample],
    batch_size: int = 64,
) -> Dict[str, float]:
    """
    Evaluate model performance on IOI dataset.
    
    Args:
        model: The transformer model
        dataset: List of IOI examples
        batch_size: Batch size for processing
        
    Returns:
        Dictionary with metrics:
        - accuracy: Proportion where logit(IO) > logit(S)
        - zero_rank: Proportion where IO is the top predicted token
        - logit_diff: Mean of logit(IO) - logit(S)
        - s_beats_io: Proportion where logit(S) > logit(IO)
    """
    prompts = [ex.prompt for ex in dataset]
    next_logits = get_logits_for_next_token(model, prompts, batch_size=batch_size)

    # Convert names to token IDs
    io_ids = torch.tensor(
        [model.to_single_token(ex.io_name) for ex in dataset],
        device=next_logits.device,
    )
    s_ids = torch.tensor(
        [model.to_single_token(ex.s_name) for ex in dataset],
        device=next_logits.device,
    )

    # Sanity check: all names should be single-token
    assert io_ids.min().item() >= 0, "IO names must be single tokens"
    assert s_ids.min().item() >= 0, "S names must be single tokens"

    # Extract logits for IO and S tokens
    batch_index = torch.arange(len(dataset), device=next_logits.device)
    io_logits = next_logits[batch_index, io_ids]
    s_logits = next_logits[batch_index, s_ids]

    # Compute metrics
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
    Compute mean logit difference between IO and S tokens.
    
    Args:
        model: The transformer model
        tokens: Token tensor of shape [batch, seq_len]
        s_toks: Token IDs for S names, shape [batch]
        io_toks: Token IDs for IO names, shape [batch]
        
    Returns:
        Mean of logit(IO) - logit(S) across the batch
        
    Note:
        Logits are extracted at the final position of each sequence.
    """
    logits = model(tokens)            # [batch, seq, vocab]
    next_logits = logits[:, -1, :]    # [batch, vocab]

    batch_idx = torch.arange(tokens.size(0), device=tokens.device)

    io_logits = next_logits[batch_idx, io_toks]   # [batch]
    s_logits  = next_logits[batch_idx, s_toks]    # [batch]

    return (io_logits - s_logits).mean().item()


# ============================================================================
# Token Position Helpers
# ============================================================================

def find_token_positions(
    tokens: torch.Tensor,
    target_tokens: torch.Tensor
) -> torch.Tensor:
    """
    Find positions of target tokens in each sequence.
    
    Args:
        tokens: Token sequences of shape [batch, seq_len]
        target_tokens: Target token IDs of shape [batch]
        
    Returns:
        Positions tensor of shape [batch] with position of first occurrence
        
    Note:
        Returns -1 if target token is not found in sequence.
    """
    batch_size, seq_len = tokens.shape
    positions = torch.zeros(batch_size, dtype=torch.long, device=tokens.device)
    
    for i in range(batch_size):
        # Find first occurrence of target token
        matches = (tokens[i] == target_tokens[i]).nonzero(as_tuple=True)[0]
        if len(matches) > 0:
            positions[i] = matches[0]
        else:
            positions[i] = -1  # Not found
            
    return positions
