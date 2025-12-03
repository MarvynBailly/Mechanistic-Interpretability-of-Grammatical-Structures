"""
Data loading utilities for path patching experiments.

This module provides functions to load IOI pairs from the data_generation_simple
dataset and prepare them for path patching experiments.
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple
import torch
from transformer_lens import HookedTransformer


# ============================================================================
# Data Loading
# ============================================================================

def get_dataset_path(size: str = "small") -> Path:
    """
    Get path to the IOI dataset file.
    
    Args:
        size: Dataset size - 'small', 'medium', or 'large'
        
    Returns:
        Path to the dataset JSON file
        
    Note:
        Assumes data_generation_simple/output/ is at the same level as path_patching_full
    """
    # Navigate up to parent directory, then into data_generation
    current_dir = Path(__file__).parent
    base_dir = current_dir.parent / "data_generation" / "output"
    return base_dir / f"ioi_pairs_{size}.json"


def load_ioi_pairs(dataset_path: Path) -> List[Dict]:
    """
    Load IOI pairs from JSON file.
    
    Args:
        dataset_path: Path to the JSON dataset file
        
    Returns:
        List of pair dictionaries with 'clean', 'corrupt', 'io_name', etc.
    """
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['pairs']


def prepare_batch_for_patching(
    model: HookedTransformer,
    pairs: List[Dict],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepare a batch of IOI pairs for path patching.
    
    This function:
    1. Tokenizes clean and corrupt prompts
    2. Extracts IO and S token IDs
    3. Returns batched tensors ready for path patching
    
    Args:
        model: The transformer model
        pairs: List of pair dictionaries from load_ioi_pairs()
        
    Returns:
        Tuple of (clean_tokens, corrupt_tokens, io_toks, s_toks)
        - clean_tokens: [batch, seq_len] tokenized clean prompts
        - corrupt_tokens: [batch, seq_len] tokenized corrupt prompts
        - io_toks: [batch] token IDs for IO names
        - s_toks: [batch] token IDs for S names
        
    Example:
        >>> pairs = load_ioi_pairs(get_dataset_path("small"))
        >>> clean, corrupt, io, s = prepare_batch_for_patching(model, pairs[:50])
        >>> # Now ready for path patching
    """
    clean_prompts = []
    corrupt_prompts = []
    io_token_ids = []
    s_token_ids = []
    
    for pair in pairs:
        clean_prompts.append(pair['clean'])
        corrupt_prompts.append(pair['corrupt'])
        
        # Get token IDs for IO and S names
        # The dataset stores io_token with leading space already
        io_name = pair['io_token']
        s_name = f" {pair['s_name']}"
        
        io_tok = model.to_single_token(io_name)
        s_tok = model.to_single_token(s_name)
        
        assert io_tok is not None, f"IO name '{io_name}' is not a single token"
        assert s_tok is not None, f"S name '{s_name}' is not a single token"
        
        io_token_ids.append(io_tok)
        s_token_ids.append(s_tok)
    
    # Tokenize prompts
    clean_tokens = model.to_tokens(clean_prompts, prepend_bos=False).to(model.cfg.device)
    corrupt_tokens = model.to_tokens(corrupt_prompts, prepend_bos=False).to(model.cfg.device)
    
    # Convert token IDs to tensors
    io_toks = torch.tensor(io_token_ids, device=model.cfg.device)
    s_toks = torch.tensor(s_token_ids, device=model.cfg.device)
    
    return clean_tokens, corrupt_tokens, io_toks, s_toks


def load_dataset_for_patching(
    model: HookedTransformer,
    size: str = "small",
    n_examples: int = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[Dict]]:
    """
    Convenience function to load and prepare dataset in one step.
    
    Args:
        model: The transformer model
        size: Dataset size - 'small', 'medium', or 'large'
        n_examples: Number of examples to use (None = use all)
        
    Returns:
        Tuple of (clean_tokens, corrupt_tokens, io_toks, s_toks, pairs)
        - Tensors ready for path patching
        - pairs: Original pair data for reference
        
    Example:
        >>> model = HookedTransformer.from_pretrained("gpt2-small")
        >>> clean, corrupt, io, s, pairs = load_dataset_for_patching(
        ...     model, size="small", n_examples=50
        ... )
        >>> # Ready to use with path_patch()
    """
    # Load dataset
    dataset_path = get_dataset_path(size)
    pairs = load_ioi_pairs(dataset_path)
    
    # Limit number of examples if requested
    if n_examples is not None:
        pairs = pairs[:n_examples]
    
    # Prepare for patching
    clean_tokens, corrupt_tokens, io_toks, s_toks = prepare_batch_for_patching(
        model, pairs
    )
    
    print(f"Loaded {len(pairs)} IOI pairs from {dataset_path.name}")
    print(f"Clean tokens shape: {clean_tokens.shape}")
    print(f"Corrupt tokens shape: {corrupt_tokens.shape}")
    
    return clean_tokens, corrupt_tokens, io_toks, s_toks, pairs


# ============================================================================
# Data Inspection
# ============================================================================

def print_pair_examples(pairs: List[Dict], n: int = 3):
    """
    Print example pairs for inspection.
    
    Args:
        pairs: List of pair dictionaries
        n: Number of examples to print
    """
    print(f"\nShowing {min(n, len(pairs))} example pairs:")
    print("=" * 70)
    
    for i, pair in enumerate(pairs[:n]):
        print(f"\nPair {i+1}:")
        print(f"  Clean:   {pair['clean']}")
        print(f"  Corrupt: {pair['corrupt']}")
        print(f"  IO: {pair['io_name']} (token: {pair['io_token']})")
        print(f"  S:  {pair['s_name']}")
