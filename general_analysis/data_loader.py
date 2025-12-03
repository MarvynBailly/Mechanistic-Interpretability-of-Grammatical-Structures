"""
Data loading utilities for color-object association path patching experiments.

This module provides functions to load color-object pairs from the data_generation_color
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
    Get path to the color-object dataset file.
    
    Args:
        size: Dataset size - 'small', 'medium', or 'large'
        
    Returns:
        Path to the dataset JSON file
    """
    current_dir = Path(__file__).parent
    base_dir = current_dir.parent / "data_generation" / "output"
    return base_dir / f"color_pairs_{size}.json"


def load_color_pairs(dataset_path: Path) -> List[Dict]:
    """
    Load color-object pairs from JSON file.
    
    Args:
        dataset_path: Path to the JSON dataset file
        
    Returns:
        List of pair dictionaries with 'clean', 'corrupt', 'correct_object', etc.
    """
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def prepare_batch_for_patching(
    model: HookedTransformer,
    pairs: List[Dict],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepare a batch of color-object pairs for path patching.
    
    This function:
    1. Tokenizes clean and corrupt prompts
    2. Extracts correct and incorrect object token IDs
    3. Returns batched tensors ready for path patching
    
    Args:
        model: The transformer model
        pairs: List of pair dictionaries from load_color_pairs()
        
    Returns:
        Tuple of (clean_tokens, corrupt_tokens, correct_toks, incorrect_toks)
        - clean_tokens: [batch, seq_len] tokenized clean prompts
        - corrupt_tokens: [batch, seq_len] tokenized corrupt prompts
        - correct_toks: [batch] token IDs for correct objects
        - incorrect_toks: [batch] token IDs for incorrect objects
    """
    device = model.cfg.device
    
    # Extract prompts and answers
    clean_prompts = [pair['clean'] for pair in pairs]
    corrupt_prompts = [pair['corrupt'] for pair in pairs]
    correct_objects = [pair['correct_object'] for pair in pairs]
    incorrect_objects = [pair['incorrect_object'] for pair in pairs]
    
    # Tokenize prompts
    clean_tokens = model.to_tokens(clean_prompts, prepend_bos=True)
    corrupt_tokens = model.to_tokens(corrupt_prompts, prepend_bos=True)
    
    # Get token IDs for objects (prepend space since these are mid-sentence tokens)
    correct_toks = torch.tensor([
        model.to_single_token(" " + obj) for obj in correct_objects
    ], device=device)
    
    incorrect_toks = torch.tensor([
        model.to_single_token(" " + obj) for obj in incorrect_objects
    ], device=device)
    
    return clean_tokens, corrupt_tokens, correct_toks, incorrect_toks


def load_dataset_for_patching(
    model: HookedTransformer,
    size: str = "small",
    n_examples: int = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[Dict]]:
    """
    Load and prepare color-object dataset for path patching experiments.
    
    This is the main entry point for loading data.
    
    Args:
        model: The transformer model
        size: Dataset size - 'small' (100), 'medium' (500), or 'large' (1000)
        n_examples: Optional number of examples to use (defaults to all)
        
    Returns:
        Tuple of (clean_tokens, corrupt_tokens, correct_toks, incorrect_toks, pairs)
        where pairs is the original data for reference
        
    Example:
        >>> model = HookedTransformer.from_pretrained("gpt2-small")
        >>> clean, corrupt, correct, incorrect, pairs = load_dataset_for_patching(
        ...     model, size="small", n_examples=50
        ... )
        >>> # Now ready for path patching
    """
    dataset_path = get_dataset_path(size)
    
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}\n"
            f"Please run: cd data_generation && python generate_color_pairs.py"
        )
    
    # Load pairs
    pairs = load_color_pairs(dataset_path)
    
    # Subset if requested
    if n_examples is not None:
        pairs = pairs[:n_examples]
    
    print(f"Loaded {len(pairs)} color-object pairs from {dataset_path.name}")
    
    # Prepare for patching
    clean_tokens, corrupt_tokens, correct_toks, incorrect_toks = \
        prepare_batch_for_patching(model, pairs)
    
    return clean_tokens, corrupt_tokens, correct_toks, incorrect_toks, pairs


def compute_logit_diff(
    model: HookedTransformer,
    tokens: torch.Tensor,
    correct_toks: torch.Tensor,
    incorrect_toks: torch.Tensor,
) -> float:
    """
    Compute average logit difference for color-object task.
    
    logit_diff = logit(correct_object) - logit(incorrect_object)
    
    Args:
        model: The transformer model
        tokens: Input tokens [batch, seq_len]
        correct_toks: Correct object token IDs [batch]
        incorrect_toks: Incorrect object token IDs [batch]
        
    Returns:
        Average logit difference across the batch
    """
    device = model.cfg.device
    batch_idx = torch.arange(len(correct_toks), device=device)
    
    with torch.inference_mode():
        logits = model(tokens)[:, -1, :]  # Get last position logits
        correct_logits = logits[batch_idx, correct_toks]
        incorrect_logits = logits[batch_idx, incorrect_toks]
        logit_diff = (correct_logits - incorrect_logits).mean().item()
    
    return logit_diff


def analyze_dataset_accuracy(
    model: HookedTransformer,
    clean_tokens: torch.Tensor,
    corrupt_tokens: torch.Tensor,
    correct_toks: torch.Tensor,
    incorrect_toks: torch.Tensor,
) -> Dict[str, float]:
    """
    Analyze model performance on clean and corrupt prompts.
    
    Returns statistics about:
    - Clean logit difference (should be high)
    - Corrupt logit difference (should be low/~0)
    - Accuracy on clean (correct object has highest logit?)
    
    Args:
        model: The transformer model
        clean_tokens: Clean input tokens [batch, seq_len]
        corrupt_tokens: Corrupt input tokens [batch, seq_len]
        correct_toks: Correct object token IDs [batch]
        incorrect_toks: Incorrect object token IDs [batch]
        
    Returns:
        Dictionary with performance metrics
    """
    device = model.cfg.device
    batch_idx = torch.arange(len(correct_toks), device=device)
    
    with torch.inference_mode():
        # Clean performance
        clean_logits = model(clean_tokens)[:, -1, :]
        clean_correct = clean_logits[batch_idx, correct_toks]
        clean_incorrect = clean_logits[batch_idx, incorrect_toks]
        clean_diff = (clean_correct - clean_incorrect).mean().item()
        
        # Accuracy: is correct object the top prediction?
        clean_preds = clean_logits.argmax(dim=-1)
        clean_acc = (clean_preds == correct_toks).float().mean().item()
        
        # Corrupt performance
        corrupt_logits = model(corrupt_tokens)[:, -1, :]
        corrupt_correct = corrupt_logits[batch_idx, correct_toks]
        corrupt_incorrect = corrupt_logits[batch_idx, incorrect_toks]
        corrupt_diff = (corrupt_correct - corrupt_incorrect).mean().item()
        
        corrupt_preds = corrupt_logits.argmax(dim=-1)
        corrupt_acc = (corrupt_preds == correct_toks).float().mean().item()
    
    return {
        'clean_logit_diff': clean_diff,
        'corrupt_logit_diff': corrupt_diff,
        'clean_accuracy': clean_acc,
        'corrupt_accuracy': corrupt_acc,
        'corruption_effect': clean_diff - corrupt_diff,
    }
