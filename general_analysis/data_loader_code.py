"""
Data loader for Python code completion task.

Loads code pairs and prepares them for path patching experiments.
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple
import torch
from transformer_lens import HookedTransformer


def get_code_dataset_path(var_type: str = "colors") -> Path:
    """Get path to code completion dataset file."""
    current_dir = Path(__file__).parent
    base_dir = current_dir.parent / "data_generation_color" / "output"
    return base_dir / f"code_pairs_{var_type}.json"


def load_code_pairs(dataset_path: Path) -> List[Dict]:
    """Load code completion pairs from JSON file."""
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def prepare_code_batch_for_patching(
    model: HookedTransformer,
    pairs: List[Dict],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepare a batch of code pairs for path patching.
    
    Returns:
        Tuple of (clean_tokens, corrupt_tokens, correct_toks, incorrect_toks)
    """
    device = model.cfg.device
    
    # Extract prompts and answers
    clean_prompts = [pair['clean'] for pair in pairs]
    corrupt_prompts = [pair['corrupt'] for pair in pairs]
    correct_args = [pair['correct_arg'] for pair in pairs]
    incorrect_args = [pair['incorrect_arg'] for pair in pairs]
    
    # Tokenize prompts
    clean_tokens = model.to_tokens(clean_prompts, prepend_bos=True)
    corrupt_tokens = model.to_tokens(corrupt_prompts, prepend_bos=True)
    
    # Get token IDs for arguments (with space prefix since mid-sentence)
    correct_toks = torch.tensor([
        model.to_single_token(" " + arg) for arg in correct_args
    ], device=device)
    
    incorrect_toks = torch.tensor([
        model.to_single_token(" " + arg) for arg in incorrect_args
    ], device=device)
    
    return clean_tokens, corrupt_tokens, correct_toks, incorrect_toks


def load_code_dataset_for_patching(
    model: HookedTransformer,
    var_type: str = "colors",
    n_examples: int = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[Dict]]:
    """
    Load and prepare code completion dataset for path patching.
    
    Args:
        model: The transformer model
        var_type: Variable type - 'colors', 'objects', or 'letters'
        n_examples: Optional number of examples to use
        
    Returns:
        Tuple of (clean_tokens, corrupt_tokens, correct_toks, incorrect_toks, pairs)
    """
    dataset_path = get_code_dataset_path(var_type)
    
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}\n"
            f"Please run: python data_generation_color/generate_code_pairs.py"
        )
    
    # Load pairs
    pairs = load_code_pairs(dataset_path)
    
    # Subset if requested
    if n_examples is not None:
        pairs = pairs[:n_examples]
    
    print(f"Loaded {len(pairs)} code pairs ({var_type}) from {dataset_path.name}")
    
    # Prepare for patching
    clean_tokens, corrupt_tokens, correct_toks, incorrect_toks = \
        prepare_code_batch_for_patching(model, pairs)
    
    return clean_tokens, corrupt_tokens, correct_toks, incorrect_toks, pairs


def compute_code_logit_diff(
    model: HookedTransformer,
    tokens: torch.Tensor,
    correct_toks: torch.Tensor,
    incorrect_toks: torch.Tensor,
) -> float:
    """Compute average logit difference for code completion task."""
    device = model.cfg.device
    batch_idx = torch.arange(len(correct_toks), device=device)
    
    with torch.inference_mode():
        logits = model(tokens)[:, -1, :]
        correct_logits = logits[batch_idx, correct_toks]
        incorrect_logits = logits[batch_idx, incorrect_toks]
        logit_diff = (correct_logits - incorrect_logits).mean().item()
    
    return logit_diff


def analyze_code_accuracy(
    model: HookedTransformer,
    clean_tokens: torch.Tensor,
    corrupt_tokens: torch.Tensor,
    correct_toks: torch.Tensor,
    incorrect_toks: torch.Tensor,
) -> Dict[str, float]:
    """Analyze model performance on code completion task."""
    device = model.cfg.device
    batch_idx = torch.arange(len(correct_toks), device=device)
    
    with torch.inference_mode():
        # Clean performance
        clean_logits = model(clean_tokens)[:, -1, :]
        clean_correct = clean_logits[batch_idx, correct_toks]
        clean_incorrect = clean_logits[batch_idx, incorrect_toks]
        clean_diff = (clean_correct - clean_incorrect).mean().item()
        
        # Accuracy: is correct arg the top prediction?
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
