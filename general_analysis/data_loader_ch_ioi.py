"""
Data loader for Chinese IOI (Indirect Object Identification) task.

Loads Chinese IOI pairs and prepares them for path patching experiments.
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple
import torch
from transformer_lens import HookedTransformer


def get_ch_ioi_dataset_path(size: str = "small") -> Path:
    """Get path to Chinese IOI dataset file."""
    current_dir = Path(__file__).parent
    base_dir = current_dir.parent / "data_generation" / "output"
    return base_dir / f"chinese_ioi_pairs_{size}.json"


def load_ch_ioi_pairs(dataset_path: Path) -> List[Dict]:
    """Load Chinese IOI pairs from JSON file."""
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['pairs']  # Extract pairs from the JSON structure


def prepare_ch_ioi_batch_for_patching(
    model: HookedTransformer,
    pairs: List[Dict],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepare a batch of Chinese IOI pairs for path patching.
    
    Returns:
        Tuple of (clean_tokens, corrupt_tokens, io_toks)
        - clean_tokens: tokenized clean prompts
        - corrupt_tokens: tokenized corrupt prompts  
        - io_toks: token IDs for indirect object (correct answer)
    """
    device = model.cfg.device
    
    # Extract prompts and answers
    clean_prompts = [pair['clean'] for pair in pairs]
    corrupt_prompts = [pair['corrupt'] for pair in pairs]
    io_tokens = [pair['io_token'] for pair in pairs]
    
    # Tokenize prompts
    clean_tokens = model.to_tokens(clean_prompts, prepend_bos=True)
    corrupt_tokens = model.to_tokens(corrupt_prompts, prepend_bos=True)
    
    # Get token IDs for IO names (they already include the space prefix)
    io_toks = torch.tensor([
        model.to_single_token(io_tok) for io_tok in io_tokens
    ], device=device)
    
    return clean_tokens, corrupt_tokens, io_toks


def load_ch_ioi_dataset_for_patching(
    model: HookedTransformer,
    size: str = "small",
    n_examples: int = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Dict]]:
    """
    Load and prepare Chinese IOI dataset for path patching.
    
    Args:
        model: The transformer model
        size: Dataset size - 'small', 'medium', or 'large'
        n_examples: Optional number of examples to use
        
    Returns:
        Tuple of (clean_tokens, corrupt_tokens, io_toks, pairs)
    """
    dataset_path = get_ch_ioi_dataset_path(size)
    
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}\n"
            f"Please run: python data_generation/generate_ch_ioi_pairs.py"
        )
    
    # Load pairs
    pairs = load_ch_ioi_pairs(dataset_path)
    
    # Subset if requested
    if n_examples is not None:
        pairs = pairs[:n_examples]
    
    print(f"Loaded {len(pairs)} Chinese IOI pairs ({size}) from {dataset_path.name}")
    
    # Prepare for patching
    clean_tokens, corrupt_tokens, io_toks = \
        prepare_ch_ioi_batch_for_patching(model, pairs)
    
    return clean_tokens, corrupt_tokens, io_toks, pairs


def compute_ch_ioi_logit_diff(
    model: HookedTransformer,
    tokens: torch.Tensor,
    io_toks: torch.Tensor,
) -> float:
    """
    Compute average logit for IO token (higher is better).
    
    Args:
        model: The transformer model
        tokens: Input token sequences
        io_toks: Token IDs for indirect objects
        
    Returns:
        Average logit for IO tokens
    """
    device = model.cfg.device
    batch_idx = torch.arange(len(io_toks), device=device)
    
    with torch.inference_mode():
        logits = model(tokens)[:, -1, :]
        io_logits = logits[batch_idx, io_toks]
        avg_logit = io_logits.mean().item()
    
    return avg_logit


def analyze_ch_ioi_accuracy(
    model: HookedTransformer,
    clean_tokens: torch.Tensor,
    corrupt_tokens: torch.Tensor,
    io_toks: torch.Tensor,
) -> Dict[str, float]:
    """
    Analyze model performance on Chinese IOI task.
    
    Returns:
        Dictionary with accuracy metrics
    """
    device = model.cfg.device
    batch_idx = torch.arange(len(io_toks), device=device)
    
    with torch.inference_mode():
        # Clean performance
        clean_logits = model(clean_tokens)[:, -1, :]
        clean_io_logits = clean_logits[batch_idx, io_toks]
        
        # Accuracy: is IO name the top prediction?
        clean_preds = clean_logits.argmax(dim=-1)
        clean_acc = (clean_preds == io_toks).float().mean().item()
        
        # Corrupt performance
        corrupt_logits = model(corrupt_tokens)[:, -1, :]
        corrupt_io_logits = corrupt_logits[batch_idx, io_toks]
        
        corrupt_preds = corrupt_logits.argmax(dim=-1)
        corrupt_acc = (corrupt_preds == io_toks).float().mean().item()
    
    return {
        'clean_avg_logit': clean_io_logits.mean().item(),
        'corrupt_avg_logit': corrupt_io_logits.mean().item(),
        'clean_accuracy': clean_acc,
        'corrupt_accuracy': corrupt_acc,
        'performance_drop': clean_acc - corrupt_acc,
    }


if __name__ == "__main__":
    """Test the data loader."""
    from transformer_lens import HookedTransformer
    
    print("Loading model...")
    model = HookedTransformer.from_pretrained("gpt2-small")
    
    print("\nLoading Chinese IOI dataset...")
    clean_tokens, corrupt_tokens, io_toks, pairs = \
        load_ch_ioi_dataset_for_patching(model, size="small", n_examples=5)
    
    print(f"\nClean tokens shape: {clean_tokens.shape}")
    print(f"Corrupt tokens shape: {corrupt_tokens.shape}")
    print(f"IO tokens shape: {io_toks.shape}")
    
    print("\nExample pair:")
    print(f"Clean:   {pairs[0]['clean']}")
    print(f"Corrupt: {pairs[0]['corrupt']}")
    print(f"IO name: {pairs[0]['io_name']}")
    print(f"IO token: '{pairs[0]['io_token']}'")
    
    print("\nAnalyzing accuracy...")
    metrics = analyze_ch_ioi_accuracy(model, clean_tokens, corrupt_tokens, io_toks)
    
    print("\nMetrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
