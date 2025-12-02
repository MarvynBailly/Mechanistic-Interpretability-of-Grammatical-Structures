"""
Utility functions to load and use simple IOI pairs with path patching.
"""
import json
from pathlib import Path
from typing import List, Dict, Tuple
import torch
from transformer_lens import HookedTransformer


def load_ioi_pairs(dataset_path: str) -> List[Dict]:
    """Load IOI pairs from JSON file."""
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['pairs']


def make_ioi_pair_from_data(
    model: HookedTransformer,
    pair_data: Dict,
) -> Tuple[Dict, Dict]:
    """
    Convert a pair from the dataset into the format expected by path patching.
    
    Args:
        model: HookedTransformer model
        pair_data: Dictionary with 'clean', 'corrupt', 'io_name', 's_name', 'io_token'
    
    Returns:
        (clean_dict, corrupt_dict) with keys: prompt, tokens, io_tok, s_tok
    """
    clean_prompt = pair_data['clean']
    corrupt_prompt = pair_data['corrupt']
    
    # Tokenize
    clean_tokens = model.to_tokens(clean_prompt, prepend_bos=False).to(model.cfg.device)
    corrupt_tokens = model.to_tokens(corrupt_prompt, prepend_bos=False).to(model.cfg.device)
    
    # Get token IDs for IO and S names (with leading space)
    io_name = pair_data['io_token']  # Already has leading space
    s_name = f" {pair_data['s_name']}"
    
    io_tok = model.to_single_token(io_name)
    s_tok = model.to_single_token(s_name)
    
    assert io_tok is not None, f"IO name '{io_name}' is not a single token"
    assert s_tok is not None, f"S name '{s_name}' is not a single token"
    
    clean = {
        "prompt": clean_prompt,
        "tokens": clean_tokens,
        "io_tok": io_tok,
        "s_tok": s_tok,
    }
    
    corrupt = {
        "prompt": corrupt_prompt,
        "tokens": corrupt_tokens,
        "io_tok": io_tok,  # Same expected answer as clean
        "s_tok": s_tok,     # Same expected answer as clean
    }
    
    return clean, corrupt


def get_dataset_path(size: str = "small") -> Path:
    """Get path to dataset file."""
    base_dir = Path(__file__).parent / "output"
    return base_dir / f"ioi_pairs_{size}.json"


# Example usage
if __name__ == "__main__":
    from transformer_lens import HookedTransformer
    
    # Load model
    model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
    
    # Load dataset
    pairs = load_ioi_pairs(get_dataset_path("small"))
    
    # Get first pair
    clean, corrupt = make_ioi_pair_from_data(model, pairs[0])
    
    print("Clean prompt:", clean['prompt'])
    print("Corrupt prompt:", corrupt['prompt'])
    print(f"Expected IO token ID: {clean['io_tok']}")
    print(f"Subject token ID: {clean['s_tok']}")
