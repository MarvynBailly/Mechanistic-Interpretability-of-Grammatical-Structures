"""
Example: Using pre-generated IOI pairs with path patching.

This demonstrates how to use the simple dataset generator with the path patching code.
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent / "path_patching"))
sys.path.append(str(Path(__file__).parent.parent / "data_generation_simple"))

from IOI_pathpatching_gpu import IOIConfig
from load_pairs import load_ioi_pairs, make_ioi_pair_from_data, get_dataset_path
from transformer_lens import HookedTransformer
from utils import logit_diff_io_s_from_tokens
from plotting import save_residual_patching_heatmap
import torch


def run_path_patching_with_dataset(
    dataset_size: str = "small",
    device: str = None,
    output_dir: str = "results/simple_dataset",
):
    """
    Run path patching using pre-generated clean/corrupt pairs.
    
    Args:
        dataset_size: "small", "medium", or "large"
        device: Device to use (None for auto-detect)
        output_dir: Where to save results
    """
    # Auto-detect device if not specified
    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    
    print(f"Loading GPT-2 small on {device}...")
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    model.eval()
    
    # Load dataset
    dataset_path = get_dataset_path(dataset_size)
    print(f"Loading dataset from {dataset_path}...")
    pairs = load_ioi_pairs(str(dataset_path))
    print(f"Loaded {len(pairs)} pairs\n")
    
    # Use first pair for path patching demonstration
    pair_data = pairs[0]
    clean, corrupt = make_ioi_pair_from_data(model, pair_data)
    
    print("=== Path Patching Example ===")
    print(f"Clean:   {clean['prompt']}")
    print(f"Corrupt: {corrupt['prompt']}")
    print(f"Expected answer: {pair_data['io_token']}\n")
    
    # Get caches
    print("Running model with caching...")
    _, clean_cache = model.run_with_cache(clean["tokens"])
    _, corrupt_cache = model.run_with_cache(corrupt["tokens"])
    
    # Compute baseline logit differences
    clean_ld = logit_diff_io_s_from_tokens(
        model,
        clean["tokens"],
        torch.tensor([clean["s_tok"]], device=model.cfg.device),
        torch.tensor([clean["io_tok"]], device=model.cfg.device),
    )
    
    corrupt_ld = logit_diff_io_s_from_tokens(
        model,
        corrupt["tokens"],
        torch.tensor([corrupt["s_tok"]], device=model.cfg.device),
        torch.tensor([corrupt["io_tok"]], device=model.cfg.device),
    )
    
    print(f"Clean logit diff (IO - S):   {clean_ld:.3f}")
    print(f"Corrupt logit diff (IO - S): {corrupt_ld:.3f}")
    print(f"Difference: {clean_ld - corrupt_ld:.3f}")
    print()
    
    # Run path patching scan
    print("Running residual path patching scan...")
    from transformer_lens.utils import get_act_name
    
    L = model.cfg.n_layers
    seq_len = corrupt["tokens"].shape[1]
    effects = torch.zeros(L, seq_len, device=model.cfg.device)
    
    for layer in range(L):
        for pos in range(seq_len):
            # Patch this position
            act_name = get_act_name("resid_pre", layer)
            
            def hook(value, hook):
                clean_value = clean_cache[act_name]
                value[:, pos, :] = clean_value[:, pos, :]
                return value
            
            logits = model.run_with_hooks(
                corrupt["tokens"],
                fwd_hooks=[(act_name, hook)]
            )
            
            next_logits = logits[:, -1, :]
            batch_idx = torch.arange(1, device=model.cfg.device)
            io_logits = next_logits[batch_idx, corrupt["io_tok"]]
            s_logits = next_logits[batch_idx, corrupt["s_tok"]]
            patched_ld = (io_logits - s_logits).mean().item()
            
            effects[layer, pos] = patched_ld - corrupt_ld
    
    print("Saving heatmap...")
    save_residual_patching_heatmap(effects, output_dir=output_dir)
    print(f"\n✓ Path patching complete! Results saved to {output_dir}/")


if __name__ == "__main__":
    # Run with small dataset on GPU (if available)
    run_path_patching_with_dataset(
        dataset_size="small",
        device=None,  # Auto-detect
        output_dir="results/simple_dataset_example"
    )
