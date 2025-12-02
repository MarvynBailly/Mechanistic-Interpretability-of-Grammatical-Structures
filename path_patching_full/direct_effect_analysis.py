"""
Direct Effect Analysis: Measure how much each attention head directly affects logit difference.

This script implements the direct effect measurement similar to Figure 3b in 
"Interpretability in the Wild" (Wang et al., 2022).

Direct effect measures: If we zero-ablate head (L, H), how much does logit_diff change?
"""

import torch
from transformer_lens import HookedTransformer
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from data_loader import load_dataset_for_patching
from plotting import save_heatmap
from utils import set_seed


def measure_direct_effect(
    model: HookedTransformer,
    clean_tokens: torch.Tensor,
    io_toks: torch.Tensor,
    s_toks: torch.Tensor,
    layer: int,
    head: int,
) -> float:
    """
    Measure the direct effect of a single attention head.
    
    Direct effect = logit_diff(normal) - logit_diff(head zeroed)
    
    If positive: head contributes to correct prediction (IO > S)
    If negative: head works against correct prediction
    
    Args:
        model: The transformer model
        clean_tokens: Clean input tokens [batch, seq_len]
        io_toks: IO token IDs [batch]
        s_toks: S token IDs [batch]
        layer: Layer index of head to ablate
        head: Head index to ablate
        
    Returns:
        Direct effect on logit difference
    """
    device = model.cfg.device
    
    # Baseline: normal forward pass
    with torch.inference_mode():
        baseline_logits = model(clean_tokens)[:, -1, :]
        batch_idx = torch.arange(len(io_toks), device=device)
        baseline_io = baseline_logits[batch_idx, io_toks]
        baseline_s = baseline_logits[batch_idx, s_toks]
        baseline_diff = (baseline_io - baseline_s).mean().item()
    
    # Ablation: zero out the head's output
    def ablate_head(activation, hook):
        # activation shape: [batch, seq_len, n_heads, d_head]
        activation[:, :, head, :] = 0.0
        return activation
    
    hook_name = f"blocks.{layer}.attn.hook_z"
    
    with torch.inference_mode():
        ablated_logits = model.run_with_hooks(
            clean_tokens,
            fwd_hooks=[(hook_name, ablate_head)]
        )[:, -1, :]
        
        ablated_io = ablated_logits[batch_idx, io_toks]
        ablated_s = ablated_logits[batch_idx, s_toks]
        ablated_diff = (ablated_io - ablated_s).mean().item()
    
    # Direct effect = how much we lost by ablating
    direct_effect = baseline_diff - ablated_diff
    
    return direct_effect


def compute_all_direct_effects(
    model: HookedTransformer,
    clean_tokens: torch.Tensor,
    io_toks: torch.Tensor,
    s_toks: torch.Tensor,
) -> torch.Tensor:
    """
    Compute direct effects for all attention heads in the model.
    
    Returns:
        Tensor of shape [n_layers, n_heads] with direct effects
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    
    effects = torch.zeros(n_layers, n_heads)
    
    print(f"\nComputing direct effects for {n_layers} layers × {n_heads} heads...")
    print("This measures: logit_diff(normal) - logit_diff(head ablated)")
    
    for layer in range(n_layers):
        print(f"Layer {layer}...", end=" ", flush=True)
        for head in range(n_heads):
            effect = measure_direct_effect(
                model, clean_tokens, io_toks, s_toks, layer, head
            )
            effects[layer, head] = effect
        print("✓")
    
    return effects


def main():
    """Generate direct effect heatmap similar to Figure 3b."""
    
    print("="*70)
    print("DIRECT EFFECT ANALYSIS")
    print("Measuring how much each head contributes to logit_diff")
    print("="*70)
    
    # Configuration
    MODEL_NAME = "gpt2-small"
    DATASET_SIZE = "small"
    N_EXAMPLES = 50
    SEED = 42
    OUTPUT_DIR = "path_patching_results"
    
    # Setup
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\nLoading model: {MODEL_NAME}")
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=device)
    
    # Load dataset
    print(f"\nLoading dataset: {DATASET_SIZE}, n={N_EXAMPLES}")
    clean_tokens, _, io_toks, s_toks, _ = load_dataset_for_patching(
        model, size=DATASET_SIZE, n_examples=N_EXAMPLES
    )
    
    # Compute baseline
    with torch.inference_mode():
        baseline_logits = model(clean_tokens)[:, -1, :]
        batch_idx = torch.arange(len(io_toks), device=device)
        baseline_io = baseline_logits[batch_idx, io_toks]
        baseline_s = baseline_logits[batch_idx, s_toks]
        baseline_diff = (baseline_io - baseline_s).mean().item()
    
    print(f"\nBaseline logit_diff: {baseline_diff:.3f}")
    
    # Compute all direct effects
    effects = compute_all_direct_effects(model, clean_tokens, io_toks, s_toks)
    
    # Print statistics
    print(f"\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Mean direct effect: {effects.mean().item():.4f}")
    print(f"Max direct effect:  {effects.max().item():.4f}")
    print(f"Min direct effect:  {effects.min().item():.4f}")
    
    # Find most important heads
    flat_effects = effects.flatten()
    flat_indices = flat_effects.argsort(descending=True)
    
    print(f"\nTop 10 most important heads (positive contribution):")
    for i in range(10):
        idx = flat_indices[i].item()
        layer = idx // model.cfg.n_heads
        head = idx % model.cfg.n_heads
        effect = flat_effects[idx].item()
        print(f"  {i+1}. L{layer}H{head}: {effect:.4f}")
    
    print(f"\nTop 5 most negative heads (working against IOI):")
    for i in range(5):
        idx = flat_indices[-(i+1)].item()
        layer = idx // model.cfg.n_heads
        head = idx % model.cfg.n_heads
        effect = flat_effects[idx].item()
        print(f"  {i+1}. L{layer}H{head}: {effect:.4f}")
    
    # Save heatmap
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving heatmap to {OUTPUT_DIR}/direct_effect_heatmap.png...")
    save_heatmap(
        data=effects,
        title="Direct Effect of Attention Heads on IOI Task\n(Positive = helps predict IO)",
        filename="direct_effect_heatmap.png",
        output_dir=OUTPUT_DIR,
        xlabel="Head",
        ylabel="Layer",
        colorbar_label="Direct Effect (Δ logit_diff)",
        cmap="RdBu_r",  # Diverging colormap: red=negative, blue=positive
        figsize=(12, 8),
    )
    
    print("\n" + "="*70)
    print("DONE!")
    print("="*70)
    print(f"Results saved to: {OUTPUT_DIR}/direct_effect_heatmap.png")


if __name__ == "__main__":
    main()
