"""
Direct Effect Analysis for Color-Object Association Task

This script implements path patching to identify which attention heads have a direct effect
on the logits for the color-object association task. This is analogous to Figure 3b from
"Interpretability in the Wild" (Wang et al., 2022).

The analysis:
1. Runs clean forward pass and caches each head's output
2. Runs corrupt forward pass with patched head output from clean
3. Measures recovery of logit difference (correct_object - incorrect_object)
4. Identifies "Object Mover Heads" that directly move correct object to logits
"""

import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from data_loader import load_dataset_for_patching, analyze_dataset_accuracy
from plotting import save_path_patching_heatmap, plot_top_heads, print_top_heads
from utils import set_seed


def path_patch_head_to_logits(
    model: HookedTransformer,
    clean_tokens: torch.Tensor,
    corrupt_tokens: torch.Tensor,
    correct_toks: torch.Tensor,
    incorrect_toks: torch.Tensor,
    layer: int,
    head: int,
) -> float:
    """
    Path patch from a single head to logits.
    
    This measures the direct effect of head (layer, head) on the logits by:
    1. Running clean forward pass and caching head output (z @ W_O)
    2. Running corrupt forward pass
    3. Replacing the head's output in step 2 with cached clean output
    4. Measuring change in logit difference
    
    Args:
        model: The transformer model
        clean_tokens: Clean input tokens [batch, seq_len]
        corrupt_tokens: Corrupt input tokens [batch, seq_len]
        correct_toks: Correct object token IDs [batch]
        incorrect_toks: Incorrect object token IDs [batch]
        layer: Layer index of head to patch
        head: Head index to patch
        
    Returns:
        Effect on logit difference (positive = head helps task)
    """
    device = model.cfg.device
    batch_idx = torch.arange(len(correct_toks), device=device)
    
    # 1. Cache the head output (pre-W_O) on clean input
    cache = {}
    hook_z_name = f"blocks.{layer}.attn.hook_z"
    
    def cache_clean_z(activation, hook):
        # activation shape: [batch, seq_len, n_heads, d_head]
        cache['z'] = activation[:, :, head, :].clone()
        return activation
    
    with torch.inference_mode():
        model.run_with_hooks(
            clean_tokens,
            fwd_hooks=[(hook_z_name, cache_clean_z)]
        )
    
    clean_z = cache['z'].detach()  # [batch, seq_len, d_head]
    
    # 2. Get corrupt baseline
    with torch.inference_mode():
        corrupt_logits = model(corrupt_tokens)[:, -1, :]
        corrupt_correct = corrupt_logits[batch_idx, correct_toks]
        corrupt_incorrect = corrupt_logits[batch_idx, incorrect_toks]
        corrupt_diff = (corrupt_correct - corrupt_incorrect).mean().item()
    
    # 3. Run corrupt input but patch in clean head output
    def patch_z(activation, hook):
        # Replace this head's z with clean version
        # activation shape: [batch, seq_len, n_heads, d_head]
        activation[:, :, head, :] = clean_z
        return activation
    
    with torch.inference_mode():
        patched_logits = model.run_with_hooks(
            corrupt_tokens,
            fwd_hooks=[(hook_z_name, patch_z)]
        )[:, -1, :]
        
        patched_correct = patched_logits[batch_idx, correct_toks]
        patched_incorrect = patched_logits[batch_idx, incorrect_toks]
        patched_diff = (patched_correct - patched_incorrect).mean().item()
    
    # Path patching effect: how much did patching recover from corrupt toward clean?
    # Positive = head is important (patching helps)
    effect = patched_diff - corrupt_diff
    
    return effect


def compute_all_path_patching_effects(
    model: HookedTransformer,
    clean_tokens: torch.Tensor,
    corrupt_tokens: torch.Tensor,
    correct_toks: torch.Tensor,
    incorrect_toks: torch.Tensor,
) -> torch.Tensor:
    """
    Compute path patching effects for all attention heads.
    
    For each head, measures: head → Logits direct path effect
    
    Returns:
        Tensor of shape [n_layers, n_heads] with path patching effects
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    
    effects = torch.zeros(n_layers, n_heads)
    
    print(f"\nComputing path patching Head → Logits for {n_layers} layers × {n_heads} heads...")
    print("This measures: how much does patching recover logit_diff from corrupt?")
    
    for layer in range(n_layers):
        print(f"Layer {layer}...", end=" ", flush=True)
        for head in range(n_heads):
            effect = path_patch_head_to_logits(
                model, clean_tokens, corrupt_tokens, correct_toks, incorrect_toks, layer, head
            )
            effects[layer, head] = effect
        print("✓")
    
    return effects


def main():
    """Run direct effect analysis for color-object association task."""
    
    print("="*80)
    print("DIRECT EFFECT ANALYSIS: Color-Object Association")
    print("Path Patching Head → Logits (Identifying Object Mover Heads)")
    print("="*80)
    
    # Configuration
    MODEL_NAME = "gpt2-small"
    DATASET_SIZE = "small"  # Start with small for testing
    N_EXAMPLES = 100
    SEED = 42
    OUTPUT_DIR = f"results/direct_effect/{N_EXAMPLES}_examples"
    
    # Setup
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    print(f"\nLoading model: {MODEL_NAME}")
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=device)
    
    # Load dataset
    print(f"\nLoading dataset: {DATASET_SIZE}, n={N_EXAMPLES}")
    clean_tokens, corrupt_tokens, correct_toks, incorrect_toks, pairs = \
        load_dataset_for_patching(model, size=DATASET_SIZE, n_examples=N_EXAMPLES)
    
    # Analyze baseline performance
    print("\n" + "="*80)
    print("BASELINE PERFORMANCE")
    print("="*80)
    
    stats = analyze_dataset_accuracy(
        model, clean_tokens, corrupt_tokens, correct_toks, incorrect_toks
    )
    
    print(f"\nClean prompts:")
    print(f"  Logit difference: {stats['clean_logit_diff']:.3f}")
    print(f"  Accuracy: {stats['clean_accuracy']*100:.1f}%")
    
    print(f"\nCorrupt prompts:")
    print(f"  Logit difference: {stats['corrupt_logit_diff']:.3f}")
    print(f"  Accuracy: {stats['corrupt_accuracy']*100:.1f}%")
    
    print(f"\nCorruption effect:")
    print(f"  Logit diff reduction: {stats['corruption_effect']:.3f}")
    
    # Check if task is solvable
    if stats['clean_logit_diff'] < 0.5:
        print("\n⚠️  WARNING: Clean logit difference is low!")
        print("   The model may not be solving this task well.")
    
    if stats['corruption_effect'] < 0.5:
        print("\n⚠️  WARNING: Corruption effect is small!")
        print("   The corrupt/clean distinction may not be strong enough.")
    
    # Compute path patching effects
    print("\n" + "="*80)
    print("PATH PATCHING ANALYSIS")
    print("="*80)
    
    effects = compute_all_path_patching_effects(
        model, clean_tokens, corrupt_tokens, correct_toks, incorrect_toks
    )
    
    # Print statistics
    print(f"\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    print(f"\nEffect statistics:")
    print(f"  Mean: {effects.mean():.4f}")
    print(f"  Std:  {effects.std():.4f}")
    print(f"  Max:  {effects.max():.4f}")
    print(f"  Min:  {effects.min():.4f}")
    
    # Identify top heads
    print_top_heads(effects, top_k=15, label="Top 15 Object Mover Heads (Positive Effect)")
    
    # Identify negative heads
    print_top_heads(-effects, top_k=10, label="Top 10 Negative Object Mover Heads")
    
    # Save visualizations
    print(f"\n" + "="*80)
    print("SAVING VISUALIZATIONS")
    print("="*80)
    
    # Save heatmap
    save_path_patching_heatmap(
        effects=effects,
        output_dir=OUTPUT_DIR,
        filename="object_mover_heatmap.png",
        title=f"Direct Effect: Head → Logits (n={N_EXAMPLES})"
    )
    
    # Save top heads bar chart
    plot_top_heads(
        effects=effects,
        top_k=15,
        output_dir=OUTPUT_DIR,
        filename="top_object_movers.png",
        title="Top Object Mover Heads"
    )
    
    # Save effects tensor for later analysis
    results_dir = Path(OUTPUT_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)
    torch.save(effects, results_dir / "direct_effects.pt")
    print(f"\nSaved effects tensor to: {results_dir / 'direct_effects.pt'}")
    
    # Save summary statistics
    summary = {
        'model': MODEL_NAME,
        'dataset_size': DATASET_SIZE,
        'n_examples': N_EXAMPLES,
        'clean_logit_diff': stats['clean_logit_diff'],
        'corrupt_logit_diff': stats['corrupt_logit_diff'],
        'corruption_effect': stats['corruption_effect'],
        'clean_accuracy': stats['clean_accuracy'],
        'corrupt_accuracy': stats['corrupt_accuracy'],
        'mean_effect': effects.mean().item(),
        'std_effect': effects.std().item(),
        'max_effect': effects.max().item(),
        'min_effect': effects.min().item(),
    }
    
    import json
    with open(results_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to: {results_dir / 'summary.json'}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {OUTPUT_DIR}/")
    print(f"  - object_mover_heatmap.png")
    print(f"  - top_object_movers.png")
    print(f"  - direct_effects.pt")
    print(f"  - summary.json")


if __name__ == "__main__":
    main()
