"""
Direct Effect Analysis for Python Code Completion Task

Identifies which attention heads have direct effects on predicting the correct
function argument in Python code completion.

Task: def func(arg1, arg2): return arg1 + ??? → should predict "arg2"
"""

import torch
from transformer_lens import HookedTransformer
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from data_loader_code import load_code_dataset_for_patching, analyze_code_accuracy
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
    """Path patch from a single head to logits."""
    device = model.cfg.device
    batch_idx = torch.arange(len(correct_toks), device=device)
    
    # Cache clean head output
    cache = {}
    hook_z_name = f"blocks.{layer}.attn.hook_z"
    
    def cache_clean_z(activation, hook):
        cache['z'] = activation[:, :, head, :].clone()
        return activation
    
    with torch.inference_mode():
        model.run_with_hooks(clean_tokens, fwd_hooks=[(hook_z_name, cache_clean_z)])
    
    clean_z = cache['z'].detach()
    
    # Get corrupt baseline
    with torch.inference_mode():
        corrupt_logits = model(corrupt_tokens)[:, -1, :]
        corrupt_correct = corrupt_logits[batch_idx, correct_toks]
        corrupt_incorrect = corrupt_logits[batch_idx, incorrect_toks]
        corrupt_diff = (corrupt_correct - corrupt_incorrect).mean().item()
    
    # Run corrupt with patched head
    def patch_z(activation, hook):
        activation[:, :, head, :] = clean_z
        return activation
    
    with torch.inference_mode():
        patched_logits = model.run_with_hooks(
            corrupt_tokens, fwd_hooks=[(hook_z_name, patch_z)]
        )[:, -1, :]
        
        patched_correct = patched_logits[batch_idx, correct_toks]
        patched_incorrect = patched_logits[batch_idx, incorrect_toks]
        patched_diff = (patched_correct - patched_incorrect).mean().item()
    
    effect = patched_diff - corrupt_diff
    return effect


def compute_all_effects(
    model: HookedTransformer,
    clean_tokens: torch.Tensor,
    corrupt_tokens: torch.Tensor,
    correct_toks: torch.Tensor,
    incorrect_toks: torch.Tensor,
) -> torch.Tensor:
    """Compute path patching effects for all heads."""
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    effects = torch.zeros(n_layers, n_heads)
    
    print(f"\nComputing path patching Head → Logits for {n_layers} layers × {n_heads} heads...")
    
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
    """Run direct effect analysis on code completion task."""
    
    print("="*80)
    print("DIRECT EFFECT ANALYSIS: Python Code Completion")
    print("="*80)
    
    # Configuration - use OBJECTS since it has 94% accuracy!
    VAR_TYPE = "objects"  # or "letters" (93% accuracy)
    N_EXAMPLES = 100
    SEED = 42
    OUTPUT_DIR = f"results/code_completion_{VAR_TYPE}/{N_EXAMPLES}_examples"
    
    # Setup
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\nLoading model...")
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    
    # Load dataset
    print(f"\nLoading {VAR_TYPE} code completion dataset...")
    clean_tokens, corrupt_tokens, correct_toks, incorrect_toks, pairs = \
        load_code_dataset_for_patching(model, var_type=VAR_TYPE, n_examples=N_EXAMPLES)
    
    # Analyze baseline
    print("\n" + "="*80)
    print("BASELINE PERFORMANCE")
    print("="*80)
    
    stats = analyze_code_accuracy(
        model, clean_tokens, corrupt_tokens, correct_toks, incorrect_toks
    )
    
    print(f"\nClean prompts:")
    print(f"  Logit difference: {stats['clean_logit_diff']:.3f}")
    print(f"  Accuracy: {stats['clean_accuracy']*100:.1f}%")
    
    print(f"\nCorrupt prompts:")
    print(f"  Logit difference: {stats['corrupt_logit_diff']:.3f}")
    print(f"  Accuracy: {stats['corrupt_accuracy']*100:.1f}%")
    
    print(f"\nCorruption effect:")
    print(f"  Logit diff change: {stats['corruption_effect']:.3f}")
    
    if stats['clean_accuracy'] > 0.8:
        print(f"\n✅ EXCELLENT! Model solves this task very well!")
    
    # Compute effects
    print("\n" + "="*80)
    print("PATH PATCHING ANALYSIS")
    print("="*80)
    
    effects = compute_all_effects(
        model, clean_tokens, corrupt_tokens, correct_toks, incorrect_toks
    )
    
    # Print results
    print(f"\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    print(f"\nEffect statistics:")
    print(f"  Mean: {effects.mean():.4f}")
    print(f"  Std:  {effects.std():.4f}")
    print(f"  Max:  {effects.max():.4f}")
    print(f"  Min:  {effects.min():.4f}")
    
    print_top_heads(effects, top_k=20, label="Top 20 Argument Mover Heads")
    
    # Save visualizations
    print(f"\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    save_path_patching_heatmap(
        effects=effects,
        output_dir=OUTPUT_DIR,
        filename="argument_mover_heatmap.png",
        title=f"Direct Effect: Head → Logits (Code {VAR_TYPE.title()}, n={N_EXAMPLES})"
    )
    
    plot_top_heads(
        effects=effects,
        top_k=20,
        output_dir=OUTPUT_DIR,
        filename="top_argument_movers.png",
        title="Top Argument Mover Heads"
    )
    
    # Save data
    results_dir = Path(OUTPUT_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)
    torch.save(effects, results_dir / "direct_effects.pt")
    
    import json
    summary = {
        'var_type': VAR_TYPE,
        'n_examples': N_EXAMPLES,
        'clean_logit_diff': stats['clean_logit_diff'],
        'corrupt_logit_diff': stats['corrupt_logit_diff'],
        'corruption_effect': stats['corruption_effect'],
        'clean_accuracy': stats['clean_accuracy'],
        'corrupt_accuracy': stats['corrupt_accuracy'],
        'mean_effect': effects.mean().item(),
        'max_effect': effects.max().item(),
    }
    
    with open(results_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Results saved to: {OUTPUT_DIR}/")
    print(f"  - argument_mover_heatmap.png")
    print(f"  - top_argument_movers.png")
    print(f"  - direct_effects.pt")
    print(f"  - summary.json")


if __name__ == "__main__":
    main()
