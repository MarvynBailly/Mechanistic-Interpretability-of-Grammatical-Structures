"""
Direct Effect Analysis: Reproduce Figure 3b from "Interpretability in the Wild" (Wang et al., 2022).

This script implements:
1. Path patching from each head h → Logits to measure direct effect on logit difference
2. Copy score analysis to verify Name Mover Heads copy names via their OV matrices
3. Negative copy score to identify Negative Name Mover Heads

Path patching runs forward pass on clean IOI input, but replaces activations from a head
with those from corrupted ABC input, then measures effect on logit difference.
"""

import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from data_loader import load_dataset_for_patching
from plotting import save_heatmap
from utils import set_seed
from typing import Tuple


def path_patch_head_to_logits(
    model: HookedTransformer,
    clean_tokens: torch.Tensor,
    corrupt_tokens: torch.Tensor,
    io_toks: torch.Tensor,
    s_toks: torch.Tensor,
    layer: int,
    head: int,
) -> float:
    """
    Path patch from a single head to logits (Figure 3b methodology).
    
    This measures the direct effect of head (layer, head) on the logits by:
    1. Running clean forward pass and caching head output (z @ W_O)
    2. Running corrupt forward pass
    3. Replacing the head's output in step 2 with cached clean output
    4. Measuring change in logit difference
    
    Path patching tests: "Does this head carry information critical for IOI task?"
    Large drop in logit_diff = head is important for the task
    
    Args:
        model: The transformer model
        clean_tokens: Clean IOI input tokens [batch, seq_len]
        corrupt_tokens: Corrupt ABC input tokens [batch, seq_len]
        io_toks: IO token IDs [batch]
        s_toks: S token IDs [batch]
        layer: Layer index of head to patch
        head: Head index to patch
        
    Returns:
        Effect on logit difference (positive = head helps IOI task)
    """
    device = model.cfg.device
    batch_idx = torch.arange(len(io_toks), device=device)
    
    # 1. Cache the head output (after W_O) on clean input
    # We'll use hook_z (pre-W_O) and manually apply W_O
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
        corrupt_io = corrupt_logits[batch_idx, io_toks]
        corrupt_s = corrupt_logits[batch_idx, s_toks]
        corrupt_diff = (corrupt_io - corrupt_s).mean().item()
    
    # 3. Run corrupt input but patch in clean head output
    # We patch at the z level and let W_O be applied normally
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
        
        patched_io = patched_logits[batch_idx, io_toks]
        patched_s = patched_logits[batch_idx, s_toks]
        patched_diff = (patched_io - patched_s).mean().item()
    
    # Path patching effect: how much did patching recover from corrupt to clean?
    # Positive = head is important (patching helps)
    effect = corrupt_diff - patched_diff
    
    return effect


def compute_all_path_patching_effects(
    model: HookedTransformer,
    clean_tokens: torch.Tensor,
    corrupt_tokens: torch.Tensor,
    io_toks: torch.Tensor,
    s_toks: torch.Tensor,
) -> torch.Tensor:
    """
    Compute path patching effects for all attention heads (Figure 3b).
    
    For each head, measures: h → Logits direct path effect
    
    Returns:
        Tensor of shape [n_layers, n_heads] with path patching effects
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    
    effects = torch.zeros(n_layers, n_heads)
    
    print(f"\nComputing path patching h → Logits for {n_layers} layers × {n_heads} heads...")
    print("This measures: how much does patching recover logit_diff from corrupt?")
    
    for layer in range(n_layers):
        print(f"Layer {layer}...", end=" ", flush=True)
        for head in range(n_heads):
            effect = path_patch_head_to_logits(
                model, clean_tokens, corrupt_tokens, io_toks, s_toks, layer, head
            )
            effects[layer, head] = effect
        print("✓")
    
    return effects


def get_name_token_positions(
    model: HookedTransformer,
    clean_tokens: torch.Tensor,
    io_toks: torch.Tensor,
    s_toks: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Find the positions of IO and S name tokens in the clean prompts.
    
    The IOI task has structure: "When IO and S ... S gave ... to"
    - IO appears once (first position)
    - S appears twice (S1 and S2)
    
    Args:
        model: The transformer model
        clean_tokens: Clean input tokens [batch, seq_len]
        io_toks: IO token IDs [batch]
        s_toks: S token IDs [batch]
        
    Returns:
        Tuple of (io_positions, s1_positions, s2_positions) each [batch]
    """
    batch_size, seq_len = clean_tokens.shape
    device = clean_tokens.device
    
    io_positions = torch.zeros(batch_size, dtype=torch.long, device=device)
    s1_positions = torch.zeros(batch_size, dtype=torch.long, device=device)
    s2_positions = torch.zeros(batch_size, dtype=torch.long, device=device)
    
    for i in range(batch_size):
        # Find IO position (appears once)
        io_mask = clean_tokens[i] == io_toks[i]
        io_pos = torch.where(io_mask)[0]
        if len(io_pos) > 0:
            io_positions[i] = io_pos[0]  # First (and usually only) occurrence
        
        # Find S positions (appears twice: S1 and S2)
        s_mask = clean_tokens[i] == s_toks[i]
        s_pos = torch.where(s_mask)[0]
        if len(s_pos) >= 2:
            s1_positions[i] = s_pos[0]  # First S
            s2_positions[i] = s_pos[1]  # Second S
        elif len(s_pos) == 1:
            s1_positions[i] = s_pos[0]
            s2_positions[i] = s_pos[0]
    
    return io_positions, s1_positions, s2_positions


def compute_copy_score_for_head(
    model: HookedTransformer,
    clean_tokens: torch.Tensor,
    name_positions: torch.Tensor,
    name_token_ids: torch.Tensor,
    layer: int,
    head: int,
    mlp_layer: int = 0,
    top_k: int = 5,
    negative: bool = False,
) -> float:
    """
    Compute copy score for a head's OV circuit following the paper's methodology.
    
    Paper's process:
    1. Get residual stream at name token position after first MLP layer
    2. Multiply by head's OV matrix (simulating perfect attention to that token)
    3. Multiply by unembedding matrix and apply layer norm
    4. Check if name token appears in top-k logits
    
    Args:
        model: The transformer model
        clean_tokens: Clean input tokens [batch, seq_len]
        name_positions: Positions of name tokens to test [batch]
        name_token_ids: The name token IDs to check for [batch]
        layer: Layer index of head to analyze
        head: Head index to analyze
        mlp_layer: Which MLP layer output to use (default: 0 = first MLP)
        top_k: How many top logits to check (default: 5)
        negative: If True, use negative OV matrix (for Negative Name Movers)
        
    Returns:
        Copy score: proportion of samples where name token is in top-k
    """
    device = model.cfg.device
    batch_size = clean_tokens.shape[0]
    
    # Step 1: Get residual stream at name positions after MLP layer
    hook_name = f"blocks.{mlp_layer}.hook_resid_post"
    cache = {}
    
    def cache_hook(activation, hook):
        cache['resid'] = activation.clone()
        return activation
    
    with torch.inference_mode():
        model.run_with_hooks(
            clean_tokens,
            fwd_hooks=[(hook_name, cache_hook)]
        )
    
    resid = cache['resid']  # [batch, seq_len, d_model]
    
    # Extract activations at name positions
    batch_idx = torch.arange(batch_size, device=device)
    name_activations = resid[batch_idx, name_positions]  # [batch, d_model]
    
    # Step 2: Apply head's OV matrix
    W_V = model.W_V[layer, head]  # [d_model, d_head]
    W_O = model.W_O[layer, head]  # [d_head, d_model]
    W_OV = W_V @ W_O  # [d_model, d_model]
    
    if negative:
        W_OV = -W_OV
    
    ov_output = name_activations @ W_OV  # [batch, d_model]
    
    # Step 3: Apply layer norm and unembedding
    # Note: Paper says "multiplied by the unembedding matrix, and applied the final layer norm"
    # The standard order is layer norm then unembed
    ln_out = model.ln_final(ov_output)  # [batch, d_model]
    logits = ln_out @ model.W_U  # [batch, d_vocab]
    
    # Step 4: Check if name token is in top-k
    top_k_tokens = torch.topk(logits, k=top_k, dim=-1).indices  # [batch, top_k]
    target_in_topk = (top_k_tokens == name_token_ids.unsqueeze(-1)).any(dim=-1)
    
    copy_score = target_in_topk.float().mean().item()
    
    return copy_score


def compute_all_copy_scores(
    model: HookedTransformer,
    clean_tokens: torch.Tensor,
    io_toks: torch.Tensor,
    s_toks: torch.Tensor,
    top_k: int = 5,
    negative: bool = False,
) -> torch.Tensor:
    """
    Compute copy scores for all attention heads following the paper's methodology.
    
    Tests all name token positions (IO, S1, S2) and aggregates the results.
    
    Args:
        model: The transformer model
        clean_tokens: Clean input tokens [batch, seq_len]
        io_toks: IO token IDs [batch]
        s_toks: S token IDs [batch]
        top_k: How many top logits to check (default: 5)
        negative: If True, use negative OV matrices
        
    Returns:
        Tensor of shape [n_layers, n_heads] with copy scores
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    
    # Get all name token positions
    io_positions, s1_positions, s2_positions = get_name_token_positions(
        model, clean_tokens, io_toks, s_toks
    )
    
    scores = torch.zeros(n_layers, n_heads)
    
    sign = "negative" if negative else "positive"
    print(f"\nComputing {sign} copy scores for {n_layers} layers × {n_heads} heads...")
    print(f"Testing all name positions (IO, S1, S2), top-{top_k}")
    
    for layer in range(n_layers):
        print(f"Layer {layer}...", end=" ", flush=True)
        for head in range(n_heads):
            # Test on all name positions and average
            # IO position
            io_score = compute_copy_score_for_head(
                model, clean_tokens, io_positions, io_toks,
                layer, head, mlp_layer=0, top_k=top_k, negative=negative
            )
            
            # S positions (both S1 and S2)
            s1_score = compute_copy_score_for_head(
                model, clean_tokens, s1_positions, s_toks,
                layer, head, mlp_layer=0, top_k=top_k, negative=negative
            )
            
            s2_score = compute_copy_score_for_head(
                model, clean_tokens, s2_positions, s_toks,
                layer, head, mlp_layer=0, top_k=top_k, negative=negative
            )
            
            # Average across all name positions
            avg_score = (io_score + s1_score + s2_score) / 3.0
            scores[layer, head] = avg_score
        print("✓")
    
    return scores


def main():
    """Reproduce Figure 3b from the IOI paper using path patching."""
    
    print("="*70)
    print("FIGURE 3B: PATH PATCHING h → Logits")
    print("Identifying Name Mover Heads and Negative Name Mover Heads")
    print("="*70)
    
    # Configuration
    MODEL_NAME = "gpt2-small"
    DATASET_SIZE = "large"
    N_EXAMPLES = 1000  # Paper uses N=1000 for copy score, we'll use 100 for speed
    SEED = 42
    OUTPUT_DIR = "path_patching_results"
    
    # Setup
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\nLoading model: {MODEL_NAME}")
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=device)
    
    # Load dataset
    print(f"\nLoading dataset: {DATASET_SIZE}, n={N_EXAMPLES}")
    clean_tokens, corrupt_tokens, io_toks, s_toks, _ = load_dataset_for_patching(
        model, size=DATASET_SIZE, n_examples=N_EXAMPLES
    )
    
    # Compute baseline
    with torch.inference_mode():
        baseline_logits = model(clean_tokens)[:, -1, :]
        batch_idx = torch.arange(len(io_toks), device=device)
        baseline_io = baseline_logits[batch_idx, io_toks]
        baseline_s = baseline_logits[batch_idx, s_toks]
        baseline_diff = (baseline_io - baseline_s).mean().item()
    
    print(f"\nBaseline logit_diff (clean): {baseline_diff:.3f}")
    
    # Also compute corrupt baseline
    with torch.inference_mode():
        corrupt_logits = model(corrupt_tokens)[:, -1, :]
        corrupt_io = corrupt_logits[batch_idx, io_toks]
        corrupt_s = corrupt_logits[batch_idx, s_toks]
        corrupt_diff = (corrupt_io - corrupt_s).mean().item()
    
    print(f"Baseline logit_diff (corrupt): {corrupt_diff:.3f}")
    print(f"→ Corruption reduces logit_diff by: {baseline_diff - corrupt_diff:.3f}")
    
    # Compute path patching effects (Figure 3b methodology)
    effects = compute_all_path_patching_effects(
        model, clean_tokens, corrupt_tokens, io_toks, s_toks
    )
    
    # Print statistics
    print(f"\n" + "="*70)
    print("PATH PATCHING RESULTS (Figure 3b)")
    print("="*70)
    print(f"Mean path patching effect: {effects.mean().item():.4f}")
    print(f"Max effect (Name Movers):  {effects.max().item():.4f}")
    print(f"Min effect (Neg. Movers):  {effects.min().item():.4f}")
    
    # Find most important heads
    flat_effects = effects.flatten()
    flat_indices = flat_effects.argsort(descending=True)
    
    print(f"\nTop 10 Name Mover Heads (positive effect - help IOI task):")
    print("Paper identifies: 9.9, 10.0, 9.6")
    for i in range(10):
        idx = flat_indices[i].item()
        layer = idx // model.cfg.n_heads
        head = idx % model.cfg.n_heads
        effect = flat_effects[idx].item()
        print(f"  {i+1}. L{layer}H{head}: {effect:.4f}")
    
    print(f"\nTop 5 Negative Name Mover Heads (negative effect):")
    print("Paper identifies: 10.7, 11.10")
    for i in range(5):
        idx = flat_indices[-(i+1)].item()
        layer = idx // model.cfg.n_heads
        head = idx % model.cfg.n_heads
        effect = flat_effects[idx].item()
        print(f"  {i+1}. L{layer}H{head}: {effect:.4f}")
    
    # Save heatmap
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving heatmap to {OUTPUT_DIR}/figure_3b_path_patching.png...")
    save_heatmap(
        data=effects,
        title="Figure 3b: Path Patching h → Logits\n(Positive = Name Movers, Negative = Negative Name Movers)",
        filename="figure_3b_path_patching.png",
        output_dir=OUTPUT_DIR,
        xlabel="Head",
        ylabel="Layer",
        colorbar_label="Path Patching Effect",
        cmap="RdBu",  # Diverging colormap: red=positive (Name Movers), blue=negative
        figsize=(12, 8),
    )
    
    print("\n" + "="*70)
    print("COPY SCORE ANALYSIS")
    print("Testing if Name Mover Heads copy names via OV matrices")
    print("="*70)
    
    # Compute copy scores (Name Mover Heads should score >95%)
    print("\n1. Testing positive copy scores (Name Mover Heads):")
    print("Paper reports: Name Mover Heads >95%, average head <20%")
    copy_scores = compute_all_copy_scores(
        model, clean_tokens, io_toks, s_toks,
        top_k=5, negative=False
    )
    
    # Print statistics
    print(f"\nCopy Score Statistics:")
    print(f"Mean copy score: {copy_scores.mean().item():.1%}")
    print(f"Max copy score:  {copy_scores.max().item():.1%}")
    
    # Find heads with highest copy scores
    flat_scores = copy_scores.flatten()
    flat_indices = flat_scores.argsort(descending=True)
    
    print(f"\nTop 10 heads by copy score (likely Name Mover Heads):")
    for i in range(10):
        idx = flat_indices[i].item()
        layer = idx // model.cfg.n_heads
        head = idx % model.cfg.n_heads
        score = flat_scores[idx].item()
        # Also show direct effect for comparison
        direct_eff = effects[layer, head].item()
        print(f"  {i+1}. L{layer}H{head}: {score:.1%} copy score, {direct_eff:.4f} direct effect")
    
    # Save copy score heatmap
    print(f"\nSaving copy score heatmap to {OUTPUT_DIR}/copy_score_heatmap.png...")
    save_heatmap(
        data=copy_scores,
        title="Copy Score: Proportion of Samples with Name in Top-5 Logits\n(via OV Matrix Analysis)",
        filename="copy_score_heatmap.png",
        output_dir=OUTPUT_DIR,
        xlabel="Head",
        ylabel="Layer",
        colorbar_label="Copy Score",
        cmap="viridis",
        figsize=(12, 8),
    )
    
    # Compute negative copy scores (Negative Name Mover Heads should score >98%)
    print("\n2. Testing negative copy scores (Negative Name Mover Heads):")
    print("Paper reports: Negative Name Mover Heads >98%, average head <12%")
    negative_copy_scores = compute_all_copy_scores(
        model, clean_tokens, io_toks, s_toks,
        top_k=5, negative=True
    )
    
    print(f"\nNegative Copy Score Statistics:")
    print(f"Mean negative copy score: {negative_copy_scores.mean().item():.1%}")
    print(f"Max negative copy score:  {negative_copy_scores.max().item():.1%}")
    
    # Find heads with highest negative copy scores
    flat_neg_scores = negative_copy_scores.flatten()
    flat_neg_indices = flat_neg_scores.argsort(descending=True)
    
    print(f"\nTop 10 heads by negative copy score (likely Negative Name Mover Heads):")
    for i in range(10):
        idx = flat_neg_indices[i].item()
        layer = idx // model.cfg.n_heads
        head = idx % model.cfg.n_heads
        score = flat_neg_scores[idx].item()
        direct_eff = effects[layer, head].item()
        print(f"  {i+1}. L{layer}H{head}: {score:.1%} negative copy score, {direct_eff:.4f} direct effect")
    
    # Save negative copy score heatmap
    print(f"\nSaving negative copy score heatmap to {OUTPUT_DIR}/negative_copy_score_heatmap.png...")
    save_heatmap(
        data=negative_copy_scores,
        title="Negative Copy Score: Heads that Write Opposite of Names\n(via -OV Matrix Analysis)",
        filename="negative_copy_score_heatmap.png",
        output_dir=OUTPUT_DIR,
        xlabel="Head",
        ylabel="Layer",
        colorbar_label="Negative Copy Score",
        cmap="plasma",
        figsize=(12, 8),
    )
    
    print("\n" + "="*70)
    print("DONE! Figure 3b Recreation Complete")
    print("="*70)
    print(f"\nResults saved to {OUTPUT_DIR}/:")
    print(f"  - figure_3b_path_patching.png  (Main result)")
    print(f"  - copy_score_heatmap.png       (Name Mover verification)")
    print(f"  - negative_copy_score_heatmap.png  (Negative Name Mover verification)")
    print(f"\nCompare your figure_3b_path_patching.png with the paper's Figure 3b")
    print(f"Expected Name Movers: 9.9, 10.0, 9.6")
    print(f"Expected Negative Movers: 10.7, 11.10")


if __name__ == "__main__":
    main()
