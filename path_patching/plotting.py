"""
Plotting utilities for IOI path patching experiments.
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
from typing import Optional


def save_attention_heatmap(
    attn_matrix: torch.Tensor,
    title: str,
    filename: str,
    output_dir: Optional[str] = None,
    xlabel: str = "Head",
    ylabel: str = "Layer",
    dpi: int = 160,
):
    """
    Save an attention heatmap (layer x head).
    
    Args:
        attn_matrix: [L, H] tensor of attention values
        title: Plot title
        filename: Output filename (e.g., "attn_end_to_io.png")
        output_dir: Optional directory to save to (default: current directory)
        xlabel: X-axis label
        ylabel: Y-axis label
        dpi: Resolution for saved figure
    """
    if output_dir:
        output_path = Path(output_dir) / filename
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    else:
        output_path = Path(filename)
    
    plt.figure(figsize=(10, 6))
    plt.imshow(attn_matrix.detach().cpu(), aspect="auto")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()
    print(f"Saved {title} to: {output_path}")


def save_residual_patching_heatmap(
    effects: torch.Tensor,
    filename: str = "resid_patching_heatmap.png",
    output_dir: Optional[str] = None,
    dpi: int = 160,
):
    """
    Save residual stream path patching heatmap (layer x position).
    
    Args:
        effects: [L, seq_len] tensor of patching effects
        filename: Output filename
        output_dir: Optional directory to save to
        dpi: Resolution for saved figure
    """
    if output_dir:
        output_path = Path(output_dir) / filename
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    else:
        output_path = Path(filename)
    
    effects_cpu = effects.detach().cpu().numpy()
    
    plt.figure(figsize=(10, 6))
    plt.imshow(effects_cpu, aspect="auto")
    plt.xlabel("Token position")
    plt.ylabel("Layer")
    plt.title("Residual patching effect\nΔ logit_diff (patched - corrupt)")
    plt.colorbar(label="Δ logit_diff")
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()
    print(f"Saved residual patching heatmap to: {output_path}")


def plot_all_attention_heatmaps(
    attn_io: torch.Tensor,
    attn_s2: torch.Tensor,
    output_dir: Optional[str] = None,
    dpi: int = 160,
):
    """
    Generate and save all three attention heatmaps: END->IO, END->S2, and difference.
    
    Args:
        attn_io: [L, H] attention from END to IO position
        attn_s2: [L, H] attention from END to S2 position
        output_dir: Optional directory to save to
        dpi: Resolution for saved figures
    """
    # END -> IO
    save_attention_heatmap(
        attn_io,
        title="Avg attention from END to IO",
        filename="attn_end_to_io.png",
        output_dir=output_dir,
        dpi=dpi,
    )
    
    # END -> S2
    save_attention_heatmap(
        attn_s2,
        title="Avg attention from END to S2",
        filename="attn_end_to_s2.png",
        output_dir=output_dir,
        dpi=dpi,
    )
    
    # Difference (IO preference over S2)
    attn_diff = attn_io - attn_s2
    save_attention_heatmap(
        attn_diff,
        title="Attention difference (END -> IO minus END -> S2)",
        filename="attn_end_io_minus_s2.png",
        output_dir=output_dir,
        dpi=dpi,
    )
