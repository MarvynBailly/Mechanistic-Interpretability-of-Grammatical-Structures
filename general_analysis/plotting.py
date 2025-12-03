"""
Plotting utilities for color-object association path patching experiments.

This module provides visualization functions for:
- Path patching effect heatmaps
- Attention pattern visualizations
- Comparative analysis plots
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Tuple


def save_heatmap(
    data: torch.Tensor,
    title: str,
    filename: str,
    output_dir: Optional[str] = None,
    xlabel: str = "Head",
    ylabel: str = "Layer",
    colorbar_label: Optional[str] = None,
    dpi: int = 160,
    cmap: str = "RdBu_r",
    figsize: Tuple[int, int] = (12, 8),
    vmin: float = None,
    vmax: float = None,
    annotate: bool = False,
):
    """
    Save a heatmap visualization.
    
    Args:
        data: 2D tensor to visualize [n_layers, n_heads]
        title: Plot title
        filename: Output filename (e.g., "heatmap.png")
        output_dir: Optional directory to save to (creates if needed)
        xlabel: X-axis label
        ylabel: Y-axis label
        colorbar_label: Label for colorbar (optional)
        dpi: Resolution for saved figure
        cmap: Colormap name
        figsize: Figure size as (width, height)
        vmin: Minimum value for colormap
        vmax: Maximum value for colormap
        annotate: Whether to annotate cells with values
    """
    if output_dir:
        output_path = Path(output_dir) / filename
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    else:
        output_path = Path(filename)
    
    data_cpu = data.detach().cpu().numpy()
    n_layers, n_heads = data_cpu.shape
    
    # Auto-scale vmin/vmax if not provided
    if vmin is None or vmax is None:
        abs_max = np.abs(data_cpu).max()
        if vmin is None:
            vmin = -abs_max
        if vmax is None:
            vmax = abs_max
    
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data_cpu, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    
    # Set ticks
    ax.set_xticks(np.arange(n_heads))
    ax.set_yticks(np.arange(n_layers))
    ax.set_xticklabels(np.arange(n_heads))
    ax.set_yticklabels(np.arange(n_layers))
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, pad=20)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    if colorbar_label:
        cbar.set_label(colorbar_label, fontsize=11)
    
    # Optionally annotate cells with values
    if annotate:
        for i in range(n_layers):
            for j in range(n_heads):
                text = ax.text(j, i, f'{data_cpu[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Saved {title} to: {output_path}")


def save_path_patching_heatmap(
    effects: torch.Tensor,
    output_dir: str,
    filename: str = "direct_effect_heatmap.png",
    title: str = "Direct Effect: Head → Logits",
    vmin: float = None,
    vmax: float = None,
):
    """
    Save path patching effects as a heatmap (analogous to Figure 3b).
    
    Args:
        effects: Tensor of shape [n_layers, n_heads] with path patching effects
        output_dir: Directory to save heatmap
        filename: Output filename
        title: Plot title
    """
    save_heatmap(
        data=effects,
        title=title,
        filename=filename,
        output_dir=output_dir,
        xlabel="Head",
        ylabel="Layer",
        colorbar_label="Logit Difference Recovery",
        cmap="RdBu_r",  # Red = positive (helps task), Blue = negative (hurts task)
        vmin=vmin,
        vmax=vmax,
    )


def plot_top_heads(
    effects: torch.Tensor,
    top_k: int = 10,
    output_dir: Optional[str] = None,
    filename: str = "top_heads.png",
    title: str = "Top Heads by Direct Effect",
):
    """
    Plot bar chart of top-k heads by effect magnitude.
    
    Args:
        effects: Tensor of shape [n_layers, n_heads] with path patching effects
        top_k: Number of top heads to show
        output_dir: Optional directory to save to
        filename: Output filename
        title: Plot title
    """
    if output_dir:
        output_path = Path(output_dir) / filename
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    else:
        output_path = Path(filename)
    
    effects_cpu = effects.detach().cpu().numpy()
    n_layers, n_heads = effects_cpu.shape
    
    # Flatten and get top-k
    flat_effects = effects_cpu.flatten()
    flat_indices = np.argsort(np.abs(flat_effects))[::-1][:top_k]
    
    # Convert back to (layer, head) coordinates
    layers = flat_indices // n_heads
    heads = flat_indices % n_heads
    values = flat_effects[flat_indices]
    
    # Create labels
    labels = [f"L{l}H{h}" for l, h in zip(layers, heads)]
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['red' if v > 0 else 'blue' for v in values]
    ax.barh(labels, values, color=colors, alpha=0.7)
    ax.set_xlabel('Effect on Logit Difference', fontsize=12)
    ax.set_title(title, fontsize=14, pad=20)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved top heads plot to: {output_path}")


def print_top_heads(
    effects: torch.Tensor,
    top_k: int = 10,
    label: str = "Top Heads",
):
    """
    Print top-k heads by effect magnitude.
    
    Args:
        effects: Tensor of shape [n_layers, n_heads]
        top_k: Number of top heads to print
        label: Label for the printout
    """
    effects_cpu = effects.detach().cpu().numpy()
    n_layers, n_heads = effects_cpu.shape
    
    # Flatten and get top-k
    flat_effects = effects_cpu.flatten()
    flat_indices = np.argsort(np.abs(flat_effects))[::-1][:top_k]
    
    # Convert back to (layer, head) coordinates
    layers = flat_indices // n_heads
    heads = flat_indices % n_heads
    values = flat_effects[flat_indices]
    
    print(f"\n{label}:")
    print("-" * 50)
    for i, (l, h, v) in enumerate(zip(layers, heads, values), 1):
        sign = "+" if v > 0 else ""
        print(f"{i:2d}. Layer {l:2d}, Head {h:2d}: {sign}{v:.4f}")
