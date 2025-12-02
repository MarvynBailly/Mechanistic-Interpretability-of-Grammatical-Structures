"""
Plotting utilities for path patching experiments.

This module provides visualization functions for:
- Path patching effect heatmaps
- Attention pattern visualizations
- Comparative analysis plots
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
from typing import Optional, List, Tuple


# ============================================================================
# Core Plotting Functions
# ============================================================================

def save_heatmap(
    data: torch.Tensor,
    title: str,
    filename: str,
    output_dir: Optional[str] = None,
    xlabel: str = "Column",
    ylabel: str = "Row",
    colorbar_label: Optional[str] = None,
    dpi: int = 160,
    cmap: str = "viridis",
    figsize: Tuple[int, int] = (10, 6),
    VMIN: float = None,
    VMAX: float = None,
):
    """
    Save a generic heatmap visualization.
    
    Args:
        data: 2D tensor to visualize
        title: Plot title
        filename: Output filename (e.g., "heatmap.png")
        output_dir: Optional directory to save to (creates if needed)
        xlabel: X-axis label
        ylabel: Y-axis label
        colorbar_label: Label for colorbar (optional)
        dpi: Resolution for saved figure
        cmap: Colormap name
        figsize: Figure size as (width, height)
    """
    if output_dir:
        output_path = Path(output_dir) / filename
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    else:
        output_path = Path(filename)
    
    data_cpu = data.detach().cpu().numpy()
    
    if VMIN is None:
        VMIN = data_cpu.min()
    if VMAX is None:
        VMAX = data_cpu.max()
    
    plt.figure(figsize=figsize)
    im = plt.imshow(data_cpu, aspect="auto", cmap=cmap, vmin=VMIN, vmax=VMAX)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    
    cbar = plt.colorbar(im)
    if colorbar_label:
        cbar.set_label(colorbar_label)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()
    print(f"Saved {title} to: {output_path}")


def save_path_patching_heatmap(
    effects: torch.Tensor,
    title: str = "Path Patching Effects",
    filename: str = "path_patching_heatmap.png",
    output_dir: Optional[str] = None,
    xlabel: str = "Receiver Component",
    ylabel: str = "Sender Head",
    colorbar_label: str = "Δ logit_diff",
    dpi: int = 160,
):
    """
    Save a path patching effect heatmap.
    
    Args:
        effects: 2D tensor of patching effects [senders, receivers]
        title: Plot title
        filename: Output filename
        output_dir: Optional directory to save to
        xlabel: X-axis label
        ylabel: Y-axis label
        colorbar_label: Label for colorbar
        dpi: Resolution for saved figure
        
    Note:
        Uses RdBu_r colormap centered at 0 for diverging effects.
    """
    effects_cpu = effects.detach().cpu().numpy()
    
    # Use diverging colormap centered at 0
    vmax = np.abs(effects_cpu).max()
    vmin = -vmax
    
    if output_dir:
        output_path = Path(output_dir) / filename
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    else:
        output_path = Path(filename)
    
    plt.figure(figsize=(12, 8))
    im = plt.imshow(
        effects_cpu,
        aspect="auto",
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    
    cbar = plt.colorbar(im)
    cbar.set_label(colorbar_label)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()
    print(f"Saved path patching heatmap to: {output_path}")


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
    Save an attention pattern heatmap (layer x head).
    
    Args:
        attn_matrix: [num_layers, num_heads] tensor of attention values
        title: Plot title
        filename: Output filename (e.g., "attention_pattern.png")
        output_dir: Optional directory to save to
        xlabel: X-axis label
        ylabel: Y-axis label
        dpi: Resolution for saved figure
    """
    save_heatmap(
        data=attn_matrix,
        title=title,
        filename=filename,
        output_dir=output_dir,
        xlabel=xlabel,
        ylabel=ylabel,
        dpi=dpi,
        cmap="viridis",
    )


def save_residual_patching_heatmap(
    effects: torch.Tensor,
    filename: str = "resid_patching_heatmap.png",
    output_dir: Optional[str] = None,
    dpi: int = 160,
):
    """
    Save residual stream patching effect heatmap (layer x position).
    
    Args:
        effects: [num_layers, seq_len] tensor of patching effects
        filename: Output filename
        output_dir: Optional directory to save to
        dpi: Resolution for saved figure
        
    Note:
        This is a specialized wrapper for residual stream patching visualization.
    """
    save_path_patching_heatmap(
        effects=effects,
        title="Residual Stream Patching Effect\nΔ logit_diff (patched - corrupt)",
        filename=filename,
        output_dir=output_dir,
        xlabel="Token Position",
        ylabel="Layer",
        colorbar_label="Δ logit_diff",
        dpi=dpi,
    )


# ============================================================================
# Comparative Plotting
# ============================================================================

def plot_all_attention_heatmaps(
    attn_io: torch.Tensor,
    attn_s2: torch.Tensor,
    output_dir: Optional[str] = None,
    dpi: int = 160,
):
    """
    Generate and save all three attention heatmaps: END->IO, END->S2, and difference.
    
    Args:
        attn_io: [num_layers, num_heads] attention from END to IO position
        attn_s2: [num_layers, num_heads] attention from END to S2 position
        output_dir: Optional directory to save to
        dpi: Resolution for saved figures
        
    Note:
        Generates three separate plots:
        1. Attention from END to IO
        2. Attention from END to S2  
        3. Difference (IO preference over S2)
    """
    # END -> IO
    save_attention_heatmap(
        attn_io,
        title="Average Attention from END to IO",
        filename="attn_end_to_io.png",
        output_dir=output_dir,
        dpi=dpi,
    )
    
    # END -> S2
    save_attention_heatmap(
        attn_s2,
        title="Average Attention from END to S2",
        filename="attn_end_to_s2.png",
        output_dir=output_dir,
        dpi=dpi,
    )
    
    # Difference (IO preference over S2)
    attn_diff = attn_io - attn_s2
    save_attention_heatmap(
        attn_diff,
        title="Attention Difference (END → IO minus END → S2)",
        filename="attn_end_io_minus_s2.png",
        output_dir=output_dir,
        dpi=dpi,
    )


def save_comparison_plot(
    data_list: List[torch.Tensor],
    labels: List[str],
    title: str,
    filename: str,
    output_dir: Optional[str] = None,
    xlabel: str = "Index",
    ylabel: str = "Value",
    dpi: int = 160,
):
    """
    Save a comparison line plot for multiple data series.
    
    Args:
        data_list: List of 1D tensors to plot
        labels: Labels for each data series
        title: Plot title
        filename: Output filename
        output_dir: Optional directory to save to
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
    
    for data, label in zip(data_list, labels):
        data_cpu = data.detach().cpu().numpy()
        plt.plot(data_cpu, label=label, marker='o', markersize=3)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()
    print(f"Saved comparison plot to: {output_path}")
