"""
Path Patching Full: Complete implementation of the 5-step path patching algorithm.

This package provides a self-contained implementation of path patching for
mechanistic interpretability research, as described in Wang et al. (2022).

Main components:
- path_patching: Core 5-step algorithm
- utils: IOI task utilities and evaluation
- plotting: Visualization tools

Quick start:
    >>> from path_patching_full import HeadSpec, ReceiverSpec, path_patch
    >>> from transformer_lens import HookedTransformer
    >>> 
    >>> model = HookedTransformer.from_pretrained("gpt2-small")
    >>> sender = HeadSpec(layer=9, head=9)
    >>> receiver = ReceiverSpec(layer=10, head=0, component='q')
    >>> 
    >>> effect = path_patch(model, clean_tokens, corrupt_tokens,
    ...                     sender, [receiver], io_toks, s_toks)
"""

__version__ = "1.0.0"
__author__ = "Marvyn Bailly"

# Import main classes and functions for convenient access
from .path_patching import (
    # Core data structures
    HeadSpec,
    ReceiverSpec,
    ActivationCache,
    
    # Main functions
    path_patch,
    batch_path_patch,
    
    # Step-by-step functions (for advanced users)
    gather_activations,
    create_freeze_and_patch_hooks,
    run_frozen_forward_pass,
    run_final_patched_forward_pass,
    
    # Utility functions
    get_all_attention_head_receivers,
    get_residual_stream_receivers,
)

from .utils import (
    # Data structures
    IOIExample,
    
    # Dataset generation
    set_seed,
    sample_name_pair,
    build_dataset,
    
    # Evaluation
    get_logits_for_next_token,
    evaluate,
    logit_diff_io_s_from_tokens,
    
    # Helpers
    find_token_positions,
)

from .plotting import (
    # Core plotting
    save_heatmap,
    save_path_patching_heatmap,
    
    # Specialized plots
    save_attention_heatmap,
    save_residual_patching_heatmap,
    
    # Comparative plots
    plot_all_attention_heatmaps,
    save_comparison_plot,
)

from .data_loader import (
    # Dataset loading
    get_dataset_path,
    load_ioi_pairs,
    prepare_batch_for_patching,
    load_dataset_for_patching,
    
    # Utilities
    print_pair_examples,
)

__all__ = [
    # Version info
    "__version__",
    "__author__",
    
    # Path patching
    "HeadSpec",
    "ReceiverSpec",
    "ActivationCache",
    "path_patch",
    "batch_path_patch",
    "gather_activations",
    "create_freeze_and_patch_hooks",
    "run_frozen_forward_pass",
    "run_final_patched_forward_pass",
    "get_all_attention_head_receivers",
    "get_residual_stream_receivers",
    
    # Utils
    "IOIExample",
    "set_seed",
    "sample_name_pair",
    "build_dataset",
    "get_logits_for_next_token",
    "evaluate",
    "logit_diff_io_s_from_tokens",
    "find_token_positions",
    
    # Data loading
    "get_dataset_path",
    "load_ioi_pairs",
    "prepare_batch_for_patching",
    "load_dataset_for_patching",
    "print_pair_examples",
    
    # Plotting
    "save_heatmap",
    "save_path_patching_heatmap",
    "save_attention_heatmap",
    "save_residual_patching_heatmap",
    "plot_all_attention_heatmaps",
    "save_comparison_plot",
]
