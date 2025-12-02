"""
Full Path Patching Algorithm Implementation.

This module implements the complete 5-step path patching algorithm as described in:
Wang et al., "Interpretability in the Wild" (2022)

The algorithm traces causal paths from a sender attention head to receiver components
by patching activations between clean and corrupted inputs.

Algorithm Overview (5 Steps):
1. Gather activations on x_origin and x_new
2. Freeze all heads to x_origin except sender h (patched to x_new)
3. Run forward pass with frozen/patched activations
4. Save receiver R activations from this forward pass
5. Run final forward pass patching receivers R to saved values
"""

from typing import List, Tuple, Optional, Set, Union, Dict
from dataclasses import dataclass
import torch
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
import numpy as np


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class HeadSpec:
    """
    Specification for an attention head.
    
    Attributes:
        layer: Layer index (0-indexed)
        head: Head index (0-indexed)
    """
    layer: int
    head: int
    
    def __hash__(self):
        return hash((self.layer, self.head))
    
    def __eq__(self, other):
        return self.layer == other.layer and self.head == other.head
    
    def __repr__(self):
        return f"L{self.layer}H{self.head}"


@dataclass
class ReceiverSpec:
    """
    Specification for a receiver component.
    
    Can be either:
    - An attention head (with component type: 'q', 'k', or 'v')
    - A residual stream position (component type: 'resid')
    
    Attributes:
        layer: Layer index (0-indexed)
        head: Head index (optional, for attention heads)
        component: Component type ('q', 'k', 'v', or 'resid')
        position: Token position (optional, for position-specific patching)
    """
    layer: int
    component: str  # 'q', 'k', 'v', or 'resid'
    head: Optional[int] = None  # Required for q/k/v, None for resid
    position: Optional[int] = None  # Optional position-specific patching
    
    def __hash__(self):
        return hash((self.layer, self.head, self.component, self.position))
    
    def __eq__(self, other):
        return (self.layer == other.layer and 
                self.head == other.head and 
                self.component == other.component and
                self.position == other.position)
    
    def __repr__(self):
        base = f"L{self.layer}"
        if self.head is not None:
            base += f"H{self.head}"
        base += f".{self.component}"
        if self.position is not None:
            base += f"[pos{self.position}]"
        return base
    
    def get_hook_name(self) -> str:
        """Get the TransformerLens hook name for this receiver."""
        if self.component == 'resid':
            return f"blocks.{self.layer}.hook_resid_post"
        elif self.component in ['q', 'k', 'v']:
            return f"blocks.{self.layer}.attn.hook_{self.component}"
        else:
            raise ValueError(f"Unknown component type: {self.component}")


# ============================================================================
# Activation Cache
# ============================================================================

class ActivationCache:
    """
    Cache for storing model activations during forward passes.
    
    This class manages:
    - Caching activations from x_origin and x_new
    - Providing frozen activations during patched forward passes
    - Storing receiver activations for final patching
    """
    
    def __init__(self):
        self.cache: Dict[str, torch.Tensor] = {}
    
    def store(self, name: str, activation: torch.Tensor):
        """Store an activation tensor."""
        self.cache[name] = activation.detach().clone()
    
    def get(self, name: str) -> Optional[torch.Tensor]:
        """Retrieve a stored activation."""
        return self.cache.get(name)
    
    def clear(self):
        """Clear all cached activations."""
        self.cache.clear()


# ============================================================================
# Step 1: Gather Activations
# ============================================================================

def gather_activations(
    model: HookedTransformer,
    tokens_origin: torch.Tensor,
    tokens_new: torch.Tensor,
    sender_head: HeadSpec,
    receivers: List[ReceiverSpec],
) -> Tuple[ActivationCache, ActivationCache]:
    """
    STEP 1: Gather activations on x_origin and x_new.
    
    This function runs forward passes on both inputs and caches all
    relevant activations (sender head outputs and receiver inputs).
    
    Args:
        model: The transformer model
        tokens_origin: Original input tokens [batch, seq_len]
        tokens_new: New (corrupted) input tokens [batch, seq_len]
        sender_head: The sender attention head to patch
        receivers: List of receiver components to monitor
        
    Returns:
        Tuple of (cache_origin, cache_new) with stored activations
        
    Note:
        We cache:
        - Sender head output (pattern and z)
        - All attention head outputs (to freeze them)
        - Receiver component inputs
    """
    cache_origin = ActivationCache()
    cache_new = ActivationCache()
    
    # Define hook names to cache
    hook_names = []
    
    # Cache all attention head outputs (z) for freezing
    for layer_idx in range(model.cfg.n_layers):
        hook_names.append(f"blocks.{layer_idx}.attn.hook_z")
    
    # Cache sender head output specifically
    sender_z_hook = f"blocks.{sender_head.layer}.attn.hook_z"
    if sender_z_hook not in hook_names:
        hook_names.append(sender_z_hook)
    
    # Cache receiver components
    for receiver in receivers:
        hook_name = receiver.get_hook_name()
        if hook_name not in hook_names:
            hook_names.append(hook_name)
    
    # Run on x_origin
    def cache_origin_hook(activation: torch.Tensor, hook: HookPoint):
        cache_origin.store(hook.name, activation)
        return activation
    
    with model.hooks([(name, cache_origin_hook) for name in hook_names]):
        _ = model(tokens_origin)
    
    # Run on x_new
    def cache_new_hook(activation: torch.Tensor, hook: HookPoint):
        cache_new.store(hook.name, activation)
        return activation
    
    with model.hooks([(name, cache_new_hook) for name in hook_names]):
        _ = model(tokens_new)
    
    return cache_origin, cache_new


# ============================================================================
# Step 2-3: Freeze and Patch
# ============================================================================

def create_freeze_and_patch_hooks(
    sender_head: HeadSpec,
    cache_origin: ActivationCache,
    cache_new: ActivationCache,
    model: HookedTransformer,
) -> List[Tuple[str, callable]]:
    """
    STEP 2: Create hooks to freeze all heads to x_origin except sender h.
    
    This creates hook functions that will:
    - Replace all attention head outputs with their x_origin values (freeze)
    - Except the sender head, which gets its x_new value (patch)
    
    Args:
        sender_head: The sender attention head to patch
        cache_origin: Cached activations from x_origin
        cache_new: Cached activations from x_new
        model: The transformer model
        
    Returns:
        List of (hook_name, hook_function) tuples
        
    Note:
        This implements the "freeze all except sender" mechanism.
        MLPs and layer norms are recomputed (not frozen).
    """
    hooks = []
    
    for layer_idx in range(model.cfg.n_layers):
        hook_name = f"blocks.{layer_idx}.attn.hook_z"
        
        def make_freeze_patch_hook(layer: int):
            def hook_fn(activation: torch.Tensor, hook: HookPoint):
                """
                Freeze all heads to origin, patch sender to new.
                
                activation shape: [batch, seq_len, n_heads, d_head]
                """
                frozen = cache_origin.get(hook.name).clone()
                
                # If this is the sender head's layer, patch it
                if layer == sender_head.layer:
                    new_z = cache_new.get(hook.name)
                    frozen[:, :, sender_head.head, :] = new_z[:, :, sender_head.head, :]
                
                return frozen
            
            return hook_fn
        
        hooks.append((hook_name, make_freeze_patch_hook(layer_idx)))
    
    return hooks


def run_frozen_forward_pass(
    model: HookedTransformer,
    tokens_origin: torch.Tensor,
    sender_head: HeadSpec,
    receivers: List[ReceiverSpec],
    cache_origin: ActivationCache,
    cache_new: ActivationCache,
) -> ActivationCache:
    """
    STEP 3-4: Run forward pass with frozen/patched heads and save receiver activations.
    
    This runs a forward pass where:
    - All attention heads are frozen to their x_origin activations
    - Except the sender head, which is patched to its x_new activation
    - MLPs and layer norms are recomputed normally
    - Receiver component activations are saved
    
    Args:
        model: The transformer model
        tokens_origin: Original input tokens [batch, seq_len]
        sender_head: The sender attention head (patched to x_new)
        receivers: List of receiver components to save
        cache_origin: Cached activations from x_origin
        cache_new: Cached activations from x_new
        
    Returns:
        Cache with saved receiver activations from this forward pass
    """
    cache_receivers = ActivationCache()
    
    # Create hooks to freeze all heads except sender
    freeze_hooks = create_freeze_and_patch_hooks(
        sender_head, cache_origin, cache_new, model
    )
    
    # Add hooks to save receiver activations
    receiver_hook_names = [r.get_hook_name() for r in receivers]
    
    def save_receiver_hook(activation: torch.Tensor, hook: HookPoint):
        cache_receivers.store(hook.name, activation)
        return activation
    
    receiver_hooks = [(name, save_receiver_hook) for name in receiver_hook_names]
    
    # Combine all hooks
    all_hooks = freeze_hooks + receiver_hooks
    
    # Run forward pass with frozen/patched activations
    with model.hooks(all_hooks):
        _ = model(tokens_origin)
    
    return cache_receivers


# ============================================================================
# Step 5: Final Forward Pass with Patched Receivers
# ============================================================================

def run_final_patched_forward_pass(
    model: HookedTransformer,
    tokens_origin: torch.Tensor,
    receivers: List[ReceiverSpec],
    cache_receivers: ActivationCache,
    io_toks: torch.Tensor,
    s_toks: torch.Tensor,
) -> float:
    """
    STEP 5: Run final forward pass patching receivers R to saved values.
    
    This runs a normal forward pass on x_origin, but patches the receiver
    components to the values saved in Step 4.
    
    Args:
        model: The transformer model
        tokens_origin: Original input tokens [batch, seq_len]
        receivers: List of receiver components to patch
        cache_receivers: Cached receiver activations from Step 4
        io_toks: Token IDs for IO names [batch]
        s_toks: Token IDs for S names [batch]
        
    Returns:
        Mean logit difference: logit(IO) - logit(S)
        
    Note:
        This is the final measurement that quantifies the causal effect
        of the path from sender h through receivers R.
    """
    # Create hooks to patch receivers
    hooks = []
    
    for receiver in receivers:
        hook_name = receiver.get_hook_name()
        saved_activation = cache_receivers.get(hook_name)
        
        def make_patch_hook(saved_act, rec: ReceiverSpec):
            def hook_fn(activation: torch.Tensor, hook: HookPoint):
                """Patch receiver to saved activation."""
                patched = activation.clone()
                
                # Handle different component types
                if rec.component == 'resid':
                    # Patch entire residual stream
                    patched = saved_act.clone()
                elif rec.component in ['q', 'k', 'v']:
                    # Patch specific head's q/k/v
                    # Shape: [batch, seq_len, n_heads, d_head]
                    if rec.head is not None:
                        patched[:, :, rec.head, :] = saved_act[:, :, rec.head, :]
                    else:
                        # Patch all heads for this component
                        patched = saved_act.clone()
                
                # Handle position-specific patching
                if rec.position is not None:
                    # Only patch at specific position
                    result = activation.clone()
                    result[:, rec.position] = patched[:, rec.position]
                    return result
                
                return patched
            
            return hook_fn
        
        hooks.append((hook_name, make_patch_hook(saved_activation, receiver)))
    
    # Run forward pass with patched receivers
    with model.hooks(hooks):
        logits = model(tokens_origin)  # [batch, seq_len, vocab]
    
    # Compute logit difference
    next_logits = logits[:, -1, :]  # [batch, vocab]
    batch_idx = torch.arange(tokens_origin.size(0), device=tokens_origin.device)
    
    io_logits = next_logits[batch_idx, io_toks]
    s_logits = next_logits[batch_idx, s_toks]
    
    logit_diff = (io_logits - s_logits).mean().item()
    
    return logit_diff


# ============================================================================
# Complete Path Patching Function
# ============================================================================

def path_patch(
    model: HookedTransformer,
    tokens_origin: torch.Tensor,
    tokens_new: torch.Tensor,
    sender_head: HeadSpec,
    receivers: List[ReceiverSpec],
    io_toks: torch.Tensor,
    s_toks: torch.Tensor,
) -> float:
    """
    Execute the complete 5-step path patching algorithm.
    
    This function implements the full path patching algorithm to measure
    the causal effect of the path from sender head h to receiver components R.
    
    Algorithm Steps:
    1. Gather activations on x_origin and x_new
    2. Freeze all heads to x_origin except sender h (patched to x_new)
    3. Run forward pass with frozen/patched activations
    4. Save receiver R activations from this forward pass
    5. Run final forward pass patching R to saved values, measure logit_diff
    
    Args:
        model: The transformer model
        tokens_origin: Original input tokens [batch, seq_len]
        tokens_new: New/corrupted input tokens [batch, seq_len]
        sender_head: The sender attention head to patch
        receivers: List of receiver components to measure
        io_toks: Token IDs for IO names [batch]
        s_toks: Token IDs for S names [batch]
        
    Returns:
        Mean logit difference: logit(IO) - logit(S) after path patching
        
    Example:
        >>> sender = HeadSpec(layer=9, head=9)
        >>> receivers = [ReceiverSpec(layer=10, component='q', head=0)]
        >>> effect = path_patch(model, clean_tokens, corrupt_tokens, 
        ...                     sender, receivers, io_ids, s_ids)
        
    Note:
        The returned logit difference quantifies how much information flows
        from the sender head through the receiver components to influence
        the model's final prediction.
    """
    # Step 1: Gather activations on both inputs
    cache_origin, cache_new = gather_activations(
        model, tokens_origin, tokens_new, sender_head, receivers
    )
    
    # Steps 2-4: Run forward pass with frozen/patched heads, save receivers
    cache_receivers = run_frozen_forward_pass(
        model, tokens_origin, sender_head, receivers,
        cache_origin, cache_new
    )
    
    # Step 5: Final forward pass with patched receivers, measure effect
    logit_diff = run_final_patched_forward_pass(
        model, tokens_origin, receivers, cache_receivers,
        io_toks, s_toks
    )
    
    return logit_diff


# ============================================================================
# Batch Path Patching for Experiments
# ============================================================================

def batch_path_patch(
    model: HookedTransformer,
    tokens_origin: torch.Tensor,
    tokens_new: torch.Tensor,
    sender_heads: List[HeadSpec],
    receivers: List[ReceiverSpec],
    io_toks: torch.Tensor,
    s_toks: torch.Tensor,
) -> torch.Tensor:
    """
    Run path patching for multiple sender heads to multiple receivers.
    
    This is useful for creating heatmaps showing which paths are important.
    
    Args:
        model: The transformer model
        tokens_origin: Original input tokens [batch, seq_len]
        tokens_new: New/corrupted input tokens [batch, seq_len]
        sender_heads: List of sender attention heads to test
        receivers: List of receiver components to test
        io_toks: Token IDs for IO names [batch]
        s_toks: Token IDs for S names [batch]
        
    Returns:
        Tensor of shape [len(sender_heads), len(receivers)] with path effects
        
    Note:
        Each entry (i, j) is the logit_diff when patching from sender_heads[i]
        to receivers[j].
    """
    effects = torch.zeros(len(sender_heads), len(receivers))
    
    for i, sender in enumerate(sender_heads):
        for j, receiver in enumerate(receivers):
            effect = path_patch(
                model, tokens_origin, tokens_new,
                sender, [receiver],  # Single receiver
                io_toks, s_toks
            )
            effects[i, j] = effect
    
    return effects


# ============================================================================
# Utility Functions for Receiver Generation
# ============================================================================

def get_all_attention_head_receivers(
    n_layers: int,
    n_heads: int,
    component: str = 'q',
) -> List[ReceiverSpec]:
    """
    Generate receiver specs for all attention heads in the model.
    
    Args:
        n_layers: Number of layers in the model
        n_heads: Number of heads per layer
        component: Which component to target ('q', 'k', or 'v')
        
    Returns:
        List of ReceiverSpec for all heads' specified component
    """
    receivers = []
    for layer in range(n_layers):
        for head in range(n_heads):
            receivers.append(ReceiverSpec(
                layer=layer,
                head=head,
                component=component
            ))
    return receivers


def get_residual_stream_receivers(
    n_layers: int,
    position: Optional[int] = None,
) -> List[ReceiverSpec]:
    """
    Generate receiver specs for residual stream at each layer.
    
    Args:
        n_layers: Number of layers in the model
        position: Optional specific token position to patch
        
    Returns:
        List of ReceiverSpec for residual stream at each layer
    """
    receivers = []
    for layer in range(n_layers):
        receivers.append(ReceiverSpec(
            layer=layer,
            component='resid',
            position=position
        ))
    return receivers
