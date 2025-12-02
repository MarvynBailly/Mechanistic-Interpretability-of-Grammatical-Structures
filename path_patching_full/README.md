# Full Path Patching Implementation

This directory contains a complete, self-contained implementation of the **path patching algorithm** as described in Wang et al., "Interpretability in the Wild" (2022).

## Overview

Path patching is a mechanistic interpretability technique that identifies causal paths through neural network circuits. It traces how information flows from a **sender** attention head through **receiver** components to influence the model's output.

### The 5-Step Algorithm

Given:
- $x_{\text{origin}}$ - the original (clean) input
- $x_{\text{new}}$ - the corrupted/counterfactual input  
- $h$ - the **sender** attention head
- $R$ - the set of **receiver** nodes (attention head inputs or residual stream positions)

The algorithm:

1. **Gather Activations**: Run forward passes on both $x_{\text{origin}}$ and $x_{\text{new}}$, caching all attention head outputs
2. **Freeze and Patch**: Freeze all attention heads to their $x_{\text{origin}}$ activations, except sender $h$ which is patched to its $x_{\text{new}}$ activation
3. **Forward Pass**: Run a forward pass on $x_{\text{origin}}$ with frozen/patched heads (MLPs and LayerNorm are recomputed)
4. **Save Receivers**: During this pass, save the activations of receiver components $r \in R$
5. **Final Forward**: Run a final forward pass on $x_{\text{origin}}$, patching receivers $R$ to the saved values, and measure the output change

The output change (e.g., logit difference) quantifies the **causal effect** of the path from sender $h$ through receivers $R$.

## Module Structure

```
path_patching_full/
├── __init__.py           # Package exports
├── utils.py              # IOI dataset utilities and evaluation
├── data_loader.py        # Load pre-generated datasets from data_generation_simple
├── path_patching.py      # Core 5-step algorithm implementation  
├── plotting.py           # Visualization utilities
├── example.py            # Complete working example
└── README.md             # This file
```

### `utils.py`

Provides IOI (Indirect Object Identification) task utilities:

- **`IOIExample`**: Data structure for IOI examples
- **`build_dataset()`**: Generate IOI datasets with templates and names (for custom datasets)
- **`evaluate()`**: Compute accuracy and logit differences
- **`logit_diff_io_s_from_tokens()`**: Calculate logit(IO) - logit(S)
- **`find_token_positions()`**: Locate token positions in sequences

### `data_loader.py`

Loads pre-generated IOI datasets from `data_generation_simple/`:

- **`load_dataset_for_patching()`**: One-step function to load and prepare datasets
- **`load_ioi_pairs()`**: Load pairs from JSON files
- **`prepare_batch_for_patching()`**: Convert pairs to tensors for path patching
- **`get_dataset_path()`**: Get path to dataset files (small/medium/large)
- **`print_pair_examples()`**: Display example pairs for inspection

### `path_patching.py`

Core path patching implementation with full 5-step algorithm:

- **`HeadSpec`**: Specifies an attention head (layer, head)
- **`ReceiverSpec`**: Specifies a receiver component (layer, head, component, position)
- **`path_patch()`**: Execute complete 5-step algorithm for a single path
- **`batch_path_patch()`**: Test multiple sender→receiver paths efficiently
- **Helper functions**:
  - `gather_activations()`: Step 1
  - `create_freeze_and_patch_hooks()`: Step 2  
  - `run_frozen_forward_pass()`: Steps 3-4
  - `run_final_patched_forward_pass()`: Step 5

### `plotting.py`

Visualization tools for path patching results:

- **`save_path_patching_heatmap()`**: Visualize sender→receiver effects matrix
- **`save_heatmap()`**: Generic heatmap plotting
- **`save_attention_heatmap()`**: Attention pattern visualization
- **`save_comparison_plot()`**: Compare multiple data series

### `example.py`

Complete working example demonstrating:

1. Single path testing (L9H9 → L10H0)
2. Batch path testing with heatmap visualization
3. Comparing paths to different receiver components (Q, K, V)

## Quick Start

### Installation

Ensure you have the required dependencies:

```bash
pip install torch transformer-lens matplotlib numpy
```

### Dataset Options

The module loads pre-generated datasets from `data_generation_simple/output/`:

- **small**: 100 pairs (fast for testing)
- **medium**: 500 pairs (balanced)
- **large**: 1000 pairs (comprehensive analysis)

Generate datasets before first use:
```bash
cd data_generation_simple
python generate_ioi_pairs.py
```

### Basic Usage

```python
from transformer_lens import HookedTransformer
from path_patching_full import (
    HeadSpec, 
    ReceiverSpec, 
    path_patch,
    load_dataset_for_patching,
)

# Load model
model = HookedTransformer.from_pretrained("gpt2-small")

# Load pre-generated dataset
clean_tokens, corrupt_tokens, io_toks, s_toks, pairs = load_dataset_for_patching(
    model, size="small", n_examples=50
)

# Define sender and receiver
sender = HeadSpec(layer=9, head=9)  # Source head
receiver = ReceiverSpec(layer=10, head=0, component='q')  # Destination

# Run path patching
effect = path_patch(
    model=model,
    tokens_origin=clean_tokens,      # [batch, seq_len]
    tokens_new=corrupt_tokens,       # [batch, seq_len]
    sender_head=sender,
    receivers=[receiver],
    io_toks=io_toks,                # [batch]
    s_toks=s_toks,                  # [batch]
)

print(f"Path effect: {effect:.3f}")
```

### Generate Dataset (First Time Only)

Before running path patching, generate the IOI dataset:

```bash
cd data_generation_simple
python generate_ioi_pairs.py
```

This creates clean/corrupt pairs with random names in `data_generation_simple/output/`.

### Running the Example

```bash
cd path_patching_full
python example.py
```

This will:
- Load GPT-2 small
- Load 50 pre-generated clean/corrupt IOI pairs from `data_generation_simple/`
- Test various paths from layers 9-10 to layers 10-11
- Save visualization heatmaps to `path_patching_results/`

## Key Concepts

### Sender Head

The **sender** is the attention head whose output we want to trace. We patch it from the clean input to the corrupt input to see where its information flows.

Example: Layer 9 Head 9 in GPT-2 is a "name mover" head that moves name information to the final token position.

```python
sender = HeadSpec(layer=9, head=9)
```

### Receiver Components

**Receivers** are the downstream components we want to measure. They can be:

1. **Attention head inputs** (Q, K, or V):
   ```python
   # Query input of Layer 10, Head 0
   ReceiverSpec(layer=10, head=0, component='q')
   ```

2. **Residual stream**:
   ```python
   # Residual stream at end of Layer 10
   ReceiverSpec(layer=10, component='resid')
   ```

3. **Position-specific**:
   ```python
   # Query input at specific token position
   ReceiverSpec(layer=10, head=0, component='q', position=5)
   ```

### Clean vs Corrupt Data

Path patching requires **contrastive pairs**:

- **Clean**: Input where model makes correct prediction
  - Example: "When John and Mary went to the store, Mary gave a drink to" → John
  
- **Corrupt**: Counterfactual input that breaks the behavior
  - Example: "When Alice and Bob went to the store, Jane gave a drink to" → ?

The corruption helps isolate which components carry task-relevant information.

**Dataset Source**: This implementation uses pre-generated datasets from `data_generation_simple/`, which creates clean/corrupt pairs with random names.

## Interpreting Results

### Logit Difference

The output metric is typically **logit difference**:

```
Δ = logit(IO) - logit(S)
```

Where:
- **IO**: Indirect object (correct answer: "John")
- **S**: Subject (incorrect distractor: "Mary")

### Path Effects

Higher logit difference after path patching indicates:
- The sender→receiver path carries **important information**
- Patching this path transfers behavior from corrupt to clean input
- This path is **causally relevant** for the task

### Example Interpretation

```python
# Test path from L9H9 to L10H0
effect = path_patch(model, clean, corrupt, 
                   sender=HeadSpec(9, 9),
                   receivers=[ReceiverSpec(10, 0, 'q')])
# effect = 2.5

# Test path from L9H9 to L10H1  
effect2 = path_patch(model, clean, corrupt,
                    sender=HeadSpec(9, 9),
                    receivers=[ReceiverSpec(10, 1, 'q')])
# effect2 = 0.1
```

**Interpretation**: L9H9 sends important information to L10H0's query input (effect=2.5), but not to L10H1 (effect=0.1). This suggests L10H0 is downstream of L9H9 in the IOI circuit.

## Advanced Usage

### Testing All Heads

```python
from path_patching import batch_path_patch

# Generate all sender heads in layers 8-9
senders = [HeadSpec(layer=l, head=h) 
           for l in [8, 9] 
           for h in range(12)]

# Generate all receiver heads in layer 10  
receivers = [ReceiverSpec(layer=10, head=h, component='q')
             for h in range(12)]

# Test all paths
effects = batch_path_patch(model, clean, corrupt, 
                          senders, receivers, io_toks, s_toks)
# effects shape: [24, 12] (senders × receivers)

# Visualize
from plotting import save_path_patching_heatmap
save_path_patching_heatmap(effects, filename="all_paths.png")
```

### Custom Receiver Sets

Test multiple receivers simultaneously:

```python
# Test if sender communicates to multiple heads
receivers = [
    ReceiverSpec(layer=10, head=0, component='q'),
    ReceiverSpec(layer=10, head=1, component='q'),
    ReceiverSpec(layer=10, head=2, component='k'),
]

effect = path_patch(model, clean, corrupt, sender, receivers)
```

### Residual Stream Patching

Compare attention vs residual stream:

```python
# Attention head path
attn_effect = path_patch(model, clean, corrupt, sender,
                        [ReceiverSpec(10, 0, 'q')])

# Residual stream path  
resid_effect = path_patch(model, clean, corrupt, sender,
                         [ReceiverSpec(10, component='resid')])

print(f"Attention path: {attn_effect:.3f}")
print(f"Residual path: {resid_effect:.3f}")
```

## Technical Details

### Hook Names

The implementation uses TransformerLens hook points:

- **Attention outputs**: `blocks.{layer}.attn.hook_z`
- **Q/K/V inputs**: `blocks.{layer}.attn.hook_{q/k/v}`
- **Residual stream**: `blocks.{layer}.hook_resid_post`

### Memory Management

Path patching requires:
- Caching activations from 2 forward passes (clean + corrupt)
- Running forward pass with frozen heads
- Storing receiver activations

For large models or long sequences, consider:
- Reducing batch size
- Testing fewer sender/receiver pairs at once
- Using gradient checkpointing

### Computation Cost

For each path test:
- 2 forward passes (gather activations)
- 1 frozen forward pass (save receivers)
- 1 patched forward pass (measure effect)

**Total: 4 forward passes per path**

Batch testing N senders × M receivers = 4×N×M forward passes
