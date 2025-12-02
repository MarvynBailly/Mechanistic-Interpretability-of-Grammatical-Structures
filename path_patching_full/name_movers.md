# Figure 3b Recreation: Step-by-Step Code Walkthrough

## Overview
This document explains how we reproduce Figure 3b from "Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small" (Wang et al., 2022) by walking through the code step-by-step. We use **path patching** to identify which attention heads directly contribute to solving the IOI task.

---

## Step 1: Setup and Data Loading

### Loading the Model
```python
MODEL_NAME = "gpt2-small"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = HookedTransformer.from_pretrained(MODEL_NAME, device=device)
```

**What this does:**
- Loads GPT-2 small (12 layers, 12 heads per layer, 768 dimensional embeddings)
- `HookedTransformer` is a wrapper that lets us intercept activations at any point in the network
- We'll use this to "hook" into specific attention heads

### Loading the Dataset
```python
clean_tokens, corrupt_tokens, io_toks, s_toks, _ = load_dataset_for_patching(
    model, size=DATASET_SIZE, n_examples=N_EXAMPLES
)
```

**What this returns:**
- `clean_tokens`: IOI prompts like "When Lisa and David went to the store, David gave the book to"
  - Shape: `[batch=10, seq_len=14]`
- `corrupt_tokens`: ABC prompts with random names like "When Patricia and Robert went to the store, Marry gave the book to"
  - Shape: `[batch=10, seq_len=14]`
- `io_toks`: Token IDs for IO names (e.g., ID for " Lisa")
  - Shape: `[batch=10]`
- `s_toks`: Token IDs for S names (e.g., ID for " David")
  - Shape: `[batch=10]`

**Why we need both clean and corrupt:**
- Clean prompts contain the correct answer (IO name)
- Corrupt prompts have random names, so the model can't solve the task
- By comparing them, we isolate which heads carry IOI-specific information

---

## Step 2: Computing Baseline Performance

### Clean Baseline
```python
with torch.inference_mode():
    baseline_logits = model(clean_tokens)[:, -1, :]
    batch_idx = torch.arange(len(io_toks), device=device)
    baseline_io = baseline_logits[batch_idx, io_toks]
    baseline_s = baseline_logits[batch_idx, s_toks]
    baseline_diff = (baseline_io - baseline_s).mean().item()
```

**Line-by-line breakdown:**

1. `model(clean_tokens)` - Forward pass through GPT-2
   - Input: `[10, 14]` token IDs
   - Output: `[10, 14, 50257]` logits for all tokens at all positions

2. `[:, -1, :]` - Extract only the last position
   - Result: `[10, 50257]` - logits for the final prediction position
   - This is where the model predicts the answer: "gave the book to ___"

3. `baseline_logits[batch_idx, io_toks]` - Extract IO token logits
   - For batch item 0: get `logits[0, io_toks[0]]` (logit for " Lisa")
   - For batch item 1: get `logits[1, io_toks[1]]` (logit for next IO name)
   - Result: `[10]` - one logit per sample

4. `baseline_logits[batch_idx, s_toks]` - Extract S token logits
   - Same process but for S names (" David", etc.)
   - Result: `[10]`

5. `(baseline_io - baseline_s).mean()` - Calculate average logit difference
   - Result: `2.538`
   - **Interpretation**: Model strongly prefers IO over S (correct behavior!)

### Corrupt Baseline
```python
with torch.inference_mode():
    corrupt_logits = model(corrupt_tokens)[:, -1, :]
    corrupt_io = corrupt_logits[batch_idx, io_toks]
    corrupt_s = corrupt_logits[batch_idx, s_toks]
    corrupt_diff = (corrupt_io - corrupt_s).mean().item()
```

**What this shows:**
- Result: `0.158`
- Model barely distinguishes between IO and S with random names
- **Corruption effect**: `2.538 - 0.158 = 2.380` drop in performance
- This proves the task relies on name-specific information, not just syntax

---

## Step 3: Path Patching - The Core Algorithm

This is where we test each head individually. The function `path_patch_head_to_logits()` implements the paper's methodology.

### Phase 1: Cache Clean Head Activations

```python
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
```

**What this does:**

1. **Hook definition**: `blocks.{layer}.attn.hook_z`
   - Example: `blocks.9.attn.hook_z` for layer 9
   - This is the attention output **before** multiplying by W_O matrix
   - Shape: `[batch=10, seq_len=14, n_heads=12, d_head=64]`

2. **Cache function**: `cache_clean_z(activation, hook)`
   - Intercepts activations during forward pass
   - `activation[:, :, head, :]` - Extract only one specific head (e.g., head 9)
   - `.clone()` - Make a copy so we can use it later
   - Saves to dictionary: `cache['z']`

3. **Run with hooks**: `model.run_with_hooks(clean_tokens, ...)`
   - Runs normal forward pass
   - But calls our function when reaching the hook point
   - We extract and save activations for the specific head we're testing

4. **Result**: `clean_z` shape `[10, 14, 64]`
   - Activations from one head, on clean input
   - 10 samples, 14 positions, 64 dimensions per head

**Why we do this:**
- These activations represent what this head "saw" and "understood" from the clean IOI prompt
- We'll insert these into the corrupt run to see if they carry important information

### Phase 2: Get Corrupt Baseline

```python
with torch.inference_mode():
    corrupt_logits = model(corrupt_tokens)[:, -1, :]
    corrupt_io = corrupt_logits[batch_idx, io_toks]
    corrupt_s = corrupt_logits[batch_idx, s_toks]
    corrupt_diff = (corrupt_io - corrupt_s).mean().item()
```

**What this measures:**
- How well the model performs on corrupt (ABC) prompts
- Result: ~0.158 (very low logit difference)
- This is our baseline for "model doesn't know the answer"

### Phase 3: Patch and Measure Recovery

```python
def patch_z(activation, hook):
    # Replace this head's z with clean version
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
```

**The magic happens here:**

1. **Patch function**: `patch_z(activation, hook)`
   - During forward pass on **corrupt** tokens
   - When we reach this head's activations
   - We **replace** them with the saved clean activations
   - Everything else processes normally

2. **What this simulates:**
   - "What if this one head saw the clean prompt, but everything else saw corrupt?"
   - If this head carries important IOI information, patching should help
   - If it's irrelevant, patching won't change anything

3. **Compute patched logit difference:**
   - Extract IO and S logits from patched run
   - Calculate difference: `patched_io - patched_s`
   - Result: Some value between corrupt (0.158) and clean (2.538)

### Phase 4: Calculate Effect

```python
effect = patched_diff - corrupt_diff
return effect
```

**Interpretation:**

- **Positive effect** (e.g., +1.2176 for L9H9):
  - Patching recovered performance significantly
  - `patched_diff` ≈ 1.38 vs `corrupt_diff` ≈ 0.16
  - This head carries critical IOI information
  - → **Name Mover Head**

- **Negative effect** (e.g., -0.5215 for L10H7):
  - Patching actually hurt performance!
  - `patched_diff` ≈ -0.36 vs `corrupt_diff` ≈ 0.16
  - This head's clean activations work against the task
  - → **Negative Name Mover Head**

- **Near-zero effect** (e.g., +0.0016 for L6H9):
  - Patching made almost no difference
  - This head is irrelevant for IOI task

---

## Step 4: Computing All Heads

```python
def compute_all_path_patching_effects(model, clean_tokens, corrupt_tokens, io_toks, s_toks):
    n_layers = model.cfg.n_layers  # 12
    n_heads = model.cfg.n_heads    # 12
    
    effects = torch.zeros(n_layers, n_heads)
    
    for layer in range(n_layers):
        for head in range(n_heads):
            effect = path_patch_head_to_logits(
                model, clean_tokens, corrupt_tokens, io_toks, s_toks, layer, head
            )
            effects[layer, head] = effect
    
    return effects
```

**What this does:**
- Tests all 144 heads (12 layers × 12 heads)
- For each head: run the full path patching procedure
- Stores result in matrix: `effects[layer, head]`
- Result shape: `[12, 12]` - ready for heatmap visualization

**Example results:**
```
effects[9, 9] = 1.2176   # L9H9 - Strong Name Mover
effects[10, 0] = 0.8908  # L10H0 - Strong Name Mover
effects[9, 6] = 0.4601   # L9H6 - Moderate Name Mover
effects[10, 7] = -0.7048 # L10H7 - Strong Negative Name Mover
effects[11, 10] = -0.4803 # L11H10 - Negative Name Mover
```

---

## Step 5: Results Analysis

### Path Patching Results

```python
flat_effects = effects.flatten()
flat_indices = flat_effects.argsort(descending=True)

print(f"\nTop 10 Name Mover Heads:")
for i in range(10):
    idx = flat_indices[i].item()
    layer = idx // model.cfg.n_heads
    head = idx % model.cfg.n_heads
    effect = flat_effects[idx].item()
    print(f"  {i+1}. L{layer}H{head}: {effect:.4f}")
```

**What this does:**

1. `effects.flatten()` - Convert `[12, 12]` matrix to `[144]` vector
   - Index 0 = L0H0, Index 1 = L0H1, ..., Index 108 = L9H0, Index 109 = L9H9

2. `argsort(descending=True)` - Sort by effect magnitude
   - Returns indices sorted from highest to lowest effect
   - Example: `[109, 120, 114, ...]` means L9H9, L10H0, L9H6 are top

3. **Convert back to layer/head:**
   - `layer = idx // 12` - Integer division gives layer number
   - `head = idx % 12` - Remainder gives head number
   - Example: idx=109 → layer=9, head=9 → L9H9

**Results:**
```
Top 10 Name Mover Heads:
  1. L9H9: 1.2176    ← Strongest contributor (recovers 51% of lost logit diff)
  2. L10H0: 0.8908   ← Second strongest (recovers 37%)
  3. L9H6: 0.4601    ← Third strongest (recovers 19%)
  4. L10H10: 0.2660
  5. L8H10: 0.2362
```

**Paper predictions: 9.9, 10.0, 9.6**