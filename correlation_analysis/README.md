# Head imortance analysis through IFR

This directory contains scripts to reproduce a core analysis from  
**“The Same but Different: Cross-Lingual Mechanistic Interpretability of Large Language Models” (ICLR 2025)**.

The goal is to measure **which attention heads are causally important** for a task in different languages, and to quantify how similar those importance patterns are across languages.

---

## Overview

We perform the analysis in two steps:

1. **Compute per-head activation frequencies** using an IFR-style causal ablation method.
2. **Compute correlation coefficients** between activation-frequency matrices from different languages.

Each attention head is treated as one unit of analysis.

---

## Files

### `activation_frequencies_ifr.py`

Computes an **activation frequency matrix** of shape [n_layers, n_heads] .


Each entry represents the fraction of prompts for which a given attention head is *active*, where activity is defined via a **causal intervention**:

> A head is active if ablating it causes the task-relevant logit difference to change by more than a threshold τ.

#### Method (high-level)

For each prompt and each attention head:
1. Run the model normally and compute a task-specific scalar (e.g. logit difference).
2. Ablate the head by removing its contribution from the residual stream.
3. Recompute the scalar.
4. Mark the head as active if the absolute difference exceeds τ.

Activation frequency is the fraction of prompts where this condition holds.

This procedure approximates **Information Flow Routes (IFR)** from the paper.

#### Output
- `.npy`: activation frequency matrix `[n_layers, n_heads]`
- (optional) `.json`: metadata (model, \tau, dataset size, etc.)

---

### `crosslingual_head_corr.py`

Computes **correlation statistics** between two activation-frequency matrices (e.g. English vs Chinese).

Each attention head contributes **one paired data point**.

#### Metrics
- **Pearson correlation** (linear agreement of magnitudes)
- **Spearman correlation** (agreement in ranking / ordering)

#### Output
- Printed correlation statistics
- Optional `.json` summary



