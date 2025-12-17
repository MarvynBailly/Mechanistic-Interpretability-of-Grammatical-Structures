#Head imortance analysis through IFR

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

Computes an **activation frequency matrix** of shape:

