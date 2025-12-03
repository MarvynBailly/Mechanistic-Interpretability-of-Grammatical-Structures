# Dataset Generation for Mechanistic Interpretability

This directory contains dataset generators for various path patching experiments using GPT-2 small. All generators create clean/corrupt pairs designed to isolate specific model behaviors.

## 📁 Available Datasets

### 1. **IOI (Indirect Object Identification)**
- **File**: `generate_ioi_pairs.py`
- **Task**: Identify the indirect object in sentences like "When Mary and John went to the store, John gave the book to [Mary]"
- **Status**: ✅ Works well (~90%+ accuracy)

### 2. **Python Code Completion**
- **File**: `generate_code_pairs.py`
- **Task**: Complete function arguments like `def process(x, y): return x + [y]`
- **Status**: ✅ Works excellently (93-94% accuracy)
- **Variants**: objects, letters, colors

### 3. **Color-Object Association**
- **File**: `generate_color_pairs.py`
- **Task**: Bind colors to objects and retrieve based on preference
- **Status**: ❌ Does not work (0-2% accuracy, model predicts articles)

## 🚀 Quick Start

### Generate IOI Pairs

```python
from generate_ioi_pairs import generate_ioi_dataset

# Generate 100 IOI examples
pairs = generate_ioi_dataset(n_examples=100)

# Saves to: output/ioi_pairs_small.json
```

**Output format:**
```json
{
  "clean": "When Mary and John went to the store, John gave the book to",
  "corrupt": "When Linda and James went to the store, James gave the book to",
  "io_name": "Mary",
  "s_name": "John"
}
```

### Generate Code Completion Pairs

```python
from generate_code_pairs import generate_code_pairs

# Generate with object variable names (ball, cube, etc.)
pairs = generate_code_pairs(var_type="objects", n_examples=100)

# Saves to: output/code_pairs_objects.json
```

**Output format:**
```json
{
  "clean": "def merge(disk, ball):\n    return disk +",
  "corrupt": "def merge(cube, star):\n    return disk +",
  "correct_arg": "ball",
  "incorrect_arg": "disk"
}
```

### Generate Color-Object Pairs

```python
from generate_color_pairs import generate_dataset

# Generate color-object association examples
pairs = generate_dataset(size="small")  # 100 examples

# Saves to: output/color_pairs_small.json
```

**Output format:**
```json
{
  "clean": "The red ball and blue cube are here. I want the red one, so I choose the",
  "corrupt": "The green sphere and yellow pyramid are here. I want the red one, so I choose the",
  "correct_object": "ball",
  "incorrect_object": "cube"
}
```

## 📊 Dataset Sizes

Each generator creates three size variants:

| Size | Examples | File Pattern |
|------|----------|--------------|
| Small | 100 | `*_pairs_small.json` |
| Medium | 1,000 | `*_pairs_medium.json` |
| Large | 5,000 | `*_pairs_large.json` |

## 🎯 Task Validation

Before running expensive path patching analysis, validate that GPT-2 can solve the task:

```python
from load_pairs import load_and_validate_task
from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained("gpt2-small")

# Load and validate
pairs, stats = load_and_validate_task(
    model=model,
    dataset_type="code",  # or "ioi" or "color"
    var_type="objects",   # for code completion
    n_examples=100
)

print(f"Accuracy: {stats['accuracy']*100:.1f}%")
print(f"Logit difference: {stats['logit_diff']:.3f}")
```

## 📝 Design Principles

### Clean/Corrupt Pairs

Each dataset follows the same pattern:

1. **Clean**: Contains correct information for task
2. **Corrupt**: Violates a key constraint while keeping structure similar
3. **Effect**: Measures how much model performance degrades

### Token Selection

All tokens are chosen to be:
- **Single-token**: Avoids multi-token complications
- **Unambiguous**: Clear semantic meaning
- **Balanced**: No frequency biases between correct/incorrect

### Template Diversity

Multiple templates prevent:
- Memorization of specific phrases
- Positional biases
- Overfitting to syntax

## 🔍 Example Usage

### Complete Workflow

```python
from generate_code_pairs import generate_code_pairs
from load_pairs import load_and_validate_task
from transformer_lens import HookedTransformer

# 1. Generate dataset
pairs = generate_code_pairs(var_type="objects", n_examples=100)

# 2. Load model
model = HookedTransformer.from_pretrained("gpt2-small")

# 3. Validate task
pairs, stats = load_and_validate_task(
    model, "code", var_type="objects", n_examples=100
)

if stats['accuracy'] > 0.5:
    print("✅ Task is solvable! Ready for path patching.")
else:
    print("❌ Task validation failed.")
```

## 📂 Output Directory Structure

```
data_generation/
├── output/
│   ├── ioi_pairs_small.json          # IOI: 100 examples
│   ├── ioi_pairs_medium.json         # IOI: 1,000 examples
│   ├── ioi_pairs_large.json          # IOI: 5,000 examples
│   ├── code_pairs_objects.json       # Code: object variables
│   ├── code_pairs_letters.json       # Code: letter variables
│   ├── code_pairs_colors.json        # Code: color variables
│   ├── color_pairs_small.json        # Color: 100 examples
│   ├── color_pairs_medium.json       # Color: 1,000 examples
│   └── color_pairs_large.json        # Color: 5,000 examples
├── generate_ioi_pairs.py             # IOI generator
├── generate_code_pairs.py            # Code completion generator
├── generate_color_pairs.py           # Color-object generator
├── load_pairs.py                     # Universal data loader
├── example_usage.py                  # Usage examples
└── README.md                         # This file
```

## 🔬 Task Comparison

| Task | Accuracy | Top Head | Effect | Notes |
|------|----------|----------|--------|-------|
| **Code (objects)** | 94% | L9H6 | +2.81 | ✅ Excellent - Argument Mover Heads |
| **Code (letters)** | 93% | L10H10 | +2.45 | ✅ Excellent - Similar circuit |
| **IOI** | ~90% | Various | Strong | ✅ Classic task - Name Mover Heads |
| **Color-Object** | 1% | - | - | ❌ Model predicts articles instead |
| **Code (colors)** | 18% | - | - | ⚠️ Partial - Multi-token issue |

## 🛠️ Utilities

### Load Pairs

```python
from load_pairs import load_pairs

# Load any dataset type
pairs = load_pairs("code", var_type="objects")
pairs = load_pairs("ioi", size="large")
pairs = load_pairs("color", size="small")
```

### Validate Task

```python
from load_pairs import validate_task_on_model

# Check if model can solve task
stats = validate_task_on_model(model, pairs)
print(f"Accuracy: {stats['accuracy']*100:.1f}%")
```

## 📖 Citation

Based on methodologies from:
- **IOI Task**: Wang et al. (2023) "Interpretability in the Wild"
- **Path Patching**: Conmy et al. (2023) "Towards Automated Circuit Discovery"
- **Code Completion**: Novel task for mechanistic interpretability

## 🔗 Related Directories

- **`color_object_association/`**: Analysis framework and path patching
- **`path_patching_full/`**: Original IOI path patching implementation
- **`path_patching_residual/`**: Residual stream path patching

## ⚙️ Requirements

```bash
pip install transformer-lens torch numpy
```

## 🎓 Educational Notes

### Why Clean/Corrupt Pairs?

Path patching requires:
1. **Clean run**: Model performs task correctly
2. **Corrupt run**: Model fails due to missing information
3. **Patched run**: Restore specific component from clean to measure its effect

### What Makes a Good Task?

✅ **Model can solve it** (>50% accuracy on clean)  
✅ **Strong corruption effect** (large logit difference drop)  
✅ **Clear causal structure** (easy to interpret what heads do)  
✅ **Single-token answers** (avoids tokenization issues)

### Common Pitfalls

❌ **Multi-token words**: "arg1" → [" arg", "1"] causes confusion  
❌ **Positional biases**: Always put answer in same position  
❌ **Template memorization**: Use diverse sentence structures  
❌ **Frequency effects**: Balance common/rare words

## 📧 Questions?

See the main repository README or check the `color_object_association/` directory for complete analysis examples.
