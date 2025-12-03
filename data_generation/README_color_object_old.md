# Color-Object Association Dataset Generator

This module generates datasets for path patching experiments on the **Color-Object Association Task** using GPT-2 small, following mechanistic interpretability methodologies.

## Overview

The dataset consists of clean/corrupt pairs designed to test whether language models can:
1. **Bind** color-object associations in context
2. **Identify** preferred colors
3. **Retrieve** objects associated with preferred colors

### Task Structure

**Clean Example:**
```
"The red ball and blue cube are here. I prefer the red, so I'll take the"
→ Expected: "ball" (not "cube")
```

**Corrupt Example:**
```
"The green sphere and yellow pyramid are here. I prefer the red, so I'll take the"
→ Expected: Model should be confused (preferred color "red" not in context)
```

## Dataset Components

### Colors (Single GPT-2 Tokens)
`red`, `blue`, `green`, `yellow`, `purple`, `orange`, `pink`, `brown`, `black`, `white`, `gray`, `silver`

### Objects (Single GPT-2 Tokens)
`ball`, `cube`, `sphere`, `pyramid`, `cylinder`, `cone`, `prism`, `disk`, `block`, `box`, `ring`, `star`

### Templates
Four template variations to ensure model learns task structure (not memorization):

1. `"The {color1} {object1} and {color2} {object2} are here. I prefer the {pref_color}, so I'll take the"`
2. `"I see a {color1} {object1} and a {color2} {object2}. I prefer the {pref_color}, so I'll choose the"`
3. `"There's a {color1} {object1} and a {color2} {object2}. I like the {pref_color}, so I want the"`
4. `"A {color1} {object1} and a {color2} {object2} are available. I prefer the {pref_color}, so I'll pick the"`

## Usage

### Quick Start

```python
from generate_color_pairs import generate_dataset

# Generate 100 examples
pairs = generate_dataset(
    n_examples=100,
    output_file="output/my_dataset.json",
    seed=42
)

# Access generated pairs
for pair in pairs:
    print(f"Clean: {pair.clean_text}")
    print(f"Expected: {pair.correct_object}")
```

### Generate Standard Datasets

```bash
python generate_color_pairs.py
```

This creates three datasets:
- `output/color_pairs_small.json` (100 examples)
- `output/color_pairs_medium.json` (500 examples)
- `output/color_pairs_large.json` (1000 examples)

### View Examples

```python
from generate_color_pairs import print_example_pairs

# Print 5 example pairs
print_example_pairs(5)
```

### Custom Dataset Generation

```python
from generate_color_pairs import generate_dataset, TEMPLATES

# Generate with custom parameters
pairs = generate_dataset(
    n_examples=200,
    templates=TEMPLATES[:2],  # Use only first 2 templates
    output_file="output/custom.json",
    seed=123
)
```

## Dataset Format

Each pair is a JSON object with the following structure:

```json
{
  "clean": "The red ball and blue cube are here. I prefer the red, so I'll take the",
  "corrupt": "The green sphere and yellow pyramid are here. I prefer the red, so I'll take the",
  "correct_object": "ball",
  "incorrect_object": "cube",
  "preferred_color": "red",
  "color1": "red",
  "color2": "blue",
  "object1": "ball",
  "object2": "cube"
}
```

### Field Descriptions

| Field | Description |
|-------|-------------|
| `clean` | Clean sentence with preferred color in context |
| `corrupt` | Corrupt sentence with preferred color NOT in context |
| `correct_object` | Object paired with preferred color (expected answer) |
| `incorrect_object` | The other object (distractor) |
| `preferred_color` | The color mentioned in "I prefer the X" |
| `color1`, `color2` | Colors in clean sentence |
| `object1`, `object2` | Objects in clean sentence |

## Path Patching Experiments

### Metric: Logit Difference

```python
logit_diff = logit(correct_object) - logit(incorrect_object)
```

- **Clean**: High logit_diff → model prefers correct object
- **Corrupt**: ~0 logit_diff → model is confused
- **Patched**: Recovery indicates head importance

### Experimental Workflow

1. **Load Dataset**
   ```python
   import json
   with open('output/color_pairs_small.json', 'r') as f:
       pairs = json.load(f)
   ```

2. **Run Model on Clean**
   ```python
   clean_logits = model(pairs[0]['clean'])
   clean_logit_diff = (
       clean_logits[pairs[0]['correct_object']] - 
       clean_logits[pairs[0]['incorrect_object']]
   )
   ```

3. **Run Model on Corrupt**
   ```python
   corrupt_logits = model(pairs[0]['corrupt'])
   corrupt_logit_diff = (
       corrupt_logits[pairs[0]['correct_object']] - 
       corrupt_logits[pairs[0]['incorrect_object']]
   )
   ```

4. **Path Patching**
   - Cache activations from clean run
   - Run corrupt with patched activations from clean
   - Measure logit difference recovery

## Examples

See `example_usage.py` for comprehensive examples:

```bash
python example_usage.py
```

This demonstrates:
- Custom dataset generation
- Dataset statistics and validation
- Loading and using generated datasets
- Single pair generation for debugging
- Available colors, objects, and templates

## Design Choices

### Why These Colors/Objects?

- **Single tokens**: Each color/object is a single token in GPT-2's vocabulary
- **Concrete nouns**: Easy for model to bind with colors
- **Unambiguous**: Clear semantic categories (no overlap between colors/objects)

### Why This Task?

- **Tests compositional reasoning**: Model must bind, retrieve, and output
- **Clear ground truth**: Unambiguous correct answer
- **Interpretable circuit**: Expected circuit structure (see `../color_object_association/README.md`)

### Clean vs Corrupt Design

- **Clean**: Preferred color IS in context → model can solve task
- **Corrupt**: Preferred color NOT in context → model should fail
- **Key insight**: Difference reveals which heads carry critical information

## Integration with Path Patching Code

This dataset is designed to work with the path patching framework in `../path_patching_residual/`:

```python
from data_generation_color.generate_color_pairs import generate_dataset
from path_patching_residual.IOI_pathpatching import run_path_patching

# Generate dataset
pairs = generate_dataset(n_examples=100)

# Convert to format expected by path patching code
clean_prompts = [p.clean_text for p in pairs]
corrupt_prompts = [p.corrupt_text for p in pairs]
correct_tokens = [p.correct_object for p in pairs]
incorrect_tokens = [p.incorrect_object for p in pairs]

# Run path patching experiments
results = run_path_patching(
    clean_prompts=clean_prompts,
    corrupt_prompts=corrupt_prompts,
    correct_tokens=correct_tokens,
    incorrect_tokens=incorrect_tokens
)
```

## Validation

### Sanity Checks

The generator ensures:
- ✅ All colors and objects are unique within each pair
- ✅ Corrupt sentences use different colors/objects than clean
- ✅ Preferred color appears in clean but NOT in corrupt
- ✅ Balanced distribution (50% prefer first color, 50% prefer second)
- ✅ Random template selection for variety

### Manual Inspection

```python
from generate_color_pairs import print_example_pairs

# Print 10 random examples to verify quality
print_example_pairs(10)
```

## File Structure

```
data_generation_color/
├── generate_color_pairs.py    # Main generator
├── example_usage.py            # Usage examples
├── load_pairs.py              # Utility to load and inspect pairs
├── README.md                  # This file
└── output/                    # Generated datasets
    ├── color_pairs_small.json
    ├── color_pairs_medium.json
    └── color_pairs_large.json
```

## Next Steps

After generating datasets:

1. **Verify Model Performance**
   - Check that GPT-2 gets high accuracy on clean sentences
   - Verify low accuracy on corrupt sentences

2. **Run Path Patching**
   - Identify Object Mover Heads (direct effect on logits)
   - Find Association Heads (bind color-object pairs)
   - Locate Preference Detection Heads (identify preferred color)
   - Discover Color Retriever Heads (lookup object for color)

3. **Visualize Results**
   - Create heatmaps of head importance
   - Plot attention patterns
   - Compare with IOI circuit structure

See `../color_object_association/README.md` for the full experimental protocol.

## References

This dataset generator follows the design principles from:
- **IOI Task**: Wang et al. (2022) "Interpretability in the Wild"
- **Path Patching**: Methods from TransformerLens library
- **Mechanistic Interpretability**: Elhage et al. (2021) "A Mathematical Framework"
