# Simple IOI Dataset Generation

This directory contains a simplified dataset generator for IOI (Indirect Object Identification) path patching experiments.

## What it generates

Clean/corrupt pairs where:
- **Clean**: Standard IOI sentence with correct IO and S names
  - Example: `"When Mary and John went to the store, John gave the book to Mary"`
  
- **Corrupt**: Same sentence structure but with **random names** (not IO/S)
  - Example: `"When Linda and James went to the store, James gave the book to Linda"`

This differs from the name-swapping approach and tests what happens when the model doesn't have the correct name information.

## Usage

```powershell
# Activate environment
. .\ioi-env\Scripts\Activate.ps1

# Generate datasets
py .\data_generation_simple\generate_ioi_pairs.py
```

This will create three files in `data_generation_simple/output/`:
- `ioi_pairs_small.json` - 100 pairs
- `ioi_pairs_medium.json` - 1,000 pairs
- `ioi_pairs_large.json` - 5,000 pairs

## Output Format

```json
{
  "n_examples": 100,
  "description": "Clean/corrupt IOI pairs for path patching",
  "pairs": [
    {
      "clean": "When Mary and John went to the store, John gave the book to",
      "corrupt": "When Linda and James went to the store, James gave the book to",
      "io_name": "Mary",
      "s_name": "John",
      "io_token": " Mary"
    }
  ]
}
```

## Using with path patching

Load and use the generated pairs:

```python
import json
from pathlib import Path

# Load dataset
with open("data_generation_simple/output/ioi_pairs_small.json") as f:
    data = json.load(f)

# Use with path patching
for pair in data["pairs"]:
    clean_text = pair["clean"]
    corrupt_text = pair["corrupt"]
    expected_token = pair["io_token"]
    # ... run path patching experiment
```
