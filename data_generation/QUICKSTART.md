# Quick Start Guide

Get started with the data generation pipeline in 5 minutes!

## Installation

```bash
# Navigate to the data generation directory
cd Mechanistic-Interpretability-of-Grammatical-Structures/data_generation

# Install dependencies (optional - pipeline uses Python standard library)
pip install -r requirements.txt
```

## Generate Your First Dataset

### Option 1: Use Default Templates and Words (Fastest)

```bash
# Run the pipeline with included templates and words
python generate_dataset.py
```

This will create an `output/` directory with:
- ✅ 900 clean sentence pairs
- ✅ 2,700 fuzzy variants
- ✅ 900 path patching pairs
- ✅ Dataset statistics

### Option 2: Generate Custom Word List

```bash
# Generate a new word list from common English words
python fetch_words_from_hf.py

# This creates fetched_words.json with placeholders for Chinese translations
```

## Translate to Chinese

### Automated Translation (with OpenAI API)

```bash
# Set your OpenAI API key
export OPENAI_API_KEY='your-api-key-here'  # Linux/Mac
set OPENAI_API_KEY=your-api-key-here       # Windows

# Run the translation helper
python translate_helper.py
# Choose option 3 (Both) to translate words and templates
```

### Manual Translation

```bash
# Create a translation template
python translate_helper.py
# Choose option 4 (Create manual translation template)

# Copy the generated template to ChatGPT or send to a translator
# Update words.json or templates.json with the translations
```

## Verify Translations

**IMPORTANT**: Always have translations verified by a native Chinese speaker!

```bash
# Create a verification checklist
python translate_helper.py
# Choose option 5 (Create verification checklist)

# Share the checklist with a native speaker
```

## Test the Pipeline

```bash
# Run the test suite to verify everything works
python test_pipeline.py
```

Expected output:
```
✓ templates.json exists
✓ words.json exists
✓ Generated 900 clean pairs (60 words × 15 templates)
✓ Generated 2,700 fuzzy pairs (900 × 3 fuzzy types)
✓ Generated 900 path patching pairs
✓ All tests passed!
```

## Use the Generated Data

Your generated datasets are in the `output/` directory:

```python
import json

# Load clean pairs for baseline analysis
with open('output/clean_pairs.json', 'r', encoding='utf-8') as f:
    clean_pairs = json.load(f)

# Load path patching pairs for mechanistic interpretability
with open('output/path_patching_pairs.json', 'r', encoding='utf-8') as f:
    patching_pairs = json.load(f)

# Example: Access first pair
first_pair = patching_pairs[0]
print("Clean English:", first_pair['clean']['english'])
print("Clean Chinese:", first_pair['clean']['chinese'])
print("Corrupted English:", first_pair['corrupted']['english'])
```

## Common Workflows

### Workflow 1: Default Dataset (5 minutes)
```bash
python generate_dataset.py
```
Done! Use the output files immediately.

### Workflow 2: Custom Words + Auto Translation (15 minutes)
```bash
python fetch_words_from_hf.py
export OPENAI_API_KEY='your-key'
python translate_helper.py  # Choose option 1
python generate_dataset.py
```

### Workflow 3: Fully Custom (30 minutes)
```bash
# 1. Edit templates.json manually
# 2. Edit words.json manually
# 3. Translate via ChatGPT
# 4. Verify with native speaker
# 5. Generate dataset
python generate_dataset.py
```

## Expected Results

According to the project proposal:

| Metric | Value |
|--------|-------|
| Templates | 15 |
| Words | 60 |
| Clean Pairs | 900 (15 × 60) |
| Fuzzy Pairs | 2,700 (900 × 3) |
| Path Patching Pairs | 900 |
| Forward Passes (12L×12H) | ~259,200 |

## Next Steps

After generating your dataset:

1. ✅ **Verify data quality**: Check a few random samples
2. ✅ **Run tests**: `python test_pipeline.py`
3. ✅ **Integrate with model**: Use the path patching pairs for experiments
4. ✅ **Document**: Note any customizations for reproducibility

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "File not found" error | Ensure you're in the `data_generation/` directory |
| Chinese characters garbled | Use UTF-8 encoding when viewing files |
| Wrong number of pairs | Check templates.json and words.json counts |
| Import errors | Install dependencies: `pip install -r requirements.txt` |

## Quick Reference

```bash
# Generate dataset
python generate_dataset.py

# Generate words
python fetch_words_from_hf.py

# Translate
python translate_helper.py

# Test
python test_pipeline.py
```

## Need Help?

- 📖 Read the full [README.md](README.md)
- 📄 Check the [project proposal](../Project_Proposal/project_proposal.pdf)
- 🐛 Run tests: `python test_pipeline.py`

Happy data generation! 🚀