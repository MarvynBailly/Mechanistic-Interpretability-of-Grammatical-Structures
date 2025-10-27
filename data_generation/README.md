# Data Generation Pipeline

This directory contains the data generation pipeline for the **Mechanistic Interpretability of Grammatical Structures** project. The pipeline generates English-Chinese parallel datasets for studying how multilingual language models process grammatical structures.

## Overview

Based on the project proposal, this pipeline generates:
- **900 sentence pairs** (60 words × 15 templates)
- **Clean variants** for baseline analysis
- **Fuzzy variants** for robustness testing (missing function words, shuffled order, misspellings)
- **Path patching pairs** for mechanistic interpretability experiments

## Project Structure

```
data_generation/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── generate_dataset.py          # Main data generation pipeline
├── fetch_words_from_hf.py      # Word fetching from Hugging Face/common words
├── templates.json              # 15 IOI sentence templates
├── words.json                  # 60 words with Chinese translations
└── output/                     # Generated datasets (created on first run)
    ├── clean_pairs.json
    ├── fuzzy_pairs.json
    ├── path_patching_pairs.json
    └── dataset_stats.json
```

## Features

### 1. Template-Based Generation
- 15 IOI (Indirect Object Identification) templates
- Each template contains a placeholder `{word}` for target words
- Parallel English-Chinese templates for cross-linguistic analysis

### 2. Word Categories
The pipeline supports multiple grammatical features:
- **Singular nouns** (e.g., "book", "pen", "apple")
- **Plural nouns** (e.g., "books", "files", "notes")
- **Past tense verbs with -ed** (e.g., "walked", "talked", "worked")
- **Abstract nouns** (e.g., "information", "knowledge", "advice")
- **Concrete nouns** (e.g., "laptop", "tablet", "charger")

### 3. Fuzzy Variants
Three types of grammatical degradation:
- **Missing function words**: Removes articles, prepositions, conjunctions
- **Shuffled word order**: Randomizes word positions
- **Misspellings**: Introduces typos by swapping adjacent characters

### 4. Path Patching Support
Generates clean-corrupted pairs for causal intervention experiments:
- Each pair contains one clean and one corrupted variant
- Designed for 270,000 forward passes (900 pairs × 2 runs × 12 layers × 12 heads)
- Embarrassingly parallel for GPU optimization

## Installation

1. **Clone the repository** (if you haven't already):
```bash
cd Mechanistic-Interpretability-of-Grammatical-Structures/data_generation
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

Generate the complete dataset with default templates and words:

```bash
python generate_dataset.py
```

This will create an `output/` directory with:
- `clean_pairs.json` - 900 grammatical sentence pairs
- `fuzzy_pairs.json` - 2,700 degraded variants (900 × 3 types)
- `path_patching_pairs.json` - 900 clean-corrupted pairs
- `dataset_stats.json` - Summary statistics

### Step-by-Step Workflow

#### 1. Customize Templates (Optional)

Edit `templates.json` to add or modify sentence templates:

```json
{
  "templates": [
    {
      "id": 0,
      "english": "When John and Mary went to the store, John gave the {word} to Mary.",
      "chinese": "当约翰和玛丽去商店时，约翰把{word}给了玛丽。",
      "description": "IOI template with two names"
    }
  ]
}
```

#### 2. Customize Words (Optional)

Edit `words.json` to modify the word list:

```json
{
  "words": [
    {
      "id": 0,
      "word": "book",
      "grammatical_feature": "singular_noun",
      "pos": "noun"
    }
  ],
  "chinese_translations": {
    "book": "书"
  }
}
```

#### 3. Generate Words from Hugging Face (Alternative)

Instead of using the default word list, you can generate words automatically:

```bash
python fetch_words_from_hf.py
```

This creates `fetched_words.json` with placeholder Chinese translations that need to be filled in.

#### 4. Translate and Verify

**Important**: As per the project proposal:
1. Use ChatGPT to translate English words/templates to Chinese
2. **Have translations manually verified by a native Chinese speaker**
3. Update the `chinese_translations` dictionary

#### 5. Generate Dataset

Run the main pipeline:

```bash
python generate_dataset.py
```

## Output Format

### Clean Pairs (`clean_pairs.json`)

```json
[
  {
    "template_id": 0,
    "word_id": 0,
    "english": "When John and Mary went to the store, John gave the book to Mary.",
    "chinese": "当约翰和玛丽去商店时，约翰把书给了玛丽。",
    "word": "book",
    "grammatical_feature": "singular_noun",
    "is_fuzzy": false,
    "fuzzy_type": null
  }
]
```

### Fuzzy Pairs (`fuzzy_pairs.json`)

```json
[
  {
    "template_id": 0,
    "word_id": 0,
    "english": "John Mary went store John gave book Mary",
    "chinese": "约翰 玛丽 去 商店 约翰 书 给了 玛丽",
    "word": "book",
    "grammatical_feature": "singular_noun",
    "is_fuzzy": true,
    "fuzzy_type": "missing_function_words"
  }
]
```

### Path Patching Pairs (`path_patching_pairs.json`)

```json
[
  {
    "pair_id": 0,
    "clean": {
      "english": "When John and Mary went to the store, John gave the book to Mary.",
      "chinese": "当约翰和玛丽去商店时，约翰把书给了玛丽。"
    },
    "corrupted": {
      "english": "Mary the to book gave John store the to went Mary and John When.",
      "chinese": "玛丽 把 书 给了 约翰 商店 时 约翰 和 玛丽 去 当。"
    },
    "metadata": {
      "template_id": 0,
      "word_id": 0,
      "word": "book",
      "grammatical_feature": "singular_noun"
    }
  }
]
```

## Computational Requirements

As specified in the project proposal:

- **Dataset size**: 900 pairs
- **Path patching**: 2 runs per pair
- **Model**: 12 layers × 12 heads
- **Forward passes**: 900 × 2 × 12 × 12 = **270,000 total**
- **Recommended GPU**: RTX 4070 or equivalent
- **Parallelization**: Embarrassingly parallel across the 900 pairs

## Customization

### Adding New Grammatical Features

1. Add words with new features to `words.json`:
```json
{
  "word": "running",
  "grammatical_feature": "gerund",
  "pos": "verb"
}
```

2. Update `fetch_words_from_hf.py` to add filtering functions for the new pattern

### Modifying Fuzzy Variants

Edit the fuzzy variant methods in `generate_dataset.py`:
- `create_fuzzy_variant_missing_function_words()`
- `create_fuzzy_variant_shuffled()`
- `create_fuzzy_variant_misspelled()`

### Changing Dataset Size

Modify the constants in your templates/words files:
- **15 templates** (can be increased/decreased)
- **60 words** (can be adjusted)
- Total pairs = templates × words

## Data Quality Checklist

Before using the generated dataset:

- [ ] Templates are grammatically correct in both languages
- [ ] Chinese translations verified by native speaker
- [ ] Word-template combinations make semantic sense
- [ ] Fuzzy variants are appropriately degraded
- [ ] Path patching pairs contain meaningful contrasts
- [ ] Dataset statistics match expectations (900 clean pairs, etc.)

## Integration with Main Project

The generated datasets can be used for:

1. **Path Patching Experiments**: Use `path_patching_pairs.json`
2. **Baseline Analysis**: Use `clean_pairs.json`
3. **Robustness Testing**: Use `fuzzy_pairs.json`
4. **Cross-linguistic Comparison**: Compare English vs Chinese processing

## Troubleshooting

### Issue: Missing templates.json or words.json
**Solution**: The pipeline includes default files. Ensure you're running from the correct directory.

### Issue: Chinese characters not displaying correctly
**Solution**: Files use UTF-8 encoding. Ensure your text editor/terminal supports UTF-8.

### Issue: Want more/fewer words
**Solution**: Run `fetch_words_from_hf.py` and adjust `max_words` parameter, or manually edit `words.json`.

### Issue: Need different fuzzy variants
**Solution**: Modify the fuzzy generation methods in `generate_dataset.py` or add new fuzzy types.

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@misc{grammatical-structure-interpretability,
  title={Mechanistic Interpretability of Grammatical Structures},
  author={[Your Team]},
  year={2024},
  note={Data generation pipeline for cross-linguistic mechanistic interpretability}
}
```

## References

- Project Proposal: `../Project_Proposal/project_proposal.pdf`
- Zyzik et al. (2025): Cross-linguistic structural representations with path patching
- IOI Task: Indirect Object Identification for probing model internals

## License

[Add your license here]

## Contact

For questions or issues, please contact [your contact information].