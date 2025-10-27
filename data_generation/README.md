# Data Generation Pipeline

This directory contains the data generation pipeline for the project as described in the project proposal. The pipeline generates English-Chinese parallel datasets for studying how multilingual language models process grammatical structures.

## Overview

This pipeline generates:
- **900 sentence pairs** (60 words × 15 templates)
- **Clean variants** for baseline analysis
- **Fuzzy variants** for robustness testing (missing function words, shuffled order, misspellings)
- **Path patching pairs** for mechanistic interpretability experiments

## Project Structure

```
data_generation/
├── README.md                    # This file
├── generate_dataset.py          # Main data generation pipeline
└── input/                       # Required to run
|    ├── templates.json          # 15 IOI sentence templates
|    ├── words.json              # 60 words with Chinese translations
└── output/                      # Generated datasets (created on first run)
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

### 4. Path Patching
Generates clean-corrupted pairs for causal intervention experiments:
- Each pair contains one clean and one corrupted variant

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

#### 3. Translate and Verify

**Important**: As per the project proposal:
1. Use ChatGPT to translate English words/templates to Chinese
2. **Have translations manually verified by a native Chinese speaker**
3. Update the `chinese_translations` dictionary

#### 4. Generate Dataset

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

## Data Quality Checklist

Before using the generated dataset:

- [ ] Templates are grammatically correct in both languages
- [ ] Chinese translations verified by native speaker
- [ ] Word-template combinations make semantic sense
- [ ] Fuzzy variants are appropriately degraded
- [ ] Path patching pairs contain meaningful contrasts
- [ ] Dataset statistics match expectations (900 clean pairs, etc.)
