# Data Generation Pipeline Overview

## Executive Summary

This document provides a comprehensive overview of the data generation pipeline for the **Mechanistic Interpretability of Grammatical Structures** project. The pipeline generates 900 English-Chinese parallel sentence pairs for studying how multilingual language models process grammatical structures through mechanistic interpretability methods.

## Project Context

Based on the project proposal (`../Project_Proposal/project_proposal.pdf`), this pipeline supports research investigating whether multilingual LLMs employ shared or distinct internal circuits when processing languages with different grammatical structures (English vs. Chinese).

### Research Goals
- Examine cross-linguistic structural representations using path patching
- Analyze Information Flow Routes (IFR) through transformer architectures
- Study how grammatical cues affect model behavior in clean vs. degraded inputs
- Compare attention head and feed-forward module activation patterns across languages

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   INPUT DATA SOURCES                         │
├─────────────────────────────────────────────────────────────┤
│  • templates.json (15 IOI sentence templates)               │
│  • words.json (60 words with grammatical features)          │
│  • chinese_translations (verified by native speaker)        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              CORE GENERATION PIPELINE                        │
│                 (generate_dataset.py)                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────────────────────────────┐          │
│  │  1. Load Templates & Words                   │          │
│  │     - Validate JSON structure                │          │
│  │     - Load Chinese translations              │          │
│  └──────────────────────────────────────────────┘          │
│                     ↓                                        │
│  ┌──────────────────────────────────────────────┐          │
│  │  2. Generate Clean Pairs                     │          │
│  │     - Fill templates with words              │          │
│  │     - Create parallel EN-ZH pairs            │          │
│  │     - Output: 900 pairs (60 × 15)           │          │
│  └──────────────────────────────────────────────┘          │
│                     ↓                                        │
│  ┌──────────────────────────────────────────────┐          │
│  │  3. Generate Fuzzy Variants                  │          │
│  │     a) Missing Function Words                │          │
│  │     b) Shuffled Word Order                   │          │
│  │     c) Random Misspellings                   │          │
│  │     - Output: 2,700 pairs (900 × 3)         │          │
│  └──────────────────────────────────────────────┘          │
│                     ↓                                        │
│  ┌──────────────────────────────────────────────┐          │
│  │  4. Generate Path Patching Pairs             │          │
│  │     - Clean-Corrupted pair generation        │          │
│  │     - Metadata for interventions             │          │
│  │     - Output: 900 pairs                      │          │
│  └──────────────────────────────────────────────┘          │
│                     ↓                                        │
│  ┌──────────────────────────────────────────────┐          │
│  │  5. Save & Validate                          │          │
│  │     - Export to JSON files                   │          │
│  │     - Generate statistics                    │          │
│  │     - Validate output format                 │          │
│  └──────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                     OUTPUT FILES                             │
├─────────────────────────────────────────────────────────────┤
│  • clean_pairs.json (900 grammatical pairs)                 │
│  • fuzzy_pairs.json (2,700 degraded variants)               │
│  • path_patching_pairs.json (900 intervention pairs)        │
│  • dataset_stats.json (summary statistics)                  │
└─────────────────────────────────────────────────────────────┘
```

## File Structure

```
data_generation/
│
├── Core Pipeline Scripts
│   ├── generate_dataset.py          # Main data generation pipeline
│   ├── fetch_words_from_hf.py       # Word list generation utility
│   └── translate_helper.py          # Translation workflow automation
│
├── Input Data
│   ├── templates.json               # 15 IOI sentence templates
│   └── words.json                   # 60 words with translations
│
├── Testing & Validation
│   └── test_pipeline.py             # Automated test suite
│
├── Documentation
│   ├── README.md                    # Comprehensive documentation
│   ├── QUICKSTART.md                # 5-minute getting started guide
│   ├── PIPELINE_OVERVIEW.md         # This file
│   └── requirements.txt             # Python dependencies
│
└── Output (generated)
    ├── clean_pairs.json
    ├── fuzzy_pairs.json
    ├── path_patching_pairs.json
    └── dataset_stats.json
```

## Data Specifications

### Input Specifications

#### Templates (templates.json)
- **Count**: 15 templates
- **Structure**: IOI (Indirect Object Identification) patterns
- **Format**: Parallel English-Chinese with `{word}` placeholder
- **Example**:
  ```json
  {
    "id": 0,
    "english": "When John and Mary went to the store, John gave the {word} to Mary.",
    "chinese": "当约翰和玛丽去商店时，约翰把{word}给了玛丽。",
    "description": "IOI template with two names"
  }
  ```

#### Words (words.json)
- **Count**: 60 words
- **Categories**:
  - Singular nouns (15)
  - Plural nouns (15)
  - Past tense verbs with -ed (10)
  - Abstract nouns (10)
  - Concrete nouns (10)
- **Metadata**: Grammatical feature, POS tag, ID
- **Translations**: Verified Chinese equivalents

### Output Specifications

#### Clean Pairs
- **Count**: 900 pairs (60 words × 15 templates)
- **Purpose**: Baseline grammatical sentences
- **Fields**: template_id, word_id, english, chinese, word, grammatical_feature, is_fuzzy
- **Quality**: Grammatically correct in both languages

#### Fuzzy Pairs
- **Count**: 2,700 pairs (900 × 3 variants)
- **Types**:
  1. **Missing Function Words**: Removes articles, prepositions, conjunctions
  2. **Shuffled Order**: Randomizes word positions
  3. **Misspelled**: Introduces character-swap typos
- **Purpose**: Test model robustness to grammatical degradation

#### Path Patching Pairs
- **Count**: 900 pairs
- **Structure**: Each pair contains:
  - Clean sentence (baseline)
  - Corrupted sentence (intervention target)
  - Metadata (IDs, features)
- **Purpose**: Causal intervention experiments
- **Usage**: Identify which model components process grammatical information

## Computational Requirements

From the project proposal:

| Parameter | Value | Notes |
|-----------|-------|-------|
| Dataset Pairs | 900 | 60 words × 15 templates |
| Runs per Pair | 2 | Clean + Corrupted |
| Model Layers | 12 | Standard transformer |
| Attention Heads | 12 | Per layer |
| **Total Forward Passes** | **259,200** | 900 × 2 × 12 × 12 |
| Recommended GPU | RTX 4070 | Or equivalent |
| Parallelization | Embarrassingly parallel | Across 900 pairs |

### Memory Estimate
- Models: Small multilingual LLMs (e.g., mBERT, XLM-R base)
- Dataset size: ~5-10 MB (JSON)
- GPU RAM: Should fit comfortably on 12GB VRAM
- Computation time: Depends on model size and GPU

## Key Design Decisions

### 1. Template-Based Generation
**Rationale**: Ensures grammatical consistency and allows systematic variation across words while maintaining sentence structure.

### 2. 60 Words × 15 Templates = 900 Pairs
**Rationale**: 
- Sufficient statistical power for analysis
- Manageable computational cost (~260K forward passes)
- Covers diverse grammatical features
- Allows for robust cross-linguistic comparison

### 3. Three Fuzzy Variant Types
**Rationale**:
- **Missing function words**: Tests syntactic structure understanding
- **Shuffled order**: Tests word order dependency
- **Misspellings**: Tests token-level robustness

### 4. Parallel English-Chinese Design
**Rationale**:
- English: SVO, rich morphology, function words critical
- Chinese: SVO, minimal morphology, topic-prominent
- Maximizes structural contrast for cross-linguistic study

### 5. Manual Translation Verification
**Rationale**: Ensures translation quality and natural phrasing, critical for valid cross-linguistic comparisons.

## Workflow Integration

### Step 1: Data Generation
```bash
python generate_dataset.py
```
**Output**: 900 clean, 2,700 fuzzy, 900 patching pairs

### Step 2: Model Loading
```python
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("bert-base-multilingual-cased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
```

### Step 3: Path Patching Experiments
```python
import json

# Load path patching pairs
with open('output/path_patching_pairs.json') as f:
    pairs = json.load(f)

# For each pair, run clean and corrupted through model
for pair in pairs:
    clean_en = pair['clean']['english']
    corrupted_en = pair['corrupted']['english']
    
    # Run path patching intervention
    # (Implementation depends on your MI framework)
```

### Step 4: Analysis
- Compare activation patterns English vs. Chinese
- Identify language-specific vs. shared circuits
- Analyze degradation effects on information flow

## Quality Assurance

### Automated Tests
Run `python test_pipeline.py` to verify:
- ✅ File structure and validity
- ✅ Correct pair counts (900, 2,700, 900)
- ✅ Required fields present
- ✅ Data format consistency
- ✅ UTF-8 encoding for Chinese characters

### Manual Verification Checklist
- [ ] Templates grammatically correct in both languages
- [ ] Chinese translations verified by native speaker
- [ ] Word-template combinations semantically valid
- [ ] Fuzzy variants appropriately degraded
- [ ] No placeholder text (`[TRANSLATE:]`) in final data
- [ ] All JSON files valid and loadable

## Extension Points

### Adding New Languages
1. Add language column to templates.json
2. Update translation workflow
3. Verify with native speaker

### Adding New Grammatical Features
1. Extend word categories in words.json
2. Add filtering logic in fetch_words_from_hf.py
3. Update documentation

### Adding New Fuzzy Types
1. Implement new degradation method in generate_dataset.py
2. Update fuzzy_types list
3. Document expected behavior

### Scaling Dataset
- Increase templates (currently 15 → N)
- Increase words (currently 60 → M)
- Total pairs: N × M
- Forward passes: N × M × 2 × layers × heads

## References

1. **Project Proposal**: `../Project_Proposal/project_proposal.pdf`
2. **ZYZEP25**: Cross-linguistic structural representations with path patching
3. **IOI Task**: Indirect Object Identification for mechanistic interpretability
4. **Path Patching**: Causal intervention method for identifying critical model components

## Version History

- **v1.0** (Current): Initial pipeline implementation
  - 15 templates, 60 words
  - 3 fuzzy variant types
  - Path patching pair generation
  - Automated testing suite

## Contact & Support

For questions, issues, or contributions:
- Review documentation in README.md
- Run test suite: `python test_pipeline.py`
- Check QUICKSTART.md for common workflows

---

**Last Updated**: 2024
**Pipeline Status**: Production Ready ✅