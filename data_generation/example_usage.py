"""
Example Usage of Data Generation Pipeline

This script demonstrates how to use the data generation pipeline
for the Mechanistic Interpretability of Grammatical Structures project.

Run this file to see the complete workflow from data generation to
basic analysis.
"""

import json
import os
from pathlib import Path
from generate_dataset import DatasetGenerator


def example_1_basic_generation():
    """
    Example 1: Basic dataset generation with default files

    This is the simplest use case - just generate the dataset using
    the provided templates and words.
    """
    print("=" * 70)
    print("EXAMPLE 1: Basic Dataset Generation")
    print("=" * 70)
    print()

    # Create generator
    generator = DatasetGenerator(output_dir="example_output")

    # Generate complete dataset
    generator.generate_full_dataset(
        templates_file="templates.json", words_file="words.json"
    )

    print("\n✓ Dataset generated in 'example_output/' directory")
    print("  - clean_pairs.json: 900 grammatical sentence pairs")
    print("  - fuzzy_pairs.json: 2,700 degraded variants")
    print("  - path_patching_pairs.json: 900 intervention pairs")
    print("  - dataset_stats.json: Summary statistics")


def example_2_load_and_inspect():
    """
    Example 2: Load and inspect generated data

    Shows how to load the generated datasets and examine their structure.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Load and Inspect Generated Data")
    print("=" * 70)
    print()

    output_dir = Path("example_output")

    if not output_dir.exists():
        print("⚠ Run Example 1 first to generate data!")
        return

    # Load clean pairs
    with open(output_dir / "clean_pairs.json", "r", encoding="utf-8") as f:
        clean_pairs = json.load(f)

    print(f"Loaded {len(clean_pairs)} clean pairs")
    print("\nFirst clean pair:")
    first_pair = clean_pairs[0]
    print(f"  Template ID: {first_pair['template_id']}")
    print(f"  Word: {first_pair['word']}")
    print(f"  Grammatical Feature: {first_pair['grammatical_feature']}")
    print(f"  English: {first_pair['english']}")
    print(f"  Chinese: {first_pair['chinese']}")

    # Load fuzzy pairs
    with open(output_dir / "fuzzy_pairs.json", "r", encoding="utf-8") as f:
        fuzzy_pairs = json.load(f)

    print(f"\nLoaded {len(fuzzy_pairs)} fuzzy pairs")
    print("\nExample fuzzy variants for same word:")

    # Find all fuzzy variants of the first word
    word = clean_pairs[0]["word"]
    template_id = clean_pairs[0]["template_id"]

    fuzzy_examples = [
        p for p in fuzzy_pairs if p["word"] == word and p["template_id"] == template_id
    ]

    print(f"  Original: {clean_pairs[0]['english']}")
    for fuzzy in fuzzy_examples:
        print(f"  {fuzzy['fuzzy_type']}: {fuzzy['english']}")

    # Load path patching pairs
    with open(output_dir / "path_patching_pairs.json", "r", encoding="utf-8") as f:
        patching_pairs = json.load(f)

    print(f"\nLoaded {len(patching_pairs)} path patching pairs")
    print("\nFirst path patching pair:")
    first_patching = patching_pairs[0]
    print(f"  Clean English: {first_patching['clean']['english']}")
    print(f"  Corrupted English: {first_patching['corrupted']['english']}")
    print(f"  Clean Chinese: {first_patching['clean']['chinese']}")
    print(f"  Corrupted Chinese: {first_patching['corrupted']['chinese']}")

    # Load statistics
    with open(output_dir / "dataset_stats.json", "r", encoding="utf-8") as f:
        stats = json.load(f)

    print("\nDataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


def example_3_filter_by_feature():
    """
    Example 3: Filter pairs by grammatical feature

    Shows how to select subsets of the data based on grammatical features.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Filter by Grammatical Feature")
    print("=" * 70)
    print()

    output_dir = Path("example_output")

    if not output_dir.exists():
        print("⚠ Run Example 1 first to generate data!")
        return

    # Load clean pairs
    with open(output_dir / "clean_pairs.json", "r", encoding="utf-8") as f:
        clean_pairs = json.load(f)

    # Group by grammatical feature
    by_feature = {}
    for pair in clean_pairs:
        feature = pair["grammatical_feature"]
        if feature not in by_feature:
            by_feature[feature] = []
        by_feature[feature].append(pair)

    print("Pairs by grammatical feature:")
    for feature, pairs in by_feature.items():
        print(f"  {feature}: {len(pairs)} pairs")
        if pairs:
            print(f"    Example word: {pairs[0]['word']}")

    # Example: Get all plural noun sentences
    plural_pairs = by_feature.get("plural_noun", [])
    if plural_pairs:
        print(f"\nExample sentences with plural nouns:")
        for pair in plural_pairs[:3]:  # Show first 3
            print(f"  {pair['english']}")


def example_4_prepare_for_path_patching():
    """
    Example 4: Prepare data for path patching experiments

    Shows how to structure the data for mechanistic interpretability experiments.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Prepare for Path Patching Experiments")
    print("=" * 70)
    print()

    output_dir = Path("example_output")

    if not output_dir.exists():
        print("⚠ Run Example 1 first to generate data!")
        return

    # Load path patching pairs
    with open(output_dir / "path_patching_pairs.json", "r", encoding="utf-8") as f:
        patching_pairs = json.load(f)

    print("Path Patching Experiment Setup:")
    print(f"  Total pairs: {len(patching_pairs)}")
    print(f"  Runs per pair: 2 (clean + corrupted)")

    # Calculate forward passes for different model sizes
    layers = 12
    heads = 12
    total_pairs = len(patching_pairs)

    forward_passes = total_pairs * 2 * layers * heads

    print(f"\nComputational Requirements (12L × 12H model):")
    print(f"  Forward passes: {forward_passes:,}")
    print(f"  Per layer: {total_pairs * 2 * heads:,}")
    print(f"  Per head per layer: {total_pairs * 2:,}")

    print("\nExample intervention setup:")
    pair = patching_pairs[0]
    print(f"  Pair ID: {pair['pair_id']}")
    print(f"  Word: {pair['metadata']['word']}")
    print(f"  Feature: {pair['metadata']['grammatical_feature']}")
    print(f"\n  Clean input (English): {pair['clean']['english']}")
    print(f"  Clean input (Chinese): {pair['clean']['chinese']}")
    print(f"\n  Corrupted input (English): {pair['corrupted']['english']}")
    print(f"  Corrupted input (Chinese): {pair['corrupted']['chinese']}")

    print("\nPseudo-code for path patching:")
    print("""
    for pair in patching_pairs:
        # Get model activations
        clean_acts = model(pair['clean'])
        corrupt_acts = model(pair['corrupted'])

        # For each layer and head
        for layer in range(12):
            for head in range(12):
                # Patch clean activations with corrupted ones
                patched_acts = patch_activation(
                    clean_acts,
                    corrupt_acts,
                    layer,
                    head
                )

                # Measure effect on output
                effect = measure_effect(patched_acts)
                results[layer][head] = effect
    """)


def example_5_cross_linguistic_comparison():
    """
    Example 5: Cross-linguistic comparison setup

    Shows how to organize data for English-Chinese comparative analysis.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Cross-Linguistic Comparison Setup")
    print("=" * 70)
    print()

    output_dir = Path("example_output")

    if not output_dir.exists():
        print("⚠ Run Example 1 first to generate data!")
        return

    # Load clean pairs
    with open(output_dir / "clean_pairs.json", "r", encoding="utf-8") as f:
        clean_pairs = json.load(f)

    print("Cross-linguistic analysis setup:")
    print(f"  Language pair: English ↔ Chinese")
    print(f"  Parallel sentences: {len(clean_pairs)}")

    # Example: Create language-specific datasets
    english_sentences = [pair["english"] for pair in clean_pairs]
    chinese_sentences = [pair["chinese"] for pair in clean_pairs]

    print(f"\n  English sentences: {len(english_sentences)}")
    print(f"  Chinese sentences: {len(chinese_sentences)}")

    print("\nExample parallel pair:")
    pair = clean_pairs[0]
    print(f"  EN: {pair['english']}")
    print(f"  ZH: {pair['chinese']}")
    print(f"  Word: {pair['word']} (Feature: {pair['grammatical_feature']})")

    print("\nAnalysis questions to explore:")
    print("  1. Do English and Chinese activate the same attention heads?")
    print("  2. How do information flow patterns differ between languages?")
    print("  3. Are there universal grammatical circuits?")
    print("  4. How does word order difference affect processing?")
    print("  5. Which components are language-specific vs. shared?")


def example_6_export_for_analysis():
    """
    Example 6: Export data in different formats for analysis

    Shows how to convert the data to formats suitable for different analysis tools.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Export for Different Analysis Tools")
    print("=" * 70)
    print()

    output_dir = Path("example_output")

    if not output_dir.exists():
        print("⚠ Run Example 1 first to generate data!")
        return

    # Load clean pairs
    with open(output_dir / "clean_pairs.json", "r", encoding="utf-8") as f:
        clean_pairs = json.load(f)

    # Export as CSV (for easy viewing in Excel, pandas, etc.)
    try:
        import csv

        csv_file = output_dir / "clean_pairs.csv"
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            if clean_pairs:
                writer = csv.DictWriter(f, fieldnames=clean_pairs[0].keys())
                writer.writeheader()
                writer.writerows(clean_pairs)

        print(f"✓ Exported to CSV: {csv_file}")
    except Exception as e:
        print(f"⚠ CSV export failed: {e}")

    # Export as simple text file (one sentence per line)
    txt_file = output_dir / "english_sentences.txt"
    with open(txt_file, "w", encoding="utf-8") as f:
        for pair in clean_pairs:
            f.write(pair["english"] + "\n")
    print(f"✓ Exported English sentences: {txt_file}")

    txt_file = output_dir / "chinese_sentences.txt"
    with open(txt_file, "w", encoding="utf-8") as f:
        for pair in clean_pairs:
            f.write(pair["chinese"] + "\n")
    print(f"✓ Exported Chinese sentences: {txt_file}")

    # Export metadata summary
    summary_file = output_dir / "dataset_summary.txt"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("DATASET SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Total pairs: {len(clean_pairs)}\n\n")

        # Count by feature
        features = {}
        for pair in clean_pairs:
            feature = pair["grammatical_feature"]
            features[feature] = features.get(feature, 0) + 1

        f.write("Distribution by grammatical feature:\n")
        for feature, count in sorted(features.items()):
            f.write(f"  {feature}: {count}\n")

        # Count by template
        templates = {}
        for pair in clean_pairs:
            template_id = pair["template_id"]
            templates[template_id] = templates.get(template_id, 0) + 1

        f.write(f"\nTemplates used: {len(templates)}\n")
        f.write(f"Sentences per template: {len(clean_pairs) // len(templates)}\n")

    print(f"✓ Exported summary: {summary_file}")


def main():
    """
    Run all examples
    """
    print("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║     Data Generation Pipeline - Example Usage                        ║
    ║     Mechanistic Interpretability of Grammatical Structures          ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """)

    # Run all examples in sequence
    example_1_basic_generation()
    example_2_load_and_inspect()
    example_3_filter_by_feature()
    example_4_prepare_for_path_patching()
    example_5_cross_linguistic_comparison()
    example_6_export_for_analysis()

    print("\n" + "=" * 70)
    print("ALL EXAMPLES COMPLETED!")
    print("=" * 70)
    print("\nGenerated files in 'example_output/' directory:")
    print("  • clean_pairs.json - 900 grammatical sentence pairs")
    print("  • fuzzy_pairs.json - 2,700 degraded variants")
    print("  • path_patching_pairs.json - 900 intervention pairs")
    print("  • dataset_stats.json - Summary statistics")
    print("  • clean_pairs.csv - CSV export for spreadsheets")
    print("  • english_sentences.txt - Plain text English sentences")
    print("  • chinese_sentences.txt - Plain text Chinese sentences")
    print("  • dataset_summary.txt - Human-readable summary")
    print("\nNext steps:")
    print("  1. Load the data in your ML framework")
    print("  2. Run path patching experiments")
    print("  3. Analyze cross-linguistic patterns")
    print("  4. Compare English vs Chinese processing")
    print("\nHappy researching! 🚀")


if __name__ == "__main__":
    main()
