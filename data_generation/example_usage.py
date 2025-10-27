"""
Example Usage of Data Generation Pipeline

This script demonstrates how to use the data generation pipeline.
"""

import json
import os
from pathlib import Path
from generate_dataset import DatasetGenerator


def example_basic_generation():
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
        templates_file="input/templates.json", words_file="input/words.json"
    )

    print("\n✓ Dataset generated in 'example_output/' directory")
    print("  - clean_pairs.json: 900 grammatical sentence pairs")
    print("  - fuzzy_pairs.json: 2,700 degraded variants")
    print("  - path_patching_pairs.json: 900 intervention pairs")
    print("  - dataset_stats.json: Summary statistics")


def main():
    """
    Run all examples
    """
    print("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║                      Data Generation Pipeline                        ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """)

    # Run all examples in sequence
    example_basic_generation()

    print("\n" + "=" * 70)
    print("ALL EXAMPLES COMPLETED!")
    print("=" * 70)
    print("\nGenerated files in 'example_output/' directory:")
    print("  • clean_pairs.json - 900 grammatical sentence pairs")
    print("  • fuzzy_pairs.json - 2,700 degraded variants")
    print("  • path_patching_pairs.json - 900 intervention pairs")
    print("  • dataset_stats.json - Summary statistics")
    print("  • clean_pairs.csv - CSV export for spreadsheets")
    print("  • dataset_summary.txt - Human-readable summary")



if __name__ == "__main__":
    main()
