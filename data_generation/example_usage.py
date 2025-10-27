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
    Example: Basic dataset generation with default files
    """
    # Create generator
    generator = DatasetGenerator(output_dir="output")

    # Generate complete dataset
    generator.generate_full_dataset(
        templates_file="input/templates.json", words_file="input/words.json"
    )


def main():
    print("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║                      Data Generation Pipeline                        ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """)

    example_basic_generation()

    print("\nGenerated files in 'example_output/' directory:")
    print("  • clean_pairs.json - 900 grammatical sentence pairs")
    print("  • fuzzy_pairs.json - 2,700 degraded variants")
    print("  • path_patching_pairs.json - 900 intervention pairs")
    print("  • dataset_stats.json - Summary statistics")
    print("  • clean_pairs.csv - CSV export for spreadsheets")
    print("  • dataset_summary.txt - Human-readable summary")
    
    print("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║                      Data Generation Pipeline                        ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    main()
