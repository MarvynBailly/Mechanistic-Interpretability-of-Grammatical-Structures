"""
Example usage of the color-object association dataset generator.

This script demonstrates:
1. Generating custom datasets with specific parameters
2. Loading and inspecting generated datasets
3. Creating datasets for different experimental needs
"""

from generate_color_pairs import (
    generate_dataset,
    generate_color_object_pair,
    print_example_pairs,
    TEMPLATES,
    COLORS,
    OBJECTS
)
from pathlib import Path
import json


def example_1_generate_custom_dataset():
    """Example 1: Generate a custom dataset with specific size and seed."""
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Generate Custom Dataset")
    print("=" * 80)
    
    # Generate 50 examples with custom seed
    pairs = generate_dataset(
        n_examples=50,
        seed=123,
        output_file="output/custom_dataset.json"
    )
    
    print(f"\nGenerated {len(pairs)} pairs")
    print(f"First pair:")
    print(f"  Clean: {pairs[0].clean_text}")
    print(f"  Expected: {pairs[0].correct_object}")


def example_2_inspect_dataset_statistics():
    """Example 2: Load a dataset and compute statistics."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Dataset Statistics")
    print("=" * 80)
    
    # Generate a small dataset for analysis
    pairs = generate_dataset(n_examples=100, seed=42)
    
    # Count color and object usage
    color_counts = {}
    object_counts = {}
    template_counts = {}
    
    for pair in pairs:
        # Count colors
        for color in [pair.color1, pair.color2]:
            color_counts[color] = color_counts.get(color, 0) + 1
        
        # Count objects
        for obj in [pair.object1, pair.object2]:
            object_counts[obj] = object_counts.get(obj, 0) + 1
        
        # Extract template type (approximate by first 20 chars)
        template_key = pair.clean_text[:30]
        template_counts[template_key] = template_counts.get(template_key, 0) + 1
    
    print(f"\nDataset size: {len(pairs)} pairs")
    print(f"\nColor distribution (top 5):")
    for color, count in sorted(color_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {color}: {count}")
    
    print(f"\nObject distribution (top 5):")
    for obj, count in sorted(object_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {obj}: {count}")
    
    print(f"\nNumber of template variations: {len(template_counts)}")
    
    # Check balance of preferred_first
    preferred_first_count = sum(1 for i in range(len(pairs)) if i % 2 == 0)
    print(f"\nBalance check:")
    print(f"  Preferred color is first: {preferred_first_count}")
    print(f"  Preferred color is second: {len(pairs) - preferred_first_count}")


def example_3_generate_for_specific_experiment():
    """Example 3: Generate datasets tailored for specific experiments."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Generate for Specific Experiments")
    print("=" * 80)
    
    output_dir = Path("output/experiments")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Quick test set for debugging
    print("\n1. Quick test set (10 examples)")
    generate_dataset(
        n_examples=10,
        output_file=output_dir / "test_quick.json",
        seed=1
    )
    
    # Validation set for hyperparameter tuning
    print("\n2. Validation set (200 examples)")
    generate_dataset(
        n_examples=200,
        output_file=output_dir / "validation.json",
        seed=2
    )
    
    # Full experimental set
    print("\n3. Full experiment set (1000 examples)")
    generate_dataset(
        n_examples=1000,
        output_file=output_dir / "full_experiment.json",
        seed=3
    )
    
    print(f"\nAll datasets saved to: {output_dir}")


def example_4_single_pair_generation():
    """Example 4: Generate and inspect individual pairs programmatically."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Single Pair Generation")
    print("=" * 80)
    
    # Generate a single pair
    template = TEMPLATES[0]
    pair = generate_color_object_pair(template, preferred_first=True)
    
    print(f"\nTemplate: {template}")
    print(f"\nGenerated pair:")
    print(f"  Clean: {pair.clean_text}")
    print(f"  Corrupt: {pair.corrupt_text}")
    print(f"\nExpected behavior:")
    print(f"  - On clean: model should predict '{pair.correct_object}'")
    print(f"  - On corrupt: model should be confused (preferred color not in context)")
    print(f"\nMetadata:")
    for key, value in pair.to_dict().items():
        if key not in ['clean', 'corrupt']:
            print(f"  {key}: {value}")


def example_5_available_items():
    """Example 5: Show available colors, objects, and templates."""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Available Dataset Items")
    print("=" * 80)
    
    print(f"\nAvailable colors ({len(COLORS)}):")
    print(f"  {', '.join(COLORS)}")
    
    print(f"\nAvailable objects ({len(OBJECTS)}):")
    print(f"  {', '.join(OBJECTS)}")
    
    print(f"\nAvailable templates ({len(TEMPLATES)}):")
    for i, template in enumerate(TEMPLATES, 1):
        print(f"  {i}. {template}")
    
    print(f"\nTotal possible unique combinations:")
    print(f"  {len(COLORS)} colors × {len(OBJECTS)} objects = {len(COLORS) * len(OBJECTS)} color-object pairs")
    print(f"  With 2 pairs per sentence and {len(TEMPLATES)} templates")


def example_6_load_and_use_dataset():
    """Example 6: Load a generated dataset and use it."""
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Load and Use Generated Dataset")
    print("=" * 80)
    
    # First generate a small dataset
    output_file = Path("output/example_load.json")
    generate_dataset(n_examples=10, output_file=output_file, seed=99)
    
    # Load the dataset
    with open(output_file, 'r') as f:
        data = json.load(f)
    
    print(f"\nLoaded {len(data)} pairs from {output_file}")
    
    # Show how to access the data
    print(f"\nExample of accessing data:")
    example = data[0]
    print(f"  Clean text: {example['clean']}")
    print(f"  Corrupt text: {example['corrupt']}")
    print(f"  Correct object: {example['correct_object']}")
    print(f"  Incorrect object: {example['incorrect_object']}")
    
    # Compute logit difference (pseudo-code)
    print(f"\nFor path patching, you would:")
    print(f"  1. Run model on clean: logit_diff = logit('{example['correct_object']}') - logit('{example['incorrect_object']}')")
    print(f"  2. Run model on corrupt: baseline_logit_diff (should be ~0)")
    print(f"  3. Patch activations from clean → corrupt and measure recovery")


if __name__ == "__main__":
    # Run all examples
    print("\n" + "=" * 80)
    print("COLOR-OBJECT ASSOCIATION DATASET GENERATOR: EXAMPLES")
    print("=" * 80)
    
    # Show some example pairs first
    print_example_pairs(2)
    
    # Run each example
    example_1_generate_custom_dataset()
    example_2_inspect_dataset_statistics()
    example_3_generate_for_specific_experiment()
    example_4_single_pair_generation()
    example_5_available_items()
    example_6_load_and_use_dataset()
    
    print("\n" + "=" * 80)
    print("All examples completed!")
    print("=" * 80)
