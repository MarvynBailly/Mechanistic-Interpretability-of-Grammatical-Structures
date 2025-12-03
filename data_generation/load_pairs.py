"""
Utility script to load and inspect color-object association pairs.

This module provides functions to:
- Load datasets from JSON files
- Inspect dataset statistics
- Validate dataset quality
- Sample random examples
"""

import json
from pathlib import Path
from typing import List, Dict, Any
from collections import Counter


def load_pairs(file_path: str) -> List[Dict[str, Any]]:
    """
    Load color-object pairs from a JSON file.
    
    Args:
        file_path: Path to JSON file containing pairs
    
    Returns:
        List of pair dictionaries
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    with open(file_path, 'r') as f:
        pairs = json.load(f)
    
    print(f"Loaded {len(pairs)} pairs from {file_path}")
    return pairs


def print_pair(pair: Dict[str, Any], index: int = None):
    """Print a single pair in a readable format."""
    header = f"--- Pair {index} ---" if index is not None else "--- Pair ---"
    print(f"\n{header}")
    print(f"CLEAN:   {pair['clean']}")
    print(f"         → Expected: '{pair['correct_object']}' (not '{pair['incorrect_object']}')")
    print(f"\nCORRUPT: {pair['corrupt']}")
    print(f"         → Model should fail ('{pair['preferred_color']}' not in context)")
    print(f"\nMetadata:")
    print(f"  Preferred color: {pair['preferred_color']}")
    print(f"  Color-object pairs: {pair['color1']}↔{pair['object1']}, {pair['color2']}↔{pair['object2']}")


def show_random_examples(file_path: str, n: int = 3):
    """Show n random examples from the dataset."""
    import random
    
    pairs = load_pairs(file_path)
    samples = random.sample(pairs, min(n, len(pairs)))
    
    print(f"\n{'='*80}")
    print(f"RANDOM EXAMPLES FROM {Path(file_path).name}")
    print(f"{'='*80}")
    
    for i, pair in enumerate(samples, 1):
        print_pair(pair, i)
    
    print(f"\n{'='*80}")


def compute_statistics(file_path: str):
    """Compute and display dataset statistics."""
    pairs = load_pairs(file_path)
    
    # Collect statistics
    colors = []
    objects = []
    preferred_colors = []
    template_prefixes = []
    
    for pair in pairs:
        colors.extend([pair['color1'], pair['color2']])
        objects.extend([pair['object1'], pair['object2']])
        preferred_colors.append(pair['preferred_color'])
        
        # Extract template type (first 30 chars)
        template_prefix = pair['clean'][:30]
        template_prefixes.append(template_prefix)
    
    # Count occurrences
    color_counts = Counter(colors)
    object_counts = Counter(objects)
    preferred_counts = Counter(preferred_colors)
    template_counts = Counter(template_prefixes)
    
    # Display statistics
    print(f"\n{'='*80}")
    print(f"DATASET STATISTICS: {Path(file_path).name}")
    print(f"{'='*80}")
    
    print(f"\nTotal pairs: {len(pairs)}")
    print(f"Unique colors used: {len(color_counts)}")
    print(f"Unique objects used: {len(object_counts)}")
    print(f"Template variations: {len(template_counts)}")
    
    print(f"\n--- Color Distribution (Top 10) ---")
    for color, count in color_counts.most_common(10):
        print(f"  {color:12} : {count:4} ({count/sum(color_counts.values())*100:.1f}%)")
    
    print(f"\n--- Object Distribution (Top 10) ---")
    for obj, count in object_counts.most_common(10):
        print(f"  {obj:12} : {count:4} ({count/sum(object_counts.values())*100:.1f}%)")
    
    print(f"\n--- Preferred Color Distribution (Top 10) ---")
    for color, count in preferred_counts.most_common(10):
        print(f"  {color:12} : {count:4} ({count/len(pairs)*100:.1f}%)")
    
    print(f"\n--- Template Usage ---")
    for i, (template, count) in enumerate(template_counts.most_common(), 1):
        print(f"  {i}. {template}... : {count} pairs")
    
    print(f"\n{'='*80}")


def validate_dataset(file_path: str) -> bool:
    """
    Validate dataset quality and consistency.
    
    Checks:
    - All required fields present
    - Preferred color in clean sentence
    - Preferred color NOT in corrupt sentence
    - Colors and objects are unique within each pair
    - Correct/incorrect objects match color-object pairs
    
    Returns:
        True if all checks pass, False otherwise
    """
    pairs = load_pairs(file_path)
    
    print(f"\n{'='*80}")
    print(f"VALIDATING DATASET: {Path(file_path).name}")
    print(f"{'='*80}")
    
    required_fields = [
        'clean', 'corrupt', 'correct_object', 'incorrect_object',
        'preferred_color', 'color1', 'color2', 'object1', 'object2'
    ]
    
    all_valid = True
    issues = []
    
    for i, pair in enumerate(pairs):
        # Check required fields
        missing_fields = [field for field in required_fields if field not in pair]
        if missing_fields:
            issues.append(f"Pair {i}: Missing fields {missing_fields}")
            all_valid = False
            continue
        
        # Check preferred color in clean
        if pair['preferred_color'] not in pair['clean']:
            issues.append(f"Pair {i}: Preferred color '{pair['preferred_color']}' not in clean sentence")
            all_valid = False
        
        # Check preferred color NOT in corrupt context (before the preference statement)
        # The corrupt should have the preference statement but different colors in the context
        # Split on all possible preference phrases
        corrupt_context = pair['corrupt']
        for phrase in ['I prefer', 'I like', 'I want', 'I choose', 'I pick', 'I\'ll take', 'I\'ll choose', 'I\'ll pick']:
            if phrase in corrupt_context:
                corrupt_context = corrupt_context.split(phrase)[0]
                break
        
        if pair['preferred_color'] in corrupt_context:
            issues.append(f"Pair {i}: Preferred color '{pair['preferred_color']}' should not be in corrupt context")
            all_valid = False
        
        # Check colors are unique within pair
        if pair['color1'] == pair['color2']:
            issues.append(f"Pair {i}: color1 and color2 are the same ('{pair['color1']}')")
            all_valid = False
        
        # Check objects are unique within pair
        if pair['object1'] == pair['object2']:
            issues.append(f"Pair {i}: object1 and object2 are the same ('{pair['object1']}')")
            all_valid = False
        
        # Check correct/incorrect objects match the pair
        objects_in_pair = {pair['object1'], pair['object2']}
        if pair['correct_object'] not in objects_in_pair:
            issues.append(f"Pair {i}: correct_object '{pair['correct_object']}' not in pair objects")
            all_valid = False
        if pair['incorrect_object'] not in objects_in_pair:
            issues.append(f"Pair {i}: incorrect_object '{pair['incorrect_object']}' not in pair objects")
            all_valid = False
        
        # Check correct and incorrect are different
        if pair['correct_object'] == pair['incorrect_object']:
            issues.append(f"Pair {i}: correct_object and incorrect_object are the same")
            all_valid = False
    
    # Print results
    if all_valid:
        print(f"\n✅ All {len(pairs)} pairs are valid!")
    else:
        print(f"\n❌ Found {len(issues)} issues:")
        for issue in issues[:20]:  # Show first 20 issues
            print(f"  - {issue}")
        if len(issues) > 20:
            print(f"  ... and {len(issues) - 20} more issues")
    
    print(f"\n{'='*80}")
    return all_valid


def compare_datasets(file1: str, file2: str):
    """Compare two datasets and show differences in statistics."""
    print(f"\n{'='*80}")
    print(f"COMPARING DATASETS")
    print(f"{'='*80}")
    
    pairs1 = load_pairs(file1)
    pairs2 = load_pairs(file2)
    
    print(f"\nDataset 1: {Path(file1).name}")
    print(f"  Size: {len(pairs1)} pairs")
    
    print(f"\nDataset 2: {Path(file2).name}")
    print(f"  Size: {len(pairs2)} pairs")
    
    print(f"\nSize difference: {abs(len(pairs1) - len(pairs2))} pairs")
    
    # Could add more detailed comparisons here (color distributions, etc.)
    print(f"\n{'='*80}")


def export_to_text(file_path: str, output_file: str = None):
    """Export dataset to human-readable text format."""
    pairs = load_pairs(file_path)
    
    if output_file is None:
        output_file = Path(file_path).with_suffix('.txt')
    
    with open(output_file, 'w') as f:
        f.write("COLOR-OBJECT ASSOCIATION DATASET\n")
        f.write("=" * 80 + "\n\n")
        
        for i, pair in enumerate(pairs, 1):
            f.write(f"--- Pair {i} ---\n")
            f.write(f"CLEAN:   {pair['clean']}\n")
            f.write(f"         → Expected: '{pair['correct_object']}'\n\n")
            f.write(f"CORRUPT: {pair['corrupt']}\n")
            f.write(f"         → Model should fail\n\n")
            f.write(f"Metadata: {pair['preferred_color']} → {pair['correct_object']}\n")
            f.write("\n")
    
    print(f"Exported to: {output_file}")


if __name__ == "__main__":
    import sys
    
    # Default datasets to inspect
    output_dir = Path(__file__).parent / "output"
    
    datasets = [
        output_dir / "color_pairs_small.json",
        output_dir / "color_pairs_medium.json",
        output_dir / "color_pairs_large.json",
    ]
    
    # Check if datasets exist
    existing_datasets = [d for d in datasets if d.exists()]
    
    if not existing_datasets:
        print("No datasets found. Generate them first by running:")
        print("  python generate_color_pairs.py")
        sys.exit(1)
    
    # Inspect the first existing dataset
    dataset = existing_datasets[0]
    
    print(f"\n{'='*80}")
    print(f"COLOR-OBJECT DATASET INSPECTOR")
    print(f"{'='*80}")
    
    # Show random examples
    show_random_examples(dataset, n=3)
    
    # Compute statistics
    compute_statistics(dataset)
    
    # Validate dataset
    validate_dataset(dataset)
    
    # Show available datasets
    print(f"\n{'='*80}")
    print(f"AVAILABLE DATASETS")
    print(f"{'='*80}")
    for d in existing_datasets:
        pairs = load_pairs(d)
        print(f"  - {d.name}: {len(pairs)} pairs")
    print(f"\n{'='*80}")
