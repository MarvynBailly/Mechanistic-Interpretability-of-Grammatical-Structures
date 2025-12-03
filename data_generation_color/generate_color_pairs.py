"""
Color-Object Association Dataset Generator for Path Patching Experiments

Generates clean/corrupt pairs where:
- Clean: Color-object pairs established, then model retrieves correct object for preferred color
- Corrupt: Different color-object pairs, preferred color not in context (model should be confused)

Example Clean:
"The red ball and blue cube are here. I prefer the red, so I'll take the" → "ball"

Example Corrupt:
"The green sphere and yellow pyramid are here. I prefer the red, so I'll take the" → confused
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass


@dataclass
class ColorObjectPair:
    """A clean/corrupt pair for path patching experiments."""
    clean_text: str
    corrupt_text: str
    correct_object: str      # Expected answer (object paired with preferred color in clean)
    incorrect_object: str    # Distractor (the other object in clean)
    preferred_color: str     # The color mentioned in "I prefer the X"
    color1: str             # First color in clean sentence
    color2: str             # Second color in clean sentence
    object1: str            # First object in clean sentence
    object2: str            # Second object in clean sentence
    
    def to_dict(self):
        return {
            "clean": self.clean_text,
            "corrupt": self.corrupt_text,
            "correct_object": self.correct_object,
            "incorrect_object": self.incorrect_object,
            "preferred_color": self.preferred_color,
            "color1": self.color1,
            "color2": self.color2,
            "object1": self.object1,
            "object2": self.object2,
        }


# Colors that are single tokens in GPT-2
COLORS = [
    "red", "blue", "green", "yellow", "purple", "orange",
    "pink", "brown", "black", "white", "gray", "silver",
]

# Objects that are single tokens in GPT-2
OBJECTS = [
    "ball", "cube", "sphere", "pyramid", "cylinder", "cone",
    "prism", "disk", "block", "box", "ring", "star",
]

# Template variations for the task
# These templates leverage GPT-2's natural completion tendencies
# Key insight: End with "which is the X" or "it's the X" to force object prediction
TEMPLATES = [
    # Pattern: "The X one is the Y" - very natural completion
    "The {color1} {object1} and the {color2} {object2} are here. I prefer the {pref_color} one, which is the",
    "I see a {color1} {object1} and a {color2} {object2}. I want the {pref_color} one, which is the",
    
    # Pattern: "I'll take the X" but make it clearer  
    "There is a {color1} {object1} and a {color2} {object2}. I like the {pref_color} one. I'll take the",
    "Here are a {color1} {object1} and a {color2} {object2}. I prefer the {pref_color} one. I'll choose the",
    
    # Pattern: Direct reference
    "A {color1} {object1} and a {color2} {object2} are available. The {pref_color} item is the",
    "The items are: a {color1} {object1} and a {color2} {object2}. The {pref_color} item is the",
]


def sample_items(items: List[str], n: int, exclude: Set[str] = None) -> List[str]:
    """Sample n unique items, excluding any in the exclude set."""
    available = [item for item in items if exclude is None or item not in exclude]
    if len(available) < n:
        raise ValueError(f"Not enough items available. Need {n}, have {len(available)}")
    return random.sample(available, n)


def generate_color_object_pair(template: str, preferred_first: bool = True) -> ColorObjectPair:
    """
    Generate a single clean/corrupt color-object pair.
    
    Args:
        template: Template string to use
        preferred_first: If True, preferred color is color1. If False, preferred color is color2.
    
    Clean: Uses two color-object pairs, preferred color is in context
    Corrupt: Uses two DIFFERENT color-object pairs, preferred color NOT in context
    """
    # Sample colors and objects for clean sentence (all unique)
    color1, color2 = sample_items(COLORS, 2)
    object1, object2 = sample_items(OBJECTS, 2)
    
    # Determine which color is preferred
    preferred_color = color1 if preferred_first else color2
    correct_object = object1 if preferred_first else object2
    incorrect_object = object2 if preferred_first else object1
    
    # Clean sentence: preferred color IS in context
    clean_text = template.format(
        color1=color1,
        object1=object1,
        color2=color2,
        object2=object2,
        pref_color=preferred_color
    )
    
    # Corrupt sentence: use DIFFERENT colors and objects, preferred color NOT in context
    # Exclude clean colors and THE PREFERRED COLOR to ensure it's not in corrupt context
    exclude_colors = {color1, color2, preferred_color}
    
    # Make sure we have enough colors to sample from
    if len(COLORS) - len(exclude_colors) < 2:
        # If not enough colors, only exclude the preferred color
        exclude_colors = {preferred_color}
    
    corrupt_color1, corrupt_color2 = sample_items(
        COLORS, 2, exclude=exclude_colors
    )
    corrupt_object1, corrupt_object2 = sample_items(
        OBJECTS, 2, exclude={object1, object2}
    )
    
    corrupt_text = template.format(
        color1=corrupt_color1,
        object1=corrupt_object1,
        color2=corrupt_color2,
        object2=corrupt_object2,
        pref_color=preferred_color  # Same preferred color, but NOT in corrupt context
    )
    
    return ColorObjectPair(
        clean_text=clean_text,
        corrupt_text=corrupt_text,
        correct_object=correct_object,
        incorrect_object=incorrect_object,
        preferred_color=preferred_color,
        color1=color1,
        color2=color2,
        object1=object1,
        object2=object2,
    )


def generate_dataset(
    n_examples: int,
    templates: List[str] = None,
    output_file: str = None,
    seed: int = 42
) -> List[ColorObjectPair]:
    """
    Generate a dataset of color-object association pairs.
    
    Args:
        n_examples: Number of examples to generate
        templates: List of template strings (uses default if None)
        output_file: Path to save JSON output (optional)
        seed: Random seed for reproducibility
    
    Returns:
        List of ColorObjectPair objects
    """
    random.seed(seed)
    
    if templates is None:
        templates = TEMPLATES
    
    pairs = []
    
    for i in range(n_examples):
        # Randomly choose template
        template = random.choice(templates)
        
        # Alternate between preferring first and second color for balance
        preferred_first = (i % 2 == 0)
        
        try:
            pair = generate_color_object_pair(template, preferred_first)
            pairs.append(pair)
        except ValueError as e:
            print(f"Warning: Could not generate pair {i}: {e}")
            continue
    
    # Save to file if specified
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump([pair.to_dict() for pair in pairs], f, indent=2)
        
        print(f"Generated {len(pairs)} color-object pairs")
        print(f"Saved to: {output_path}")
    
    return pairs


def generate_standard_datasets():
    """Generate small, medium, and large datasets for experiments."""
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Small dataset for quick testing
    generate_dataset(
        n_examples=100,
        output_file=output_dir / "color_pairs_small.json",
        seed=42
    )
    
    # Medium dataset for initial experiments
    generate_dataset(
        n_examples=500,
        output_file=output_dir / "color_pairs_medium.json",
        seed=42
    )
    
    # Large dataset for comprehensive analysis
    generate_dataset(
        n_examples=1000,
        output_file=output_dir / "color_pairs_large.json",
        seed=42
    )
    
    print("\nDataset generation complete!")
    print(f"Output directory: {output_dir}")


def print_example_pairs(n: int = 3):
    """Print a few example pairs to show what the dataset looks like."""
    print("=" * 80)
    print("EXAMPLE COLOR-OBJECT ASSOCIATION PAIRS")
    print("=" * 80)
    
    for i in range(n):
        pair = generate_color_object_pair(random.choice(TEMPLATES), preferred_first=(i % 2 == 0))
        
        print(f"\n--- Example {i+1} ---")
        print(f"CLEAN:   {pair.clean_text}")
        print(f"         → Expected: '{pair.correct_object}' (not '{pair.incorrect_object}')")
        print(f"\nCORRUPT: {pair.corrupt_text}")
        print(f"         → Model should be confused ('{pair.preferred_color}' not in context)")
        print(f"\nMetadata:")
        print(f"  - Preferred color: {pair.preferred_color}")
        print(f"  - Color-object pairs (clean): {pair.color1}↔{pair.object1}, {pair.color2}↔{pair.object2}")
        print(f"  - Correct object: {pair.correct_object}")
        print(f"  - Incorrect object: {pair.incorrect_object}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Show examples first
    print_example_pairs(3)
    
    # Generate standard datasets
    print("\nGenerating datasets...")
    generate_standard_datasets()
