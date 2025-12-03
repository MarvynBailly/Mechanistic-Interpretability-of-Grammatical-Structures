"""
Python Code Completion Dataset Generator

Generates clean/corrupt pairs based on Python function argument completion.
This leverages GPT-2's code understanding to test variable binding and retrieval.

Task: Given a function with two arguments, predict the second argument after an operator.

Example Clean:
"def process(red, blue):\n    return red +" → "blue" ✓

Example Corrupt:  
"def process(green, yellow):\n    return red +" → ??? (red not in scope)
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass


@dataclass
class CodeCompletionPair:
    """A clean/corrupt pair for code completion experiments."""
    clean_text: str
    corrupt_text: str
    correct_arg: str      # Expected completion (second argument in clean)
    incorrect_arg: str    # First argument (should not be completed)
    var1_clean: str       # First variable in clean function
    var2_clean: str       # Second variable in clean function
    
    def to_dict(self):
        return {
            "clean": self.clean_text,
            "corrupt": self.corrupt_text,
            "correct_arg": self.correct_arg,
            "incorrect_arg": self.incorrect_arg,
            "var1_clean": self.var1_clean,
            "var2_clean": self.var2_clean,
        }


# Single-token variable names (colors, objects, letters)
# These are all single tokens in GPT-2
COLORS = ["red", "blue", "green", "yellow", "orange", "purple", "pink", "white", "black"]
OBJECTS = ["ball", "cube", "box", "star", "ring", "disk", "cone"]
LETTERS = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "m", "n", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]

# Function names and operators
FUNCTION_NAMES = ["process", "combine", "add", "merge", "mix", "join", "calc", "compute"]
OPERATORS = ["+", "*"]


def sample_variables(n: int, var_type: str = "colors", exclude: Set[str] = None) -> List[str]:
    """Sample n unique variables from the specified type."""
    if var_type == "colors":
        pool = COLORS
    elif var_type == "objects":
        pool = OBJECTS
    elif var_type == "letters":
        pool = LETTERS
    else:
        pool = LETTERS
    
    available = [v for v in pool if exclude is None or v not in exclude]
    if len(available) < n:
        raise ValueError(f"Not enough variables available. Need {n}, have {len(available)}")
    return random.sample(available, n)


def generate_code_pair(var_type: str = "colors") -> CodeCompletionPair:
    """
    Generate a single clean/corrupt code completion pair.
    
    Clean: Function with (var1, var2), returns var1 + var2
    Corrupt: Function with (var3, var4), returns var1 + ??? (var1 not in scope!)
    
    Args:
        var_type: Type of variables to use ('colors', 'objects', 'letters')
    """
    # Sample variables for clean function
    var1, var2 = sample_variables(2, var_type)
    
    # Sample different variables for corrupt
    var3, var4 = sample_variables(2, var_type, exclude={var1, var2})
    
    # Random function name and operator
    func_name = random.choice(FUNCTION_NAMES)
    operator = random.choice(OPERATORS)
    
    # Clean: var1 is in scope, should predict var2
    clean_text = f"def {func_name}({var1}, {var2}):\n    return {var1} {operator}"
    
    # Corrupt: var1 is NOT in scope (function has var3, var4), so model shouldn't know what to predict
    corrupt_text = f"def {func_name}({var3}, {var4}):\n    return {var1} {operator}"
    
    return CodeCompletionPair(
        clean_text=clean_text,
        corrupt_text=corrupt_text,
        correct_arg=var2,
        incorrect_arg=var1,
        var1_clean=var1,
        var2_clean=var2,
    )


def generate_dataset(
    n_examples: int,
    var_type: str = "colors",
    output_file: str = None,
    seed: int = 42
) -> List[CodeCompletionPair]:
    """
    Generate a dataset of code completion pairs.
    
    Args:
        n_examples: Number of examples to generate
        var_type: Type of variables ('colors', 'objects', 'letters')
        output_file: Path to save JSON output (optional)
        seed: Random seed for reproducibility
    
    Returns:
        List of CodeCompletionPair objects
    """
    random.seed(seed)
    
    pairs = []
    
    for i in range(n_examples):
        try:
            pair = generate_code_pair(var_type)
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
        
        print(f"Generated {len(pairs)} code completion pairs ({var_type})")
        print(f"Saved to: {output_path}")
    
    return pairs


def generate_standard_datasets():
    """Generate datasets with different variable types."""
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Color variables
    generate_dataset(
        n_examples=100,
        var_type="colors",
        output_file=output_dir / "code_pairs_colors.json",
        seed=42
    )
    
    # Object variables
    generate_dataset(
        n_examples=100,
        var_type="objects",
        output_file=output_dir / "code_pairs_objects.json",
        seed=42
    )
    
    # Letter variables (most natural for code)
    generate_dataset(
        n_examples=100,
        var_type="letters",
        output_file=output_dir / "code_pairs_letters.json",
        seed=42
    )
    
    print("\nDataset generation complete!")
    print(f"Output directory: {output_dir}")


def print_example_pairs(n: int = 3):
    """Print a few example pairs."""
    print("=" * 80)
    print("EXAMPLE CODE COMPLETION PAIRS")
    print("=" * 80)
    
    for var_type in ["colors", "objects", "letters"]:
        print(f"\n--- {var_type.upper()} ---")
        for i in range(n):
            pair = generate_code_pair(var_type)
            
            print(f"\nExample {i+1}:")
            print(f"CLEAN:")
            print(f"  {pair.clean_text}")
            print(f"  → Expected: ' {pair.correct_arg}'")
            
            print(f"CORRUPT:")
            print(f"  {pair.corrupt_text}")
            print(f"  → Model should be confused ('{pair.var1_clean}' not in scope)")


if __name__ == "__main__":
    # Show examples
    print_example_pairs(2)
    
    # Generate datasets
    print("\n\nGenerating datasets...")
    generate_standard_datasets()
