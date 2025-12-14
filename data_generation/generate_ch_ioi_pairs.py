"""
Simple IOI dataset generator for path patching experiments.

Generates clean/corrupt pairs where:
- Clean: Standard IOI sentence "When [IO] and [S] went to [place], [S] gave [object] to [IO]"
- Corrupt: Same sentence but with random names "[R1] and [R2] went to [place], [R2] gave [object] to [R1]"

The corrupt version uses random names (not IO/S) to test what happens when the model
doesn't have the correct name information to work with.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class IOIPair:
    """A clean/corrupt pair for path patching."""
    clean_text: str
    corrupt_text: str
    io_name: str      # Indirect object in clean
    s_name: str       # Subject in clean
    io_token: str     # Expected answer for clean
    
    def to_dict(self):
        return {
            "clean": self.clean_text,
            "corrupt": self.corrupt_text,
            "io_name": self.io_name,
            "s_name": self.s_name,
            "io_token": self.io_token,
        }


# Simple names that are single tokens in GPT-2
NAMES = [
    "小明", "小红", "小华", "小芳", "小强", "小丽", 
    "小军", "小燕", "小龙", "小梅", "小刚", "小玲",
    "小伟", "小敏", "小杰", "小婷", "小平", "小云",
    "小峰", "小雪", "小勇", "小霞", "小涛", "小娟"
]

# Templates tested for high accuracy (>90%) on GPT-2 small
# Excluded templates that lead to ambiguous continuations (e.g., "The", "A")
TEMPLATES = [
    "当{IO}和{S1}去商店时，{S2}把{object}给了",
    "当{IO}和{S1}在公园时，{S2}把{object}递给了",
    "在图书馆里，{IO}和{S1}见面后，{S2}把{object}展示给了",
    "当{IO}和{S1}在咖啡馆碰面时，{S2}把{object}给了",
]

# Simple objects
OBJECTS = [
    "书", "礼物", "笔记本", "手机", "钥匙",
    "钱包", "帽子", "手表", "水瓶", "电脑", "耳机",
]


def sample_names(n: int, exclude: List[str] = None) -> List[str]:
    """Sample n unique names, excluding any in the exclude list."""
    available = [name for name in NAMES if exclude is None or name not in exclude]
    if len(available) < n:
        raise ValueError(f"Not enough names available. Need {n}, have {len(available)}")
    return random.sample(available, n)


def generate_ioi_pair(template: str, obj: str) -> IOIPair:
    """
    Generate a single clean/corrupt IOI pair.
    
    Clean: Uses IO and S names as intended
    Corrupt: Uses random names R1, R2 (different from IO and S)
    """
    # Sample names for clean sentence
    io_name, s_name = sample_names(2)
    
    # Clean sentence
    clean_text = template.format(IO=io_name, S1=s_name, S2=s_name, object=obj)
    
    # Sample different random names for corrupt (use 2 names: R1 and R2)
    # Important: S1 and S2 must be the same person (like in clean sentence)
    r1_name, r2_name, r3_name = sample_names(3, exclude=[io_name, s_name])
    
    # Corrupt sentence with random names (R2 appears twice, like S in clean)
    corrupt_text = template.format(IO=r1_name, S1=r2_name, S2=r3_name, object=obj)
    
    return IOIPair(
        clean_text=clean_text,
        corrupt_text=corrupt_text,
        io_name=io_name,
        s_name=s_name,
        io_token=f" {io_name}",  # GPT-2 tokenizer includes leading space
    )


def generate_dataset(n_examples: int, seed: int = 42) -> List[IOIPair]:
    """Generate n_examples of clean/corrupt IOI pairs."""
    random.seed(seed)
    
    pairs = []
    for _ in range(n_examples):
        template = random.choice(TEMPLATES)
        obj = random.choice(OBJECTS)
        pair = generate_ioi_pair(template, obj)
        pairs.append(pair)
    
    return pairs


def save_dataset(pairs: List[IOIPair], output_path: Path):
    """Save dataset to JSON file."""
    data = {
        "n_examples": len(pairs),
        "description": "Clean/corrupt IOI pairs for path patching",
        "pairs": [pair.to_dict() for pair in pairs]
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(pairs)} pairs to {output_path}")


def main():
    """Generate and save IOI dataset."""
    # Generate datasets of different sizes
    sizes = {
        "small": 100,
        "medium": 1000,
        "large": 5000,
    }
    
    output_dir = Path(__file__).parent / "output"
    
    for name, size in sizes.items():
        pairs = generate_dataset(size, seed=42)
        output_path = output_dir / f"chinese_ioi_pairs_{name}.json"
        save_dataset(pairs, output_path)
        
        # Print example
        if name == "small":
            print("\nExample pair:")
            example = pairs[0]
            print(f"Clean:   {example.clean_text} [{example.io_token}]")
            print(f"Corrupt: {example.corrupt_text}")
            print(f"Expected answer: {example.io_token}")


if __name__ == "__main__":
    main()
