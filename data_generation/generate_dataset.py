"""
Data Generation Pipeline for Mechanistic Interpretability of Grammatical Structures

This script generates English-Chinese parallel datasets for studying grammatical structures
in multilingual language models. It creates:
- IOI (Indirect Object Identification) sentence pairs
- Clean and fuzzy variants for path patching experiments
- 900 total sequences (60 words × 15 templates)

Based on the project proposal's data generation methodology.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple
import random
from dataclasses import dataclass, asdict


@dataclass
class SentencePair:
    """Represents an English-Chinese sentence pair with metadata"""

    template_id: int
    word_id: int
    english: str
    chinese: str
    word: str
    grammatical_feature: str
    is_fuzzy: bool = False
    fuzzy_type: str = None  # 'missing_function_words', 'shuffled', 'misspelled'


class DatasetGenerator:
    """Main class for generating the parallel English-Chinese dataset"""

    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Templates for IOI sentences
        self.templates = []
        self.words = []
        self.chinese_translations = {}

    def load_templates(self, templates_file: str):
        """Load sentence templates from JSON file"""
        with open(templates_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.templates = data["templates"]
        print(f"Loaded {len(self.templates)} templates")

    def load_words(self, words_file: str):
        """Load word list with Chinese translations from JSON file"""
        with open(words_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.words = data["words"]
            self.chinese_translations = data.get("chinese_translations", {})
        print(f"Loaded {len(self.words)} words")

    def fill_template(self, template: Dict[str, str], word: str) -> Tuple[str, str]:
        """
        Fill a template with the given word

        Args:
            template: Dictionary with 'english' and 'chinese' template strings
            word: The word to insert into the template

        Returns:
            Tuple of (english_sentence, chinese_sentence)
        """
        english = template["english"].replace("{word}", word)
        chinese_word = self.chinese_translations.get(word, word)
        chinese = template["chinese"].replace("{word}", chinese_word)

        return english, chinese

    def generate_clean_pairs(self) -> List[SentencePair]:
        """Generate clean (grammatical) sentence pairs"""
        pairs = []

        for word_idx, word_data in enumerate(self.words):
            word = word_data["word"]
            feature = word_data.get("grammatical_feature", "unknown")

            for template_idx, template in enumerate(self.templates):
                english, chinese = self.fill_template(template, word)

                pair = SentencePair(
                    template_id=template_idx,
                    word_id=word_idx,
                    english=english,
                    chinese=chinese,
                    word=word,
                    grammatical_feature=feature,
                    is_fuzzy=False,
                )
                pairs.append(pair)

        print(f"Generated {len(pairs)} clean sentence pairs")
        return pairs

    def create_fuzzy_variant_missing_function_words(self, sentence: str) -> str:
        """Remove function words from a sentence"""
        # Common English function words
        function_words = [
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "to",
            "of",
            "and",
            "in",
            "on",
            "at",
            "by",
            "for",
        ]

        words = sentence.split()
        filtered_words = [w for w in words if w.lower() not in function_words]
        return " ".join(filtered_words)

    def create_fuzzy_variant_shuffled(self, sentence: str) -> str:
        """Shuffle word order in a sentence"""
        words = sentence.split()
        random.shuffle(words)
        return " ".join(words)

    def create_fuzzy_variant_misspelled(self, sentence: str) -> str:
        """Introduce random misspellings"""
        words = sentence.split()
        if len(words) == 0:
            return sentence

        # Randomly misspell 1-2 words
        num_to_misspell = min(random.randint(1, 2), len(words))
        indices_to_misspell = random.sample(range(len(words)), num_to_misspell)

        for idx in indices_to_misspell:
            word = words[idx]
            if len(word) > 3:
                # Swap two adjacent characters
                pos = random.randint(0, len(word) - 2)
                word_list = list(word)
                word_list[pos], word_list[pos + 1] = word_list[pos + 1], word_list[pos]
                words[idx] = "".join(word_list)

        return " ".join(words)

    def generate_fuzzy_pairs(
        self, clean_pairs: List[SentencePair]
    ) -> List[SentencePair]:
        """Generate fuzzy (ungrammatical) variants of clean pairs"""
        fuzzy_pairs = []
        fuzzy_types = ["missing_function_words", "shuffled", "misspelled"]

        for clean_pair in clean_pairs:
            for fuzzy_type in fuzzy_types:
                if fuzzy_type == "missing_function_words":
                    english = self.create_fuzzy_variant_missing_function_words(
                        clean_pair.english
                    )
                    chinese = self.create_fuzzy_variant_missing_function_words(
                        clean_pair.chinese
                    )
                elif fuzzy_type == "shuffled":
                    english = self.create_fuzzy_variant_shuffled(clean_pair.english)
                    chinese = self.create_fuzzy_variant_shuffled(clean_pair.chinese)
                elif fuzzy_type == "misspelled":
                    english = self.create_fuzzy_variant_misspelled(clean_pair.english)
                    chinese = self.create_fuzzy_variant_misspelled(clean_pair.chinese)

                fuzzy_pair = SentencePair(
                    template_id=clean_pair.template_id,
                    word_id=clean_pair.word_id,
                    english=english,
                    chinese=chinese,
                    word=clean_pair.word,
                    grammatical_feature=clean_pair.grammatical_feature,
                    is_fuzzy=True,
                    fuzzy_type=fuzzy_type,
                )
                fuzzy_pairs.append(fuzzy_pair)

        print(f"Generated {len(fuzzy_pairs)} fuzzy sentence pairs")
        return fuzzy_pairs

    def generate_path_patching_pairs(
        self, clean_pairs: List[SentencePair]
    ) -> List[Dict[str, Any]]:
        """
        Generate pairs for path patching experiments
        Each pair consists of a clean sentence and its corrupted variant
        """
        patching_pairs = []

        for idx, clean_pair in enumerate(clean_pairs):
            # Create corrupted variant (using shuffled as default corruption)
            corrupted_english = self.create_fuzzy_variant_shuffled(clean_pair.english)
            corrupted_chinese = self.create_fuzzy_variant_shuffled(clean_pair.chinese)

            pair = {
                "pair_id": idx,
                "clean": {"english": clean_pair.english, "chinese": clean_pair.chinese},
                "corrupted": {
                    "english": corrupted_english,
                    "chinese": corrupted_chinese,
                },
                "metadata": {
                    "template_id": clean_pair.template_id,
                    "word_id": clean_pair.word_id,
                    "word": clean_pair.word,
                    "grammatical_feature": clean_pair.grammatical_feature,
                },
            }
            patching_pairs.append(pair)

        print(f"Generated {len(patching_pairs)} path patching pairs")
        return patching_pairs

    def save_dataset(
        self,
        clean_pairs: List[SentencePair],
        fuzzy_pairs: List[SentencePair],
        patching_pairs: List[Dict[str, Any]],
    ):
        """Save all generated data to JSON files"""

        # Save clean pairs
        clean_file = self.output_dir / "clean_pairs.json"
        with open(clean_file, "w", encoding="utf-8") as f:
            json.dump(
                [asdict(pair) for pair in clean_pairs], f, ensure_ascii=False, indent=2
            )
        print(f"Saved clean pairs to {clean_file}")

        # Save fuzzy pairs
        fuzzy_file = self.output_dir / "fuzzy_pairs.json"
        with open(fuzzy_file, "w", encoding="utf-8") as f:
            json.dump(
                [asdict(pair) for pair in fuzzy_pairs], f, ensure_ascii=False, indent=2
            )
        print(f"Saved fuzzy pairs to {fuzzy_file}")

        # Save path patching pairs
        patching_file = self.output_dir / "path_patching_pairs.json"
        with open(patching_file, "w", encoding="utf-8") as f:
            json.dump(patching_pairs, f, ensure_ascii=False, indent=2)
        print(f"Saved path patching pairs to {patching_file}")

        # Save summary statistics
        stats = {
            "total_clean_pairs": len(clean_pairs),
            "total_fuzzy_pairs": len(fuzzy_pairs),
            "total_patching_pairs": len(patching_pairs),
            "num_templates": len(self.templates),
            "num_words": len(self.words),
            "expected_pairs": len(self.templates) * len(self.words),
        }

        stats_file = self.output_dir / "dataset_stats.json"
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        print(f"Saved statistics to {stats_file}")

    def generate_full_dataset(self, templates_file: str, words_file: str):
        """Complete pipeline to generate the full dataset"""
        print("=" * 60)
        print("Starting Data Generation Pipeline")
        print("=" * 60)

        # Load data
        self.load_templates(templates_file)
        self.load_words(words_file)

        # Generate clean pairs
        clean_pairs = self.generate_clean_pairs()

        # Generate fuzzy variants
        fuzzy_pairs = self.generate_fuzzy_pairs(clean_pairs)

        # Generate path patching pairs
        patching_pairs = self.generate_path_patching_pairs(clean_pairs)

        # Save all data
        self.save_dataset(clean_pairs, fuzzy_pairs, patching_pairs)

        print("=" * 60)
        print("Data Generation Complete!")
        print("=" * 60)


def main():
    """Main execution function"""
    # Initialize generator
    generator = DatasetGenerator(output_dir="output")

    templates_file = "templates.json"
    words_file = "words.json"

    if not os.path.exists(templates_file):
        print(f"Error: {templates_file} not found!")
        print("Please create templates.json with your sentence templates.")
        return

    if not os.path.exists(words_file):
        print(f"Error: {words_file} not found!")
        print("Please create words.json with your word list.")
        return

    generator.generate_full_dataset(templates_file, words_file)


if __name__ == "__main__":
    main()
