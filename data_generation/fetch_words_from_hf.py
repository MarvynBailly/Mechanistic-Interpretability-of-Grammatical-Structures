"""
Fetch Words from Hugging Face Datasets

This script fetches words from Hugging Face's datasets that follow specific
grammatical patterns (e.g., plural nouns with 's' suffix, past tense verbs
with 'ed' suffix). As described in the project proposal, this allows for
automatic generation of word lists for the dataset.

The words can then be translated to Chinese using ChatGPT and manually verified
by a native speaker.
"""

import json
import re
from typing import List, Dict, Set
from collections import defaultdict


def filter_plural_nouns_with_s(words: List[str]) -> List[str]:
    """
    Filter words that are likely plural nouns ending with 's'

    Args:
        words: List of candidate words

    Returns:
        Filtered list of plural nouns
    """
    plural_nouns = []

    # Common plural patterns
    plural_patterns = [
        r"\w+s$",  # General plural ending in 's'
    ]

    # Exclude common false positives
    exclude_patterns = [
        r"^[a-z]s$",  # Single letter + s
        r"ss$",  # Ends in double s (like 'class', 'glass')
        r"us$",  # Ends in 'us' (often not plural)
        r"is$",  # Ends in 'is' (often not plural)
        r"as$",  # Often not plural
    ]

    for word in words:
        word = word.lower().strip()

        # Skip if too short or too long
        if len(word) < 4 or len(word) > 15:
            continue

        # Check if matches plural pattern
        matches_plural = any(re.match(pattern, word) for pattern in plural_patterns)

        # Check if matches exclusion pattern
        matches_exclusion = any(
            re.search(pattern, word) for pattern in exclude_patterns
        )

        if matches_plural and not matches_exclusion:
            plural_nouns.append(word)

    return plural_nouns


def filter_past_tense_verbs_with_ed(words: List[str]) -> List[str]:
    """
    Filter words that are likely past tense verbs ending with 'ed'

    Args:
        words: List of candidate words

    Returns:
        Filtered list of past tense verbs
    """
    past_verbs = []

    # Past tense pattern
    past_pattern = r"\w+ed$"

    # Exclude common false positives
    exclude_patterns = [
        r"^[a-z]{1,2}ed$",  # Too short
        r"eed$",  # Often nouns (like 'seed', 'feed')
    ]

    for word in words:
        word = word.lower().strip()

        # Skip if too short or too long
        if len(word) < 5 or len(word) > 15:
            continue

        # Check if matches past tense pattern
        matches_past = re.match(past_pattern, word)

        # Check if matches exclusion pattern
        matches_exclusion = any(
            re.search(pattern, word) for pattern in exclude_patterns
        )

        if matches_past and not matches_exclusion:
            past_verbs.append(word)

    return past_verbs


def filter_singular_nouns(words: List[str]) -> List[str]:
    """
    Filter words that are likely singular nouns

    Args:
        words: List of candidate words

    Returns:
        Filtered list of singular nouns
    """
    singular_nouns = []

    # Exclude patterns that suggest it's not a singular noun
    exclude_patterns = [
        r"s$",  # Likely plural
        r"ed$",  # Likely verb
        r"ing$",  # Likely gerund/verb
        r"ly$",  # Likely adverb
    ]

    for word in words:
        word = word.lower().strip()

        # Skip if too short or too long
        if len(word) < 3 or len(word) > 15:
            continue

        # Check if matches exclusion pattern
        matches_exclusion = any(
            re.search(pattern, word) for pattern in exclude_patterns
        )

        # Only include if it doesn't match exclusions
        if not matches_exclusion and word.isalpha():
            singular_nouns.append(word)

    return singular_nouns


def load_word_list_from_file(filepath: str) -> List[str]:
    """
    Load a word list from a text file (one word per line)

    Args:
        filepath: Path to the word list file

    Returns:
        List of words
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            words = [line.strip() for line in f if line.strip()]
        return words
    except FileNotFoundError:
        print(f"Error: File {filepath} not found")
        return []


def fetch_words_by_pattern(
    word_source: List[str], grammatical_pattern: str, max_words: int = 20
) -> List[Dict[str, str]]:
    """
    Fetch words matching a specific grammatical pattern

    Args:
        word_source: Source list of words
        grammatical_pattern: Pattern type ('plural_s', 'past_ed', 'singular')
        max_words: Maximum number of words to return

    Returns:
        List of word dictionaries with metadata
    """
    if grammatical_pattern == "plural_s":
        filtered = filter_plural_nouns_with_s(word_source)
        feature = "plural_noun"
        pos = "noun"
    elif grammatical_pattern == "past_ed":
        filtered = filter_past_tense_verbs_with_ed(word_source)
        feature = "past_tense_verb_ed"
        pos = "verb"
    elif grammatical_pattern == "singular":
        filtered = filter_singular_nouns(word_source)
        feature = "singular_noun"
        pos = "noun"
    else:
        print(f"Unknown pattern: {grammatical_pattern}")
        return []

    # Remove duplicates and sort
    filtered = sorted(set(filtered))

    # Take top max_words
    filtered = filtered[:max_words]

    # Create word dictionaries
    word_dicts = []
    for idx, word in enumerate(filtered):
        word_dicts.append({"word": word, "grammatical_feature": feature, "pos": pos})

    return word_dicts


def generate_word_list_from_common_words(output_file: str = "generated_words.json"):
    """
    Generate a word list using common English words

    This is a fallback method when HuggingFace dataset is not available,
    as mentioned in the project proposal.

    Args:
        output_file: Output JSON file path
    """
    # Common English words (you can expand this list)
    common_words = [
        # Singular nouns
        "book",
        "pen",
        "desk",
        "chair",
        "table",
        "door",
        "window",
        "phone",
        "computer",
        "laptop",
        "tablet",
        "mouse",
        "keyboard",
        "screen",
        "paper",
        "note",
        "file",
        "folder",
        "document",
        "report",
        "letter",
        "email",
        "message",
        "card",
        "ticket",
        "key",
        "wallet",
        "bag",
        "cup",
        "plate",
        "bowl",
        "fork",
        "knife",
        "spoon",
        "bottle",
        "apple",
        "orange",
        "banana",
        "sandwich",
        "coffee",
        "tea",
        "water",
        # Plural nouns
        "books",
        "pens",
        "files",
        "papers",
        "notes",
        "cards",
        "tickets",
        "ideas",
        "plans",
        "results",
        "details",
        "photos",
        "images",
        "documents",
        "reports",
        "emails",
        "messages",
        "items",
        "tools",
        "slides",
        "charts",
        "graphs",
        "tables",
        "figures",
        "numbers",
        "samples",
        "copies",
        "versions",
        "updates",
        "changes",
        "edits",
        # Past tense verbs
        "walked",
        "talked",
        "worked",
        "played",
        "helped",
        "called",
        "asked",
        "answered",
        "learned",
        "studied",
        "finished",
        "started",
        "opened",
        "closed",
        "moved",
        "changed",
        "created",
        "deleted",
        "saved",
        "loaded",
        "checked",
        "tested",
        "fixed",
        "solved",
        "watched",
        "listened",
        "showed",
        "explained",
        "described",
        "presented",
    ]

    # Categorize words
    all_words = []

    # Get singular nouns
    singular = fetch_words_by_pattern(common_words, "singular", max_words=15)
    all_words.extend(singular)

    # Get plural nouns
    plural = fetch_words_by_pattern(common_words, "plural_s", max_words=15)
    all_words.extend(plural)

    # Get past tense verbs
    past = fetch_words_by_pattern(common_words, "past_ed", max_words=15)
    all_words.extend(past)

    # Add IDs
    for idx, word_dict in enumerate(all_words):
        word_dict["id"] = idx

    # Create output structure
    output = {
        "words": all_words[:60],  # Limit to 60 as specified in proposal
        "chinese_translations": {},  # To be filled in by ChatGPT translation
        "metadata": {
            "total_words": len(all_words[:60]),
            "source": "common_english_words",
            "note": "Chinese translations need to be added and verified by native speaker",
        },
    }

    # Add placeholder Chinese translations
    for word_dict in output["words"]:
        output["chinese_translations"][word_dict["word"]] = (
            f"[TRANSLATE: {word_dict['word']}]"
        )

    # Save to file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Generated {len(output['words'])} words")
    print(f"Saved to {output_file}")
    print("\nWord distribution:")
    feature_counts = defaultdict(int)
    for w in output["words"]:
        feature_counts[w["grammatical_feature"]] += 1
    for feature, count in feature_counts.items():
        print(f"  {feature}: {count}")

    print("\nNext steps:")
    print("1. Use ChatGPT to translate the words to Chinese")
    print("2. Have translations manually verified by a native Chinese speaker")
    print("3. Update the chinese_translations dictionary in the output file")


def main():
    """
    Main function to fetch and filter words from Hugging Face or local source

    Note: This implementation uses common words as a fallback.
    To use Hugging Face datasets, you would need to:
    1. Install: pip install datasets
    2. Load a dataset: from datasets import load_dataset
    3. Extract words from the dataset
    4. Apply the filtering functions above
    """
    print("=" * 60)
    print("Word List Generator for Grammatical Structure Analysis")
    print("=" * 60)
    print()

    # Option 1: Generate from common words (fallback method)
    print("Generating word list from common English words...")
    generate_word_list_from_common_words("fetched_words.json")

    # Option 2: Load from custom word list file
    # Uncomment and modify if you have a custom word list
    # print("\nLoading words from custom file...")
    # words = load_word_list_from_file('custom_wordlist.txt')
    # # Apply filtering as needed
    # filtered = fetch_words_by_pattern(words, 'plural_s', max_words=20)
    # print(f"Found {len(filtered)} matching words")


if __name__ == "__main__":
    main()
