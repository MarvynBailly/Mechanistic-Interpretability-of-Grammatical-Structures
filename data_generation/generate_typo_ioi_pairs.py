"""
Typo ("blurry search") IOI dataset generator.

Builds on the English IOI task, but one mention of the subject is misspelled.
The question is whether GPT-2's name-matching machinery (duplicate token,
induction, S-inhibition heads) still treats the typo'd name as the same person.

    exact:     "When Mary and John went to the store, John gave the book to"   -> Mary
    swap:      "When Mary and John went to the store, Jhon gave the book to"   -> Mary
    drop:      "When Mary and John went to the store, Jon gave the book to"    -> Mary
    double:    "When Mary and John went to the store, Johhn gave the book to"  -> Mary
    sub:       "When Mary and John went to the store, Jojn gave the book to"   -> Mary
    unrelated: "When Mary and John went to the store, Linda gave the book to"  -> ? (control)

All conditions share the same base examples (IO, S, template, object), so the
only thing that changes between conditions is the typo.

Typo'd names are often several BPE tokens while the original is one. So that
path patching can patch every position, each corrupt prompt uses random names
where the typo'd slot gets the same kind of typo and the SAME token length as
in the clean prompt. Clean and corrupt are therefore position-aligned.

typo_target controls which mention is misspelled:
    "s2": the repeated subject in the main clause (default)
    "s1": the first mention in the dependent clause
"""

import json
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional

from transformers import GPT2TokenizerFast

from generate_eng_ioi_pairs import NAMES, OBJECTS, TEMPLATES


CONDITIONS = ["exact", "swap", "drop", "double", "sub", "unrelated"]

# QWERTY neighbours for substitution typos
KEYBOARD_NEIGHBORS = {
    "q": "wa", "w": "qes", "e": "wrd", "r": "etf", "t": "ryg", "y": "tuh",
    "u": "yij", "i": "uok", "o": "ipl", "p": "ol", "a": "qsz", "s": "adw",
    "d": "sfe", "f": "dgr", "g": "fht", "h": "gjy", "j": "hku", "k": "jli",
    "l": "kop", "z": "asx", "x": "zcs", "c": "xvd", "v": "cbf", "b": "vng",
    "n": "bmh", "m": "nj",
}


# ============================================================================
# Typo operations (first letter is kept, as in most real typos)
# ============================================================================

def typo_swap(name: str, rng: random.Random) -> str:
    """Transpose two adjacent interior characters: John -> Jhon."""
    candidates = [i for i in range(1, len(name) - 1) if name[i] != name[i + 1]]
    i = rng.choice(candidates)
    return name[:i] + name[i + 1] + name[i] + name[i + 2:]


def typo_drop(name: str, rng: random.Random) -> str:
    """Delete one interior character: John -> Jon."""
    i = rng.randrange(1, len(name))
    return name[:i] + name[i + 1:]


def typo_double(name: str, rng: random.Random) -> str:
    """Duplicate one interior character: John -> Johhn."""
    i = rng.randrange(1, len(name))
    return name[:i] + name[i] + name[i:]


def typo_sub(name: str, rng: random.Random) -> str:
    """Replace one interior character with a keyboard neighbour: John -> Jojn."""
    i = rng.randrange(1, len(name))
    return name[:i] + rng.choice(KEYBOARD_NEIGHBORS[name[i]]) + name[i + 1:]


TYPO_FNS: Dict[str, Callable[[str, random.Random], str]] = {
    "swap": typo_swap,
    "drop": typo_drop,
    "double": typo_double,
    "sub": typo_sub,
}


def make_typo(name: str, kind: str, rng: random.Random, max_tries: int = 50) -> str:
    """Apply a typo of the given kind, avoiding results that are a listed name."""
    for _ in range(max_tries):
        typo = TYPO_FNS[kind](name, rng)
        if typo != name and typo not in NAMES:
            return typo
    raise ValueError(f"Could not make a '{kind}' typo of {name}")


# ============================================================================
# Pair generation
# ============================================================================

def n_tokens(tokenizer: GPT2TokenizerFast, word: str) -> int:
    """Number of GPT-2 tokens for a mid-sentence word (leading space)."""
    return len(tokenizer.encode(" " + word))


def sample_base_examples(n_examples: int, seed: int) -> List[Dict]:
    """Sample the shared IO/S/template/object skeleton used by every condition."""
    rng = random.Random(seed)
    base = []
    for _ in range(n_examples):
        io_name, s_name = rng.sample(NAMES, 2)
        base.append({
            "template": rng.choice(TEMPLATES),
            "object": rng.choice(OBJECTS),
            "io_name": io_name,
            "s_name": s_name,
        })
    return base


def name_char_spans(text: str, io: str, s1: str, s2: str) -> Dict[str, List[int]]:
    """Character [start, end) spans of the IO, S1 and S2 mentions, in order."""
    spans = {}
    cursor = 0
    for key, name in (("io", io), ("s1", s1), ("s2", s2)):
        start = text.index(" " + name + " ", cursor) + 1
        spans[key] = [start, start + len(name)]
        cursor = start + len(name)
    return spans


def build_pair(
    base: Dict,
    condition: str,
    typo_target: str,
    tokenizer: GPT2TokenizerFast,
    rng: random.Random,
    max_tries: int = 200,
) -> Optional[Dict]:
    """
    Build one clean/corrupt pair for a condition.

    Returns None if no length-matched corrupt prompt could be found.
    """
    io_name, s_name = base["io_name"], base["s_name"]
    fill = dict(IO=io_name, S1=s_name, S2=s_name, object=base["object"])
    other_slot = "S1" if typo_target == "s2" else "S2"
    typo_slot = typo_target.upper()

    if condition == "exact":
        variant = s_name
    elif condition == "unrelated":
        variant = rng.choice([n for n in NAMES if n not in (io_name, s_name)])
    else:
        variant = make_typo(s_name, condition, rng)
    fill[typo_slot] = variant
    clean = base["template"].format(**fill)
    target_len = n_tokens(tokenizer, variant)

    # Corrupt: random names, with the typo'd slot getting the same kind of
    # perturbation and the same token length so positions line up.
    for _ in range(max_tries):
        r1, r2, r3 = rng.sample([n for n in NAMES if n not in (io_name, s_name)], 3)
        corrupt_variant = r3 if condition in ("exact", "unrelated") else make_typo(r3, condition, rng)
        if n_tokens(tokenizer, corrupt_variant) != target_len:
            continue
        cfill = dict(IO=r1, object=base["object"])
        cfill[other_slot] = r2
        cfill[typo_slot] = corrupt_variant
        corrupt = base["template"].format(**cfill)
        if len(tokenizer.encode(corrupt)) == len(tokenizer.encode(clean)):
            break
    else:
        return None

    return {
        "clean": clean,
        "corrupt": corrupt,
        "io_name": io_name,
        "s_name": s_name,
        "io_token": f" {io_name}",
        "condition": condition,
        "typo_target": typo_target,
        "variant": variant,
        "variant_n_tokens": target_len,
        "clean_spans": name_char_spans(clean, fill["IO"], fill["S1"], fill["S2"]),
    }


def generate_typo_dataset(
    n_examples: int = 500,
    typo_target: str = "s2",
    seed: int = 42,
) -> Dict[str, List[Dict]]:
    """
    Generate pairs for every condition from the same base examples.

    Examples whose corrupt prompt can't be length-matched in some condition are
    dropped from ALL conditions, so each condition has identical base examples.
    """
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    base = sample_base_examples(n_examples, seed)

    by_condition = {c: [] for c in CONDITIONS}
    for i, b in enumerate(base):
        rng = random.Random(seed * 100_003 + i)
        pairs = {c: build_pair(b, c, typo_target, tokenizer, rng) for c in CONDITIONS}
        if any(p is None for p in pairs.values()):
            continue
        for c, p in pairs.items():
            p["example_id"] = i
            by_condition[c].append(p)

    return by_condition


def save_dataset(by_condition: Dict[str, List[Dict]], typo_target: str, output_path: Path):
    """Save all conditions to one JSON file."""
    data = {
        "description": "Typo IOI pairs: one subject mention misspelled; clean/corrupt are token-aligned",
        "typo_target": typo_target,
        "conditions": CONDITIONS,
        "n_examples": len(by_condition["exact"]),
        "pairs": by_condition,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved {data['n_examples']} examples x {len(CONDITIONS)} conditions to {output_path}")


def main():
    output_dir = Path(__file__).parent / "output"
    for typo_target in ["s2", "s1"]:
        by_condition = generate_typo_dataset(n_examples=500, typo_target=typo_target, seed=42)
        save_dataset(by_condition, typo_target, output_dir / f"typo_ioi_pairs_{typo_target}.json")

        print(f"\nExamples (typo at {typo_target.upper()}):")
        for c in CONDITIONS:
            p = by_condition[c][0]
            print(f"  [{c:9s}] {p['clean']}  -> {p['io_token']}")
            print(f"  {'':11s} {p['corrupt']}")


if __name__ == "__main__":
    main()
