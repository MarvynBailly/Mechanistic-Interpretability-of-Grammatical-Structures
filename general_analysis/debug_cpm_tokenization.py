"""
Debug tokenization for CPM to understand why it's performing poorly.
"""

from transformers import AutoTokenizer
import json
from pathlib import Path

# Load CPM tokenizer
print("Loading CPM tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("TsinghuaAI/CPM-Generate", trust_remote_code=True)

# Load one example
data_path = Path(__file__).parent.parent / "data_generation" / "output" / "chinese_ioi_pairs_small.json"
with open(data_path, 'r', encoding='utf-8') as f:
    pairs = json.load(f)['pairs']

# Test tokenization
example = pairs[0]
print("\n" + "="*80)
print("TOKENIZATION DEBUG")
print("="*80)

print(f"\nPrompt: {example['clean']}")
print(f"Expected answer: {example['io_name']}")

# Tokenize prompt
prompt_tokens = tokenizer.encode(example['clean'], add_special_tokens=True)
print(f"\nPrompt tokens: {prompt_tokens}")
print(f"Decoded: {tokenizer.decode(prompt_tokens)}")

# Tokenize expected name
name_tokens = tokenizer.encode(example['io_name'], add_special_tokens=False)
print(f"\nName '{example['io_name']}' tokens: {name_tokens}")
print(f"Number of tokens: {len(name_tokens)}")

for token_id in name_tokens:
    print(f"  Token ID {token_id}: '{tokenizer.decode([token_id])}'")

# Check if this is multi-token issue
print("\n" + "="*80)
print("ANALYSIS")
print("="*80)

if len(name_tokens) > 1:
    print(f"⚠️  Name is multi-token ({len(name_tokens)} tokens)")
    print("   This means the model must predict multiple tokens correctly.")
    print("   Top-1 accuracy will be low even if the model knows the answer.")
    print("\n   Solution: Check if first token of name appears in top-k predictions")
else:
    print(f"✓ Name is single token (id={name_tokens[0]})")
    print("   Model should be able to predict this directly.")

# Test all names in dataset
print("\n" + "="*80)
print("ALL NAMES TOKENIZATION")
print("="*80)

single_token_names = 0
multi_token_names = 0

for pair in pairs[:20]:  # Check first 20
    name = pair['io_name']
    tokens = tokenizer.encode(name, add_special_tokens=False)
    if len(tokens) == 1:
        single_token_names += 1
    else:
        multi_token_names += 1
        print(f"Multi-token: '{name}' -> {tokens} -> {[tokenizer.decode([t]) for t in tokens]}")

print(f"\nSingle-token names: {single_token_names}")
print(f"Multi-token names: {multi_token_names}")

if multi_token_names > single_token_names:
    print("\n⚠️  CRITICAL: Most names are multi-token!")
    print("   This explains the poor performance.")
    print("   Recommendation: Use generation-based evaluation or check first token only.")
