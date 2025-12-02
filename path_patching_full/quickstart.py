"""
Quick start script for path patching experiments.

This minimal example shows the simplest way to use path_patching_full:
1. Load model and dataset
2. Test a single important path
3. Done!
"""

from transformer_lens import HookedTransformer
from path_patching_full import (
    HeadSpec,
    ReceiverSpec,
    path_patch,
    load_dataset_for_patching,
)

# Load model
print("Loading GPT-2 small...")
model = HookedTransformer.from_pretrained("gpt2-small")

# Load dataset (10 examples for quick test)
print("\nLoading dataset...")
clean, corrupt, io, s, pairs = load_dataset_for_patching(
    model, size="small", n_examples=10
)

print(f"\nExample pair:")
print(f"  Clean:   {pairs[0]['clean']}")
print(f"  Corrupt: {pairs[0]['corrupt']}")

# Test an important path: L9H9 (name mover) → L10H0
print("\nTesting path: L9H9 → L10H0")
sender = HeadSpec(layer=9, head=9)
receiver = ReceiverSpec(layer=10, head=0, component='q')

effect = path_patch(model, clean, corrupt, sender, [receiver], io, s)

print(f"\nPath effect: {effect:.3f}")
print("✓ Done! Higher values indicate stronger information flow.")
