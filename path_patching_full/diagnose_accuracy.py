"""
Diagnostic script to understand clean accuracy.

This script analyzes what GPT-2 is actually predicting on clean IOI examples
to understand why accuracy might be lower than expected.
"""

import torch
from transformer_lens import HookedTransformer
import sys
from pathlib import Path

# Add path_patching_full to path
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import load_dataset_for_patching

def analyze_clean_predictions():
    """Analyze model predictions on clean examples."""
    
    # Load model
    print("Loading GPT-2 small...")
    model = HookedTransformer.from_pretrained("gpt2-small", device="cuda")
    
    # Load small dataset
    print("\nLoading dataset...")
    clean_tokens, corrupt_tokens, io_toks, s_toks, pairs = load_dataset_for_patching(
        model, size="medium", n_examples=100
    )
    
    print("\n" + "="*70)
    print("CLEAN SENTENCE PREDICTION ANALYSIS")
    print("="*70)
    
    with torch.inference_mode():
        clean_logits = model(clean_tokens)[:, -1, :]
    
    correct_count = 0
    total_count = len(pairs)
    
    for i in range(len(pairs)):
        pair = pairs[i]
        
        # Get top predictions
        top_k = 5
        top_logits, top_indices = clean_logits[i].topk(top_k)
        top_tokens = [model.to_string(idx) for idx in top_indices]
        
        # Find ranks of IO and S
        io_rank = (clean_logits[i] >= clean_logits[i, io_toks[i]]).sum().item()
        s_rank = (clean_logits[i] >= clean_logits[i, s_toks[i]]).sum().item()
        
        io_logit = clean_logits[i, io_toks[i]].item()
        s_logit = clean_logits[i, s_toks[i]].item()
        logit_diff = io_logit - s_logit
        
        is_correct = (io_rank == 1)
        if is_correct:
            correct_count += 1
        
        print(f"\n{'='*70}")
        print(f"Example {i+1}")
        print(f"{'='*70}")
        print(f"Clean prompt: {pair['clean']}")
        print(f"Expected IO: '{pair['io_name']}' (token: '{pair['io_token']}')")
        print(f"Subject S: '{pair['s_name']}'")
        
        print(f"\nTop {top_k} model predictions:")
        for j, (tok, logit) in enumerate(zip(top_tokens, top_logits)):
            marker = " ← IO" if top_indices[j] == io_toks[i] else (" ← S" if top_indices[j] == s_toks[i] else "")
            print(f"  {j+1}. '{tok}' (logit: {logit:.2f}){marker}")
        
        print(f"\nTarget token analysis:")
        print(f"  IO '{pair['io_name']}': rank={io_rank}, logit={io_logit:.2f}")
        print(f"  S  '{pair['s_name']}': rank={s_rank}, logit={s_logit:.2f}")
        print(f"  Logit diff (IO-S): {logit_diff:.2f}")
        print(f"  Result: {'✓ CORRECT' if is_correct else '✗ INCORRECT'}")
        
        if not is_correct:
            print(f"  Note: Model predicted '{top_tokens[0]}' instead of '{pair['io_name']}'")
    
    # Summary
    accuracy = correct_count / total_count
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Correct predictions: {correct_count}/{total_count}")
    print(f"Accuracy: {accuracy:.1%}")
    print(f"\nNote: This is expected behavior for GPT-2 on IOI tasks.")
    print(f"The model has learned the IOI pattern but isn't always confident.")
    print(f"The logit difference (IO vs S) is more important than raw accuracy.")


if __name__ == "__main__":
    analyze_clean_predictions()
