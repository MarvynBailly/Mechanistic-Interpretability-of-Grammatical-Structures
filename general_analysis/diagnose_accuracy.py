"""
Diagnostic script to understand why GPT-2 is not solving the color-object association task.

This script will:
1. Show example prompts and model predictions
2. Check if the expected tokens are single tokens
3. Examine the top predictions for clean vs corrupt
4. Suggest improvements to the task design
"""

import torch
from transformer_lens import HookedTransformer
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from data_loader import load_dataset_for_patching
from utils import set_seed


def diagnose_task():
    """Diagnose why the model isn't solving the color-object association task."""
    
    print("="*80)
    print("DIAGNOSTIC: Color-Object Association Task")
    print("="*80)
    
    # Setup
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\nLoading model...")
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    
    # Load dataset
    print(f"\nLoading dataset (first 10 examples)...")
    clean_tokens, corrupt_tokens, correct_toks, incorrect_toks, pairs = \
        load_dataset_for_patching(model, size="small", n_examples=10)
    
    # Examine first few examples
    print("\n" + "="*80)
    print("EXAMPLE PROMPTS AND PREDICTIONS")
    print("="*80)
    
    with torch.inference_mode():
        clean_logits = model(clean_tokens)[:, -1, :]
        corrupt_logits = model(corrupt_tokens)[:, -1, :]
    
    for i in range(min(5, len(pairs))):
        pair = pairs[i]
        
        print(f"\n--- Example {i+1} ---")
        print(f"CLEAN PROMPT:")
        print(f"  {pair['clean']}")
        print(f"  Expected: '{pair['correct_object']}'")
        print(f"  Distractor: '{pair['incorrect_object']}'")
        
        # Get clean predictions
        clean_logit = clean_logits[i]
        top5_clean = torch.topk(clean_logit, k=10)
        
        print(f"\n  Top 10 predictions on CLEAN:")
        for j, (tok_id, logit_val) in enumerate(zip(top5_clean.indices, top5_clean.values), 1):
            token_str = model.to_string(tok_id)
            is_correct = "✓ CORRECT" if token_str.strip() == pair['correct_object'] else ""
            is_incorrect = "✗ DISTRACTOR" if token_str.strip() == pair['incorrect_object'] else ""
            marker = is_correct or is_incorrect
            print(f"    {j}. '{token_str}' (logit: {logit_val:.2f}) {marker}")
        
        # Check logit difference
        correct_logit = clean_logit[correct_toks[i]].item()
        incorrect_logit = clean_logit[incorrect_toks[i]].item()
        print(f"\n  Logit difference: {correct_logit - incorrect_logit:.3f}")
        print(f"    Correct '{pair['correct_object']}': {correct_logit:.2f}")
        print(f"    Incorrect '{pair['incorrect_object']}': {incorrect_logit:.2f}")
        
        print(f"\n  CORRUPT PROMPT:")
        print(f"  {pair['corrupt']}")
        
        # Get corrupt predictions
        corrupt_logit = corrupt_logits[i]
        top5_corrupt = torch.topk(corrupt_logit, k=5)
        
        print(f"\n  Top 5 predictions on CORRUPT:")
        for j, (tok_id, logit_val) in enumerate(zip(top5_corrupt.indices, top5_corrupt.values), 1):
            token_str = model.to_string(tok_id)
            print(f"    {j}. '{token_str}' (logit: {logit_val:.2f})")
    
    # Check tokenization
    print("\n" + "="*80)
    print("TOKENIZATION CHECK")
    print("="*80)
    
    print("\nChecking if objects are single tokens:")
    unique_objects = set()
    for pair in pairs[:10]:
        unique_objects.add(pair['correct_object'])
        unique_objects.add(pair['incorrect_object'])
    
    for obj in sorted(unique_objects):
        # Try with and without space
        tok_with_space = model.to_tokens(" " + obj, prepend_bos=False)[0]
        tok_without_space = model.to_tokens(obj, prepend_bos=False)[0]
        
        print(f"\n  '{obj}':")
        print(f"    With space: {tok_with_space} (length: {len(tok_with_space)})")
        if len(tok_with_space) == 1:
            decoded = model.to_string(tok_with_space[0])
            print(f"      → Decodes to: '{decoded}'")
        else:
            print(f"      → NOT a single token!")
        
        print(f"    Without space: {tok_without_space} (length: {len(tok_without_space)})")
        if len(tok_without_space) == 1:
            decoded = model.to_string(tok_without_space[0])
            print(f"      → Decodes to: '{decoded}'")
    
    # Analyze what the model typically predicts
    print("\n" + "="*80)
    print("PREDICTION ANALYSIS")
    print("="*80)
    
    with torch.inference_mode():
        all_clean_logits = model(clean_tokens)[:, -1, :]
        all_clean_preds = all_clean_logits.argmax(dim=-1)
    
    print(f"\nMost common predictions on clean prompts:")
    pred_strings = [model.to_string(tok_id) for tok_id in all_clean_preds]
    from collections import Counter
    pred_counts = Counter(pred_strings)
    
    for pred, count in pred_counts.most_common(10):
        print(f"  '{pred}': {count} times ({count/len(pairs)*100:.1f}%)")
    
    # Check template analysis
    print("\n" + "="*80)
    print("TEMPLATE ANALYSIS")
    print("="*80)
    
    templates = {}
    for pair in pairs[:10]:
        # Extract template by removing color and object words
        template = pair['clean']
        for word in [pair['color1'], pair['color2'], pair['object1'], pair['object2']]:
            template = template.replace(word, "___")
        templates[template] = templates.get(template, 0) + 1
    
    print("\nTemplates used:")
    for template, count in templates.items():
        print(f"  {count}x: {template}")
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    print("\nPotential issues:")
    print("  1. The task may be too difficult for GPT-2 small")
    print("  2. The model may not naturally complete sentences this way")
    print("  3. Template design might need adjustment")
    print("  4. May need fine-tuning or different model")
    
    print("\nNext steps:")
    print("  1. Try simpler templates: 'The red ball. The' → should predict 'red'")
    print("  2. Try larger model (GPT-2 medium or large)")
    print("  3. Check if model has seen similar patterns in training")
    print("  4. Consider few-shot prompting instead of zero-shot")


if __name__ == "__main__":
    diagnose_task()
