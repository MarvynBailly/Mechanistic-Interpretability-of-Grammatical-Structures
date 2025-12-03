"""
Quick test of code completion task performance.
"""

import torch
from transformer_lens import HookedTransformer
from data_loader_code import load_code_dataset_for_patching, analyze_code_accuracy
from utils import set_seed

def test_code_task():
    """Test code completion task with all variable types."""
    
    print("="*80)
    print("CODE COMPLETION TASK - PERFORMANCE TEST")
    print("="*80)
    
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\nLoading model...")
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    
    # Test all three variable types
    for var_type in ["letters", "colors", "objects"]:
        print(f"\n" + "="*80)
        print(f"TESTING: {var_type.upper()}")
        print("="*80)
        
        # Load dataset
        clean_tokens, corrupt_tokens, correct_toks, incorrect_toks, pairs = \
            load_code_dataset_for_patching(model, var_type=var_type, n_examples=100)
        
        # Analyze performance
        stats = analyze_code_accuracy(
            model, clean_tokens, corrupt_tokens, correct_toks, incorrect_toks
        )
        
        print(f"\n📊 PERFORMANCE:")
        print(f"  Clean logit diff:  {stats['clean_logit_diff']:>7.3f}")
        print(f"  Clean accuracy:    {stats['clean_accuracy']*100:>6.1f}%")
        print(f"  Corrupt logit diff:{stats['corrupt_logit_diff']:>7.3f}")
        print(f"  Corrupt accuracy:  {stats['corrupt_accuracy']*100:>6.1f}%")
        print(f"  Corruption effect: {stats['corruption_effect']:>7.3f}")
        
        # Show a few examples
        print(f"\n📝 EXAMPLE PREDICTIONS:")
        with torch.inference_mode():
            clean_logits = model(clean_tokens[:3])[:, -1, :]
        
        for i in range(3):
            pair = pairs[i]
            top_5 = torch.topk(clean_logits[i], k=5)
            
            print(f"\n  Example {i+1}:")
            print(f"    Prompt: {pair['clean']}")
            print(f"    Expected: ' {pair['correct_arg']}'")
            print(f"    Top 5:")
            for j, (tok_id, logit_val) in enumerate(zip(top_5.indices, top_5.values), 1):
                token_str = model.to_string(tok_id)
                marker = "✓" if token_str.strip() == pair['correct_arg'] else ""
                print(f"      {j}. '{token_str}' (logit: {logit_val:.2f}) {marker}")
        
        # Verdict
        if stats['clean_accuracy'] > 0.7:
            print(f"\n✅ EXCELLENT! {var_type.upper()} task is highly solvable ({stats['clean_accuracy']*100:.0f}% accuracy)")
        elif stats['clean_accuracy'] > 0.4:
            print(f"\n⚠️  MODERATE: {var_type.upper()} task is partially solvable ({stats['clean_accuracy']*100:.0f}% accuracy)")
        else:
            print(f"\n❌ POOR: {var_type.upper()} task is not well-solved ({stats['clean_accuracy']*100:.0f}% accuracy)")

if __name__ == "__main__":
    test_code_task()
