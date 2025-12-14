"""
Test script to evaluate GPT-2's accuracy on Chinese IOI task.

This script loads the Chinese IOI dataset and measures:
- Clean accuracy (when both names are in context)
- Corrupt accuracy (when referenced name is NOT in context)
- Performance drop due to corruption

Usage:
    python test_ch_ioi_accuracy.py
"""

import sys
from pathlib import Path
import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from data_loader_ch_ioi import (
    load_ch_ioi_dataset_for_patching,
    analyze_ch_ioi_accuracy
)
from transformer_lens import HookedTransformer


def test_ch_ioi_accuracy(size: str = "small", n_examples: int = None):
    """Test GPT-2's accuracy on Chinese IOI task."""
    
    print("=" * 80)
    print("Chinese IOI Task - Accuracy Test")
    print("=" * 80)
    
    # Load model
    print("\n[1/3] Loading GPT-2 model...")
    model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
    print(f"   ✓ Model loaded: {model.cfg.model_name}")
    
    # Load dataset
    print(f"\n[2/3] Loading Chinese IOI dataset ({size})...")
    clean_tokens, corrupt_tokens, io_toks, pairs = \
        load_ch_ioi_dataset_for_patching(model, size=size, n_examples=n_examples)
    
    print(f"   ✓ Loaded {len(pairs)} examples")
    print(f"   Clean tokens shape:   {clean_tokens.shape}")
    print(f"   Corrupt tokens shape: {corrupt_tokens.shape}")
    
    # Analyze accuracy
    print("\n[3/3] Analyzing model accuracy...")
    metrics = analyze_ch_ioi_accuracy(model, clean_tokens, corrupt_tokens, io_toks)
    
    # Display results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    
    print("\nAccuracy:")
    print(f"  Clean (IO in context):     {metrics['clean_accuracy']:.1%}")
    print(f"  Corrupt (IO NOT in context): {metrics['corrupt_accuracy']:.1%}")
    print(f"  Performance drop:          {metrics['performance_drop']:.1%}")
    
    print("\nAverage Logits for IO token:")
    print(f"  Clean:   {metrics['clean_avg_logit']:.3f}")
    print(f"  Corrupt: {metrics['corrupt_avg_logit']:.3f}")
    
    # Show some examples
    print("\n" + "=" * 80)
    print("EXAMPLE PREDICTIONS")
    print("=" * 80)
    
    with torch.inference_mode():
        clean_logits = model(clean_tokens[:5])[:, -1, :]
        corrupt_logits = model(corrupt_tokens[:5])[:, -1, :]
        
        for i in range(min(5, len(pairs))):
            print(f"\n--- Example {i+1} ---")
            print(f"Clean:   {pairs[i]['clean']}")
            print(f"Expected: {pairs[i]['io_token']}")
            
            # Get top predictions for clean
            clean_top5 = torch.topk(clean_logits[i], 5)
            clean_pred = model.to_string(clean_top5.indices[0])
            clean_correct = clean_top5.indices[0] == io_toks[i]
            
            print(f"Clean prediction: '{clean_pred}' {'✓' if clean_correct else '✗'}")
            
            # Show corrupt
            print(f"\nCorrupt: {pairs[i]['corrupt']}")
            corrupt_top5 = torch.topk(corrupt_logits[i], 5)
            corrupt_pred = model.to_string(corrupt_top5.indices[0])
            corrupt_correct = corrupt_top5.indices[0] == io_toks[i]
            
            print(f"Corrupt prediction: '{corrupt_pred}' {'✓' if corrupt_correct else '✗'}")
    
    print("\n" + "=" * 80)
    
    # Interpretation
    print("\nINTERPRETATION:")
    if metrics['clean_accuracy'] > 0.5:
        print(f"✓ Model performs reasonably on clean examples ({metrics['clean_accuracy']:.1%})")
    else:
        print(f"✗ Model struggles even on clean examples ({metrics['clean_accuracy']:.1%})")
    
    if metrics['performance_drop'] > 0.2:
        print(f"✓ Strong corruption effect ({metrics['performance_drop']:.1%} drop)")
        print("  → Model relies on contextual information")
    else:
        print(f"⚠ Weak corruption effect ({metrics['performance_drop']:.1%} drop)")
        print("  → Model may not be using context as expected")
    
    print("\n" + "=" * 80)
    
    return metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Chinese IOI accuracy")
    parser.add_argument("--size", type=str, default="small",
                        choices=["small", "medium", "large"],
                        help="Dataset size to use")
    parser.add_argument("--n_examples", type=int, default=None,
                        help="Number of examples to use (default: all)")
    
    args = parser.parse_args()
    
    try:
        metrics = test_ch_ioi_accuracy(size=args.size, n_examples=args.n_examples)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
