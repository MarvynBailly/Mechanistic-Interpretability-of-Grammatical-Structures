"""
Test script to evaluate Chinese GPT-2 models on Chinese IOI task using TransformerLens.

This ensures consistency with the rest of the mechanistic interpretability framework.
"""

import sys
from pathlib import Path
import torch
from transformer_lens import HookedTransformer
import json

sys.path.insert(0, str(Path(__file__).parent))


def load_ch_ioi_dataset(size="small"):
    """Load Chinese IOI dataset from JSON file."""
    current_dir = Path(__file__).parent
    base_dir = current_dir.parent / "data_generation" / "output"
    
    size_map = {
        "small": "chinese_ioi_pairs_small.json",
        "medium": "chinese_ioi_pairs_medium.json",
        "large": "chinese_ioi_pairs_large.json"
    }
    
    file_path = base_dir / size_map[size]
    
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data['pairs']


def test_ch_ioi_with_transformerlens(
    model_name: str = "uer/gpt2-chinese-cluecorpussmall",
    size: str = "small",
    n_examples: int = None
):
    """
    Test Chinese GPT-2 model on IOI task using TransformerLens.
    
    Available Chinese GPT-2 models:
    - uer/gpt2-chinese-cluecorpussmall (GPT-2 small, trained on Chinese)
    - uer/gpt2-chinese-poem (GPT-2, trained on Chinese poetry)
    - TsinghuaAI/CPM-Generate (if supported by TransformerLens)
    """
    
    print("=" * 80)
    print("Chinese IOI Task - TransformerLens Analysis")
    print("=" * 80)
    
    # Load model through TransformerLens
    print(f"\n[1/3] Loading model via TransformerLens: {model_name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        # Try loading as HuggingFace model
        model = HookedTransformer.from_pretrained(
            model_name,
            device=device,
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False,
        )
        print(f"   ✓ Model loaded: {model.cfg.model_name}")
        print(f"   ✓ Device: {device}")
        print(f"   ✓ Architecture: {model.cfg.n_layers} layers × {model.cfg.n_heads} heads")
        print(f"   ✓ d_model: {model.cfg.d_model}")
        print(f"   ✓ Vocab size: {model.cfg.d_vocab}")
    except Exception as e:
        print(f"   ✗ Failed to load {model_name} through TransformerLens")
        print(f"   Error: {e}")
        print("\n   NOTE: TransformerLens may not support this model directly.")
        print("   You may need to use the HuggingFace transformers library instead.")
        raise
    
    # Load dataset
    print(f"\n[2/3] Loading Chinese IOI dataset ({size})...")
    pairs = load_ch_ioi_dataset(size=size)
    if n_examples:
        pairs = pairs[:n_examples]
    print(f"   ✓ Loaded {len(pairs)} examples")
    
    # Tokenize and test
    print(f"\n[3/3] Testing model accuracy...")
    
    clean_texts = [pair['clean'] for pair in pairs]
    io_names = [pair['io_name'] for pair in pairs]
    
    # Debug first example
    print(f"\n   Debug - First example:")
    print(f"   Prompt: {clean_texts[0]}")
    print(f"   Expected: {io_names[0]}")
    
    # Tokenize first example
    tokens = model.to_tokens(clean_texts[0])
    print(f"   Tokens: {tokens[0].tolist()}")
    print(f"   Decoded: {model.to_string(tokens[0])}")
    
    # Get model prediction
    with torch.inference_mode():
        logits = model(tokens)[:, -1, :]
        
        # Get top 10 predictions
        top_k = torch.topk(logits[0], k=10)
        print(f"\n   Top 10 predictions:")
        for i, (token_id, score) in enumerate(zip(top_k.indices, top_k.values)):
            token_str = model.to_string(token_id)
            print(f"   {i+1}. '{token_str}' (id={token_id.item()}, logit={score.item():.2f})")
        
        # Check if IO name tokens appear in top predictions
        io_tokens = model.to_tokens(io_names[0], prepend_bos=False)[0]
        print(f"\n   Expected name '{io_names[0]}' tokenizes to: {io_tokens.tolist()}")
        for token_id in io_tokens:
            print(f"   Token '{model.to_string(token_id)}' (id={token_id.item()})")
    
    # Batch analysis
    print(f"\n   Running full batch analysis on {len(pairs)} examples...")
    
    clean_correct = 0
    
    for i, pair in enumerate(pairs):
        if (i + 1) % 10 == 0:
            print(f"   Processing {i+1}/{len(pairs)}...")
        
        # Tokenize
        tokens = model.to_tokens(pair['clean'])
        
        with torch.inference_mode():
            logits = model(tokens)[:, -1, :]
            pred_token = logits.argmax(dim=-1)
            pred_str = model.to_string(pred_token)
            
            # Check if IO name appears in prediction
            if pair['io_name'] in pred_str:
                clean_correct += 1
    
    accuracy = clean_correct / len(pairs)
    
    # Results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"\nAccuracy: {accuracy:.1%} ({clean_correct}/{len(pairs)})")
    
    if accuracy < 0.1:
        print("\n⚠️  Very low accuracy detected!")
        print("   Possible issues:")
        print("   1. Model may not be trained on this type of narrative task")
        print("   2. Tokenization may split Chinese names across multiple tokens")
        print("   3. Model may need specific prompt formatting")
        print("   4. Chinese IOI task may be fundamentally different from English IOI")
    elif accuracy < 0.5:
        print("\n⚠️  Moderate performance - task is partially solvable")
    else:
        print("\n✓ Good performance - model can solve this task!")
    
    return {
        'accuracy': accuracy,
        'correct': clean_correct,
        'total': len(pairs),
        'model_name': model_name
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Chinese GPT-2 on IOI task")
    parser.add_argument("--model", type=str, default="uer/gpt2-chinese-cluecorpussmall",
                        help="Model name (must be compatible with TransformerLens)")
    parser.add_argument("--size", type=str, default="small",
                        choices=["small", "medium", "large"],
                        help="Dataset size")
    parser.add_argument("--n_examples", type=int, default=None,
                        help="Number of examples (default: all)")
    
    args = parser.parse_args()
    
    try:
        metrics = test_ch_ioi_with_transformerlens(
            model_name=args.model,
            size=args.size,
            n_examples=args.n_examples
        )
        print(f"\n✓ Final accuracy: {metrics['accuracy']:.1%}")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
