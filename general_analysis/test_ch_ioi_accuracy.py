"""
Test script to evaluate CPM's accuracy on Chinese IOI task.

This script loads the Chinese IOI dataset and measures:
- Clean accuracy (when both names are in context)
- Corrupt accuracy (when referenced name is NOT in context)
- Performance drop due to corruption

Usage:
    python test_ch_ioi_accuracy.py --model TsinghuaAI/CPM-Generate
"""

import sys
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json

# Add parent directory to path for imports
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


def test_ch_ioi_accuracy(size: str = "small", n_examples: int = None, model_name: str = "TsinghuaAI/CPM-Generate"):
    """Test Chinese model's accuracy on Chinese IOI task.
    
    Recommended models:
    - TsinghuaAI/CPM-Generate (CPM-small, ~2.6B params)
    - Qwen/Qwen2.5-0.5B (Qwen small, 0.5B params)
    - Qwen/Qwen2.5-1.5B (Qwen medium, 1.5B params)
    """
    
    print("=" * 80)
    print("Chinese IOI Task - Accuracy Test with CPM")
    print("=" * 80)
    
    # Load model and tokenizer
    print(f"\n[1/4] Loading CPM model ({model_name})...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        model = model.to(device)
        model.eval()
        print(f"   ✓ Model loaded: {model_name}")
        print(f"   ✓ Device: {device}")
        print(f"   ✓ Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
        print(f"   ✓ Number of layers x heads: {model.config.num_hidden_layers} x {model.config.num_attention_heads}")
    except Exception as e:
        print(f"   ✗ Failed to load {model_name}")
        print(f"   Error: {e}")
        raise
    
    # Load dataset
    print(f"\n[2/4] Loading Chinese IOI dataset ({size})...")
    pairs = load_ch_ioi_dataset(size=size)
    if n_examples:
        pairs = pairs[:n_examples]
    
    print(f"   ✓ Loaded {len(pairs)} examples")
    
    # Tokenize dataset
    print(f"\n[3/4] Tokenizing dataset...")
    clean_texts = [pair['clean'] for pair in pairs]
    corrupt_texts = [pair['corrupt'] for pair in pairs]
    io_tokens = [pair['io_token'] for pair in pairs]
    print(f"   Example clean prompt: {clean_texts[0]}")
    print(f"   Example corrupt prompt: {corrupt_texts[0]}")
    print(f"   Example IO name to predict: {pairs[0]['io_name']}")
    
    # Debug tokenization
    print(f"\n   Tokenizer vocab size: {len(tokenizer)}")
    test_name = pairs[0]['io_name']
    test_encoding = tokenizer.encode(test_name, add_special_tokens=False)
    print(f"   Test name '{test_name}' tokenizes to: {test_encoding}")
    print(f"   Decoded back: '{tokenizer.decode(test_encoding)}'")
    
    # Tokenize with CPM tokenizer
    clean_encodings = tokenizer(clean_texts, return_tensors="pt", padding=True, truncation=True)
    corrupt_encodings = tokenizer(corrupt_texts, return_tensors="pt", padding=True, truncation=True)
    
    clean_tokens = clean_encodings['input_ids'].to(device)
    corrupt_tokens = corrupt_encodings['input_ids'].to(device)
    clean_attention = clean_encodings['attention_mask'].to(device)
    corrupt_attention = corrupt_encodings['attention_mask'].to(device)
    
    # Get IO token IDs
    io_token_ids = []
    for io_token in io_tokens:
        # Tokenize just the IO token
        io_encoding = tokenizer(io_token, add_special_tokens=False)
        if len(io_encoding['input_ids']) > 0:
            io_token_ids.append(io_encoding['input_ids'][0])  # Take first token
        else:
            io_token_ids.append(tokenizer.unk_token_id)  # Fallback
    io_toks = torch.tensor(io_token_ids, device=device)
    
    print(f"   ✓ Clean tokens shape:   {clean_tokens.shape}")
    print(f"   ✓ Corrupt tokens shape: {corrupt_tokens.shape}")
    
    # Analyze accuracy using generation instead of just logits
    print("\n[4/4] Analyzing model accuracy with generation...")
    
    clean_correct = 0
    corrupt_correct = 0
    
    with torch.no_grad():
        for i, pair in enumerate(pairs):
            if (i + 1) % 20 == 0:
                print(f"   Processing {i+1}/{len(pairs)}...")
            
            # Generate next tokens for clean
            clean_input = tokenizer(pair['clean'], return_tensors="pt").to(device)
            clean_gen = model.generate(
                **clean_input,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            clean_output = tokenizer.decode(clean_gen[0][len(clean_input['input_ids'][0]):], skip_special_tokens=True)
            
            # Generate for corrupt
            corrupt_input = tokenizer(pair['corrupt'], return_tensors="pt").to(device)
            corrupt_gen = model.generate(
                **corrupt_input,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            corrupt_output = tokenizer.decode(corrupt_gen[0][len(corrupt_input['input_ids'][0]):], skip_special_tokens=True)
            
            # Check if IO name appears in output
            io_name = pair['io_name']
            if io_name in clean_output:
                clean_correct += 1
            if io_name in corrupt_output:
                corrupt_correct += 1
    
    clean_accuracy = clean_correct / len(pairs)
    corrupt_accuracy = corrupt_correct / len(pairs)
    
    metrics = {
        'clean_accuracy': clean_accuracy,
        'corrupt_accuracy': corrupt_accuracy,
        'performance_drop': clean_accuracy - corrupt_accuracy,
    }
    
    # Display results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    
    print("\nAccuracy:")
    print(f"  Clean (IO in context):       {metrics['clean_accuracy']:.1%}")
    print(f"  Corrupt (IO NOT in context): {metrics['corrupt_accuracy']:.1%}")
    print(f"  Performance drop:            {metrics['performance_drop']:.1%}")
    
    # Show some examples
    print("\n" + "=" * 80)
    print("EXAMPLE PREDICTIONS")
    print("=" * 80)
    
    with torch.no_grad():
        for i in range(min(5, len(pairs))):
            print(f"\n--- Example {i+1} ---")
            print(f"Clean prompt:   {pairs[i]['clean']}")
            print(f"Expected:       {pairs[i]['io_name']}")
            
            # Generate for clean - try with more tokens and temperature
            clean_input = tokenizer(pairs[i]['clean'], return_tensors="pt").to(device)
            print(f"Input token IDs: {clean_input['input_ids'][0].tolist()}")
            
            clean_gen = model.generate(
                **clean_input,
                max_new_tokens=20,
                do_sample=False,
                num_beams=5,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            full_output = tokenizer.decode(clean_gen[0], skip_special_tokens=True)
            clean_output = tokenizer.decode(clean_gen[0][len(clean_input['input_ids'][0]):], skip_special_tokens=True)
            clean_correct = pairs[i]['io_name'] in clean_output
            
            print(f"Full text:       {full_output}")
            print(f"Clean generated: '{clean_output}' {'✓' if clean_correct else '✗'}")
            
            # Also show logits for next token
            with torch.no_grad():
                logits = model(**clean_input).logits[0, -1, :]
                top_k = torch.topk(logits, k=10)
                print(f"Top 10 next tokens by logit:")
                for idx, (token_id, score) in enumerate(zip(top_k.indices, top_k.values)):
                    token_text = tokenizer.decode([token_id.item()])
                    print(f"  {idx+1}. '{token_text}' (id={token_id.item()}, logit={score.item():.2f})")
            
            # Generate for corrupt
            print(f"\nCorrupt prompt: {pairs[i]['corrupt']}")
            corrupt_input = tokenizer(pairs[i]['corrupt'], return_tensors="pt").to(device)
            corrupt_gen = model.generate(
                **corrupt_input,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            corrupt_output = tokenizer.decode(corrupt_gen[0][len(corrupt_input['input_ids'][0]):], skip_special_tokens=True)
            corrupt_correct = pairs[i]['io_name'] in corrupt_output
            
            print(f"Corrupt generated: '{corrupt_output}' {'✓' if corrupt_correct else '✗'}")
    
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
    
    parser = argparse.ArgumentParser(description="Test Chinese IOI accuracy with CPM")
    parser.add_argument("--size", type=str, default="small",
                        choices=["small", "medium", "large"],
                        help="Dataset size to use")
    parser.add_argument("--n_examples", type=int, default=None,
                        help="Number of examples to use (default: all)")
    parser.add_argument("--model", type=str, default="TsinghuaAI/CPM-Generate",
                        help="Model name to use (default: TsinghuaAI/CPM-Generate)")
    
    args = parser.parse_args()
    
    try:
        metrics = test_ch_ioi_accuracy(size=args.size, n_examples=args.n_examples, model_name=args.model)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
