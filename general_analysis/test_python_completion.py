"""
Quick test: Can GPT-2 complete Python function arguments?

Test if GPT-2 can solve: "return arg1 + " → should predict "arg2"
This is a much simpler task that's likely in the training distribution.
"""

import torch
from transformer_lens import HookedTransformer

def test_python_completion():
    """Test if GPT-2 can complete Python argument references."""
    
    print("="*80)
    print("TESTING: Python Argument Completion")
    print("="*80)
    
    # Load model
    print("\nLoading GPT-2...")
    model = HookedTransformer.from_pretrained("gpt2-small", device="cuda" if torch.cuda.is_available() else "cpu")
    
    # Test cases
    test_cases = [
        # Simple addition
        "def add(arg1, arg2):\n    return arg1 +",
        "def add(x, y):\n    return x +",
        "def multiply(a, b):\n    return a *",
        
        # More complex
        "def concat(str1, str2):\n    return str1 +",
        "def calculate(num1, num2):\n    result = num1 +",
        
        # With different names
        "def process(red, blue):\n    return red +",
        "def combine(ball, cube):\n    return ball +",
    ]
    
    print("\n" + "="*80)
    print("TEST RESULTS")
    print("="*80)
    
    for i, prompt in enumerate(test_cases, 1):
        print(f"\n--- Test {i} ---")
        print(f"Prompt:\n{prompt}")
        
        # Tokenize and predict
        tokens = model.to_tokens(prompt, prepend_bos=True)
        
        with torch.inference_mode():
            logits = model(tokens)[0, -1, :]  # Last token logits
            top_k = torch.topk(logits, k=10)
        
        print(f"\nTop 10 predictions:")
        for j, (tok_id, logit_val) in enumerate(zip(top_k.indices, top_k.values), 1):
            token_str = model.to_string(tok_id)
            print(f"  {j}. '{token_str}' (logit: {logit_val:.2f})")
        
        # Check if second argument appears in top 5
        # Extract second argument from prompt
        args_line = [line for line in prompt.split('\n') if 'def ' in line][0]
        args = args_line.split('(')[1].split(')')[0].split(',')
        if len(args) >= 2:
            second_arg = args[1].strip()
            top_5_tokens = [model.to_string(tok_id) for tok_id in top_k.indices[:5]]
            
            # Check various forms of the second argument
            found = False
            for tok in top_5_tokens:
                if second_arg in tok or tok.strip() == second_arg:
                    found = True
                    break
            
            if found:
                print(f"  ✅ Second argument '{second_arg}' found in top 5!")
            else:
                print(f"  ❌ Second argument '{second_arg}' NOT in top 5")

if __name__ == "__main__":
    test_python_completion()
