from transformer_lens import HookedTransformer
import torch

model = HookedTransformer.from_pretrained('gpt2-small')
tokens = torch.tensor([[1, 2, 3, 4, 5]])

# Print all hook names
print('Available hooks for attention in layer 0:')
for name, param in model.named_parameters():
    if 'blocks.0.attn' in name:
        print(f'  {name}')

print('\nHook points:')
logits, cache = model.run_with_cache(tokens)
for key in cache.keys():
    if 'blocks.0.attn' in key:
        print(f'  {key}')
