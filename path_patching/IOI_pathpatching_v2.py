"""
IOI (Indirect Object Identification) — Starter Kit + Path Patching

What this does
--------------
1) Builds a small synthetic IOI dataset like:
   "When Mary and John went to the store, John gave a drink to"
   where the next-token should be " Mary" (the IO).

2) Evaluates a pretrained GPT-2 small on:
   - accuracy: P(IO) > P(S)
   - zero-rank rate: IO is the top-1 next token
   - mean logit diff: E[logit(IO) - logit(S)]

3) Runs a simple *path patching* experiment:
   - Construct a clean / corrupt IOI pair.
   - Run model with cache on both.
   - For each layer & position, patch the *residual stream* from clean into corrupt.
   - Measure how much that patch increases logit(IO) - logit(S) on the corrupt input.
   - Save a heatmap of "patching effect" across layers & positions.

Requirements
------------
pip install transformer-lens torch transformers einops matplotlib

Notes
-----
- We restrict names to single-token entries for GPT-2's tokenizer.
- Prompts end *right before* the IO token, so we evaluate the next-token logits.
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name

# ------------- Config -------------

SEED = 42
MODEL_NAME = "gpt2-small"  # GPT-2 small used in the IOI paper
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

# N_EXAMPLES = 2000  # for evaluation; you can increase this later
N_EXAMPLES = 2000          # full dataset for metrics
N_HEATMAP_EXAMPLES = 200   # subset for attention heatmaps / cache

# ------------- Names --------------
# Single-token names (for GPT-2 tokenizer). Must include leading space.
ONE_TOKEN_NAMES = [" John"," Mary"," James"," Susan"," Robert"," Linda"]


# ------------- Templates ----------
# These are pIOI-style templates (no {PLACE}/{OBJECT}, but fixed "store"/"drink").
# [IO] is the first name, [S] is the second name and also the subject of "gave".
# The correct next token after "to" is always the IO name (non-repeated name).
TEMPLATES = [
    "{IO} and{S} went to the store,{S} gave a drink to",
    "After {IO} and{S} went to the store,{S} gave a drink to",
    "Then {IO} and{S} went to the store,{S} gave a drink to",
    "While {IO} and{S} were working at the store,{S} gave a drink to",
    "When {IO} and{S} visited the store,{S} gave a drink to",
    "After {IO} and{S} found a drink at the store,{S} gave it to",
]



@dataclass
class IOIExample:
    prompt: str     # up to the "to" — the next token should be IO
    s_name: str
    io_name: str


# ------------- Utils -------------

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sample_name_pair(names: List[str]) -> Tuple[str, str]:
    """Return (S, IO) as two different names."""
    s, io = random.sample(names, 2)
    return s, io


def build_dataset(n: int) -> List[IOIExample]:
    data: List[IOIExample] = []
    for _ in range(n):
        s, io = sample_name_pair(ONE_TOKEN_NAMES)
        template = random.choice(TEMPLATES)
        # IO is first name in template, S is second
        prompt = template.format(S=s, IO=io)
        data.append(IOIExample(prompt=prompt, s_name=s, io_name=io))
    return data


# def get_logits_for_next_token(model: HookedTransformer, texts: List[str]) -> torch.Tensor:
#     """
#     Returns final logits for the next token after each text.
#     Shape: [batch, vocab]
#     """
#     tokens = model.to_tokens(texts, prepend_bos=False).to(model.cfg.device)
#     logits = model(tokens)               # [batch, seq, vocab]
#     next_logits = logits[:, -1, :]       # [batch, vocab]
#     return next_logits


# def evaluate(model: HookedTransformer, dataset: List[IOIExample]) -> Dict[str, float]:
#     # Batch everything for speed
#     prompts = [ex.prompt for ex in dataset]
#     next_logits = get_logits_for_next_token(model, prompts)  # [B, V]

#     # Convert names to token ids per example
#     io_ids = torch.tensor(
#         [model.to_single_token(ex.io_name) for ex in dataset],
#         device=next_logits.device
#     )
#     s_ids = torch.tensor(
#         [model.to_single_token(ex.s_name) for ex in dataset],
#         device=next_logits.device
#     )

#     # Sanity check: all names should be single-token
#     assert io_ids.min().item() >= 0
#     assert s_ids.min().item() >= 0

#     batch_index = torch.arange(len(dataset), device=next_logits.device)
#     io_logits = next_logits[batch_index, io_ids]
#     s_logits = next_logits[batch_index, s_ids]

#     # Metrics
#     acc = (io_logits > s_logits).float().mean().item()
#     zero_rank = (next_logits.argmax(dim=-1) == io_ids).float().mean().item()
#     logit_diff = (io_logits - s_logits).mean().item()
#     s_beats_io = (s_logits > io_logits).float().mean().item()

#     print(f"Mean logit diff (IO - S): {logit_diff:.3f}")
#     print(f"Proportion S logit > IO : {s_beats_io:.3f}")

#     return {
#         "accuracy": acc,
#         "zero_rank": zero_rank,
#         "logit_diff": logit_diff,
#         "s_beats_io": s_beats_io,
#     }
def get_logits_for_next_token(
    model: HookedTransformer,
    texts: List[str],
    batch_size: int = 64,
) -> torch.Tensor:
    """
    Returns final logits for the next token after each text.
    Shape: [batch, vocab]

    We do this in batches to avoid GPU/MPS OOM.
    """
    all_logits = []

    model_device = model.cfg.device

    with torch.inference_mode():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            tokens = model.to_tokens(batch_texts, prepend_bos=False).to(model_device)
            logits = model(tokens)          # [b, seq, vocab]
            next_logits = logits[:, -1, :]  # [b, vocab]
            all_logits.append(next_logits)

    return torch.cat(all_logits, dim=0)     # [B, vocab]


def evaluate(
    model: HookedTransformer,
    dataset: List[IOIExample],
    batch_size: int = 64,
) -> Dict[str, float]:
    """
    Batched evaluation to avoid OOM.
    """
    prompts = [ex.prompt for ex in dataset]
    next_logits = get_logits_for_next_token(model, prompts, batch_size=batch_size)  # [B, V]

    # Convert names to token ids per example
    io_ids = torch.tensor(
        [model.to_single_token(ex.io_name) for ex in dataset],
        device=next_logits.device,
    )
    s_ids = torch.tensor(
        [model.to_single_token(ex.s_name) for ex in dataset],
        device=next_logits.device,
    )

    # Sanity check: all names should be single-token
    assert io_ids.min().item() >= 0
    assert s_ids.min().item() >= 0

    batch_index = torch.arange(len(dataset), device=next_logits.device)
    io_logits = next_logits[batch_index, io_ids]
    s_logits = next_logits[batch_index, s_ids]

    acc = (io_logits > s_logits).float().mean().item()
    zero_rank = (next_logits.argmax(dim=-1) == io_ids).float().mean().item()
    logit_diff = (io_logits - s_logits).mean().item()
    s_beats_io = (s_logits > io_logits).float().mean().item()

    print(f"Mean logit diff (IO - S): {logit_diff:.3f}")
    print(f"Proportion S logit > IO : {s_beats_io:.3f}")

    return {
        "accuracy": acc,
        "zero_rank": zero_rank,
        "logit_diff": logit_diff,
        "s_beats_io": s_beats_io,
    }


# def logit_diff_io_s_from_tokens(
#     model: HookedTransformer,
#     tokens: torch.Tensor,
#     s_tok: int,
#     io_tok: int,
# ) -> float:
#     """
#     Compute mean logit(IO) - logit(S) at END position for a batch of tokens.
#     tokens: [B, seq]
#     """
#     logits = model(tokens)        # [B, seq, vocab]
#     next_logits = logits[:, -1, :]  # [B, vocab]
#     io_logits = next_logits[:, io_tok]
#     s_logits = next_logits[:, s_tok]
#     return (io_logits - s_logits).mean().item()

def logit_diff_io_s_from_tokens(
    model: HookedTransformer,
    tokens: torch.Tensor,
    s_toks: torch.Tensor,
    io_toks: torch.Tensor,
) -> float:
    """
    Compute mean logit(IO) - logit(S) at END position for a batch of tokens.
    
    tokens : [B, seq]
    s_toks : [B]   token ids for S for each example
    io_toks: [B]   token ids for IO for each example
    """
    logits = model(tokens)            # [B, seq, vocab]
    next_logits = logits[:, -1, :]    # [B, vocab]

    batch_idx = torch.arange(tokens.size(0), device=tokens.device)

    io_logits = next_logits[batch_idx, io_toks]   # shape [B]
    s_logits  = next_logits[batch_idx, s_toks]    # shape [B]

    return (io_logits - s_logits).mean().item()



# ------------- Path patching helpers -------------

def make_ioi_pair(model: HookedTransformer) -> Tuple[Dict, Dict]:
    """
    Build a (clean, corrupt) pair of IOI prompts.

    clean  : "When IO and S went to the store, S gave a drink to"
    corrupt: "When S and IO went to the store, S gave a drink to"

    Both should predict IO as the next token, but the internal structure
    is different. We'll see how patching activations from clean into corrupt
    affects logit(IO) - logit(S).
    """
    s_name, io_name = sample_name_pair(ONE_TOKEN_NAMES)

    clean_prompt = f"When{io_name} and{s_name} went to the store,{s_name} gave a drink to"
    corrupt_prompt = f"When{s_name} and{io_name} went to the store,{s_name} gave a drink to"

    clean_tokens = model.to_tokens(clean_prompt, prepend_bos=False).to(model.cfg.device)
    corrupt_tokens = model.to_tokens(corrupt_prompt, prepend_bos=False).to(model.cfg.device)

    io_tok = model.to_single_token(io_name)
    s_tok = model.to_single_token(s_name)
    assert io_tok is not None and s_tok is not None, "Names must be single-token."

    clean = {
        "prompt": clean_prompt,
        "tokens": clean_tokens,
        "io_tok": io_tok,
        "s_tok": s_tok,
    }
    corrupt = {
        "prompt": corrupt_prompt,
        "tokens": corrupt_tokens,
        "io_tok": io_tok,
        "s_tok": s_tok,
    }
    return clean, corrupt


def get_clean_corrupt_caches(
    model: HookedTransformer,
    clean: Dict,
    corrupt: Dict,
):
    _, clean_cache = model.run_with_cache(clean["tokens"])
    _, corrupt_cache = model.run_with_cache(corrupt["tokens"])
    return clean_cache, corrupt_cache


def patch_resid_pre_single(
    model: HookedTransformer,
    corrupt_tokens: torch.Tensor,
    clean_cache,
    layer: int,
    pos: int,
    s_tok: int,
    io_tok: int,
) -> float:
    """
    Run model on corrupt_tokens, but at resid_pre[layer, pos] replace activations
    with those from the clean run. Return logit(IO) - logit(S) after patch.
    """
    act_name = get_act_name("resid_pre", layer)

    def hook(value, hook):
        # value: [batch, seq, d_model] from corrupt run
        clean_value = clean_cache[act_name]  # [batch, seq, d_model] from clean run
        value[:, pos, :] = clean_value[:, pos, :]
        return value

    logits = model.run_with_hooks(
        corrupt_tokens,
        fwd_hooks=[(act_name, hook)]
    )
    next_logits = logits[:, -1, :]
    io_logits = next_logits[:, io_tok]
    s_logits = next_logits[:, s_tok]
    return (io_logits - s_logits).mean().item()


def scan_residual_patching(model: HookedTransformer):
    """
    Perform residual stream path patching over layers & positions for a single IOI pair.
    Returns:
      effects: [L, seq_len] tensor where each entry is:
               patched_logit_diff - corrupt_logit_diff
    """
    clean, corrupt = make_ioi_pair(model)
    clean_cache, corrupt_cache = get_clean_corrupt_caches(model, clean, corrupt)

    # clean_ld = logit_diff_io_s_from_tokens(
    #     model, clean["tokens"], clean["s_tok"], clean["io_tok"]
    # )
    clean_ld = logit_diff_io_s_from_tokens(
    model,
    clean["tokens"],                          # shape [1, seq]
    torch.tensor([clean["s_tok"]], device=model.cfg.device),
    torch.tensor([clean["io_tok"]], device=model.cfg.device),
    )

    corrupt_ld = logit_diff_io_s_from_tokens(
    model,
    corrupt["tokens"], 
    torch.tensor([corrupt["s_tok"]], device=model.cfg.device),
    torch.tensor([corrupt["io_tok"]], device=model.cfg.device),
    )

    print("\n--- Path patching setup ---")
    print(f"Clean prompt  : {clean['prompt']}")
    print(f"Corrupt prompt: {corrupt['prompt']}")
    print(f"Clean logit diff  (IO-S): {clean_ld:.3f}")
    print(f"Corrupt logit diff(IO-S): {corrupt_ld:.3f}")

    L = model.cfg.n_layers
    seq_len = corrupt["tokens"].shape[1]

    effects = torch.zeros(L, seq_len, device=model.cfg.device)

    print("\nScanning residual stream patches (layers x positions)...")
    for layer in range(L):
        for pos in range(seq_len):
            patched_ld = patch_resid_pre_single(
                model,
                corrupt["tokens"],
                clean_cache,
                layer,
                pos,
                corrupt["s_tok"],
                corrupt["io_tok"],
            )
            # How much does patching this node restore IO preference?
            effects[layer, pos] = patched_ld - corrupt_ld

    return effects, clean, corrupt


# ------------- Main -------------

# def main():
#     set_seed(SEED)
#     print("Loading model:", MODEL_NAME, "on", DEVICE)
#     model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE)
#     model.eval()

#     # Verify names are single-token for this tokenizer
#     bad = [n for n in ONE_TOKEN_NAMES if model.to_single_token(n) is None]
#     if bad:
#         raise ValueError(f"The following names are not single-token for {MODEL_NAME}: {bad}")

#     print("\nBuilding IOI dataset...")
#     dataset = build_dataset(N_EXAMPLES)
#     print(f"Dataset size: {len(dataset)}. Example prompt:\n{dataset[0].prompt}")

#     print("\nEvaluating GPT-2 small on IOI...")
#     metrics = evaluate(model, dataset)
#     print(f"IOI accuracy     : {metrics['accuracy'] * 100:.2f}%")
#     print(f"IOI zero-rank    : {metrics['zero_rank'] * 100:.2f}%")

#     # ---------- Residual stream path patching ----------
#     effects, clean_pair, corrupt_pair = scan_residual_patching(model)

#     # Move to CPU for plotting
#     effects_cpu = effects.detach().cpu().numpy()

#     plt.figure(figsize=(10, 6))
#     plt.imshow(effects_cpu, aspect="auto")
#     plt.xlabel("Token position")
#     plt.ylabel("Layer")
#     plt.title("Residual patching effect\nΔ logit_diff (patched - corrupt)")
#     plt.colorbar(label="Δ logit_diff")
#     plt.tight_layout()
#     out_path = "resid_patching_heatmap.png"
#     plt.savefig(out_path, dpi=160)
#     plt.close()
#     print(f"\nSaved residual patching heatmap to: {out_path}")
def get_name_positions(tokens: torch.Tensor,
                       s_ids: torch.Tensor,
                       io_ids: torch.Tensor):
    """
    tokens: [B, seq]
    s_ids, io_ids: [B] token ids for S and IO for each example.
    Returns:
      io_pos: [B] index of IO token
      s1_pos: [B] index of first S token
      s2_pos: [B] index of second S token (or last occurrence if more)

    Assumes IO appears once, S appears at least twice.
    """
    B, seq_len = tokens.shape
    device = tokens.device

    io_pos = torch.zeros(B, dtype=torch.long, device=device)
    s1_pos = torch.zeros(B, dtype=torch.long, device=device)
    s2_pos = torch.zeros(B, dtype=torch.long, device=device)

    for b in range(B):
        row = tokens[b]

        # IO position (should be unique)
        io_matches = (row == io_ids[b]).nonzero(as_tuple=True)[0]
        assert len(io_matches) >= 1, f"No IO token found in example {b}"
        io_pos[b] = io_matches[0]

        # All S positions (we expect two: S1 and S2)
        s_matches = (row == s_ids[b]).nonzero(as_tuple=True)[0]
        assert len(s_matches) >= 2, f"Expected at least two S tokens in example {b}"
        s1_pos[b] = s_matches[0]
        s2_pos[b] = s_matches[-1]

    return io_pos, s1_pos, s2_pos

def attn_pattern_key(cache, layer: int):
    for key in ["pattern", "attn", "attn_probs"]:
        name = get_act_name(key, layer)
        if name in cache:
            return name
    candidates = [k for k in cache.keys() if "attn" in k or "pattern" in k]
    raise KeyError(
        f"No attention-prob key found for layer {layer}. Cached keys: {candidates}"
    )


def main():
    set_seed(SEED)
    print("Loading model:", MODEL_NAME, "on", DEVICE)
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE)
    model.eval()

    # ---- Check name tokenization ----
    bad = [n for n in ONE_TOKEN_NAMES if model.to_single_token(n) is None]
    if bad:
        raise ValueError(
            f"The following names are not single-token for {MODEL_NAME}: {bad}"
        )

    # ---- Build IOI dataset ----
    print("\nBuilding IOI dataset...")
    dataset = build_dataset(N_EXAMPLES)
    print(f"Dataset size: {len(dataset)}. Example prompt:\n{dataset[0].prompt}")

    # ---- Evaluate GPT-2 small on IOI ----
    print("\nEvaluating GPT-2 small on IOI...")
    metrics = evaluate(model, dataset)
    print(f"IOI accuracy     : {metrics['accuracy'] * 100:.2f}%")
    print(f"IOI zero-rank    : {metrics['zero_rank'] * 100:.2f}%")
    print(f"Mean logit diff  : {metrics['logit_diff']:.3f}")
    print(f"Prop S logit > IO: {metrics['s_beats_io']:.3f}")

    # # ---- Attention maps: END -> IO / END -> S2 ----
    # print("\nComputing attention heatmaps (END -> IO / S2)...")
    # prompts = [ex.prompt for ex in dataset]
    # tokens = model.to_tokens(prompts, prepend_bos=False).to(model.cfg.device)
    # _, cache = model.run_with_cache(tokens)  # we need all attention patterns
        # ---- Attention maps: END -> IO / END -> S2 ----
    print("\nComputing attention heatmaps (END -> IO / S2)...")

    # Use only a subset of the dataset to keep run_with_cache memory small
    n_heat = min(N_HEATMAP_EXAMPLES, len(dataset))
    heat_examples = random.sample(dataset, n_heat)

    heat_prompts = [ex.prompt for ex in heat_examples]
    tokens = model.to_tokens(heat_prompts, prepend_bos=False).to(model.cfg.device)

    with torch.inference_mode():
        _, cache = model.run_with_cache(tokens)  # we need all attention patterns

    # Per-example token IDs for S and IO (for the subset only)
    s_ids = torch.tensor(
        [model.to_single_token(ex.s_name) for ex in heat_examples],
        device=tokens.device,
    )
    io_ids = torch.tensor(
        [model.to_single_token(ex.io_name) for ex in heat_examples],
        device=tokens.device,
    )

    # Get positions of IO, S1, S2 in each sequence
    io_pos, s1_pos, s2_pos = get_name_positions(tokens, s_ids, io_ids)

    L, H = model.cfg.n_layers, model.cfg.n_heads
    B, seq_len = tokens.shape

    # For each (layer, head) we’ll store:
    #   avg attention from END -> IO
    #   avg attention from END -> S2
    attn_io = torch.zeros(L, H, device=model.cfg.device)
    attn_s2 = torch.zeros(L, H, device=model.cfg.device)

    for l in range(L):
        probs = cache[attn_pattern_key(cache, l)]  # [B, H, dest, src]
        end_idx = -1  # END = last token position

        io_sum = torch.zeros(H, device=model.cfg.device)
        s2_sum = torch.zeros(H, device=model.cfg.device)

        # Average over batch: for each example, take attn from END to IO/S2
        for b in range(B):
            io_sum += probs[b, :, end_idx, io_pos[b]]
            s2_sum += probs[b, :, end_idx, s2_pos[b]]

        attn_io[l] = io_sum / B
        attn_s2[l] = s2_sum / B

    attn_diff = attn_io - attn_s2  # IO preference over S2


    # # Per-example token IDs for S and IO
    # s_ids = torch.tensor(
    #     [model.to_single_token(ex.s_name) for ex in dataset],
    #     device=tokens.device,
    # )
    # io_ids = torch.tensor(
    #     [model.to_single_token(ex.io_name) for ex in dataset],
    #     device=tokens.device,
    # )

    # # Get positions of IO, S1, S2 in each sequence
    # io_pos, s1_pos, s2_pos = get_name_positions(tokens, s_ids, io_ids)

    # L, H = model.cfg.n_layers, model.cfg.n_heads
    # B, seq_len = tokens.shape

    # # For each (layer, head) we’ll store:
    # #   avg attention from END -> IO
    # #   avg attention from END -> S2
    # attn_io = torch.zeros(L, H, device=model.cfg.device)
    # attn_s2 = torch.zeros(L, H, device=model.cfg.device)

    # for l in range(L):
    #     probs = cache[attn_pattern_key(cache, l)]  # [B, H, dest, src]
    #     end_idx = -1  # END = last token position

    #     io_sum = torch.zeros(H, device=model.cfg.device)
    #     s2_sum = torch.zeros(H, device=model.cfg.device)

    #     # Average over batch: for each example, take attn from END to IO/S2
    #     for b in range(B):
    #         io_sum += probs[b, :, end_idx, io_pos[b]]
    #         s2_sum += probs[b, :, end_idx, s2_pos[b]]

    #     attn_io[l] = io_sum / B
    #     attn_s2[l] = s2_sum / B

    # attn_diff = attn_io - attn_s2  # IO preference over S2

    # ---- Plot: END -> IO ----
    plt.figure(figsize=(10, 6))
    plt.imshow(attn_io.detach().cpu(), aspect="auto")
    plt.xlabel("Head")
    plt.ylabel("Layer")
    plt.title("Avg attention from END to IO")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("attn_end_to_io.png", dpi=160)
    plt.close()
    print("Saved END->IO attention heatmap to: attn_end_to_io.png")

    # ---- Plot: END -> S2 ----
    plt.figure(figsize=(10, 6))
    plt.imshow(attn_s2.detach().cpu(), aspect="auto")
    plt.xlabel("Head")
    plt.ylabel("Layer")
    plt.title("Avg attention from END to S2")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("attn_end_to_s2.png", dpi=160)
    plt.close()
    print("Saved END->S2 attention heatmap to: attn_end_to_s2.png")

    # ---- Plot: END -> IO minus END -> S2 (Name Movers stand out) ----
    plt.figure(figsize=(10, 6))
    plt.imshow(attn_diff.detach().cpu(), aspect="auto")
    plt.xlabel("Head")
    plt.ylabel("Layer")
    plt.title("Attention difference (END -> IO minus END -> S2)")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("attn_end_io_minus_s2.png", dpi=160)
    plt.close()
    print("Saved END->IO-S2 diff heatmap to: attn_end_io_minus_s2.png")

    # ---- Optional: residual path patching (if you kept scan_residual_patching) ----
    try:
        print("\nRunning residual path patching scan on a single IOI pair...")
        effects, clean_pair, corrupt_pair = scan_residual_patching(model)
        effects_cpu = effects.detach().cpu().numpy()

        plt.figure(figsize=(10, 6))
        plt.imshow(effects_cpu, aspect="auto")
        plt.xlabel("Token position")
        plt.ylabel("Layer")
        plt.title("Residual patching effect\nΔ logit_diff (patched - corrupt)")
        plt.colorbar(label="Δ logit_diff")
        plt.tight_layout()
        plt.savefig("resid_patching_heatmap.png", dpi=160)
        plt.close()
        print("Saved residual patching heatmap to: resid_patching_heatmap.png")
    except NameError:
        print("\nscan_residual_patching not defined – skipping path patching.")


if __name__ == "__main__":
    main()
