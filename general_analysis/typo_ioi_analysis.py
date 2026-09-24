"""
Typo ("blurry search") IOI analysis.

Runs the IOI path patching analysis on sentences where one mention of the
subject is misspelled, and compares every typo condition against the exact
(correctly spelled) baseline:

1. Behaviour: logit diff (IO - S), P(IO > S), top-1 accuracy
2. Head -> logits path patching effects (clean head output patched into corrupt)
3. Attention signatures of the known IOI head classes (Wang et al. 2022):
     duplicate token heads: S2 -> S1 attention
     induction heads:       S2 -> token after S1
     S-inhibition heads:    END -> S2 attention
     name mover heads:      END -> IO attention
4. Similarity of each condition's head-effect map to the exact map

Unlike PathPatchingAnalyzer, prompts here have different lengths (typo'd names
split into several tokens), so logits and attention are read at each example's
own END position rather than at index -1.

Usage:
    python general_analysis/typo_ioi_analysis.py --target s2 --n 300
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformer_lens import HookedTransformer

sys.path.insert(0, str(Path(__file__).parent))

from plotting import save_path_patching_heatmap, print_top_heads
from utils import set_seed


REPO_ROOT = Path(__file__).parent.parent

# GPT-2 small IOI circuit heads from Wang et al. (2022), as (layer, head)
KNOWN_HEADS = {
    "Duplicate token": [(0, 1), (0, 10), (3, 0)],
    "Induction": [(5, 5), (5, 8), (5, 9), (6, 9)],
    "S-inhibition": [(7, 3), (7, 9), (8, 6), (8, 10)],
    "Name mover": [(9, 6), (9, 9), (10, 0)],
    "Negative name mover": [(10, 7), (11, 10)],
}

# Attention metric that characterises each head class
CLASS_METRIC = {
    "Duplicate token": "s2_to_s1",
    "Induction": "s2_to_after_s1",
    "S-inhibition": "end_to_s2",
    "Name mover": "end_to_io",
    "Negative name mover": "end_to_io",
}

SERIES_COLOR = "#2a78d6"
TEXT_SECONDARY = "#52514e"


# ============================================================================
# Data loading
# ============================================================================

def load_typo_pairs(target: str) -> Dict:
    path = REPO_ROOT / "data_generation" / "output" / f"typo_ioi_pairs_{target}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            f"Please run: python data_generation/generate_typo_ioi_pairs.py"
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def token_span(offsets: List[List[int]], start: int, end: int) -> List[int]:
    """Token indices (with BOS offset) overlapping the character span [start, end)."""
    return [i + 1 for i, (a, b) in enumerate(offsets) if b > start and a < end]


def prepare_condition(model: HookedTransformer, pairs: List[Dict]) -> Dict[str, torch.Tensor]:
    """Tokenize one condition and locate END / IO / S1 / S2 positions per example."""
    device = model.cfg.device
    clean = [p["clean"] for p in pairs]
    corrupt = [p["corrupt"] for p in pairs]

    clean_tokens = model.to_tokens(clean, prepend_bos=True)
    corrupt_tokens = model.to_tokens(corrupt, prepend_bos=True)
    assert clean_tokens.shape == corrupt_tokens.shape, "clean/corrupt must be token-aligned"

    end_pos, io_pos, s1_last, s2_last = [], [], [], []
    s1_mask = torch.zeros(clean_tokens.shape, dtype=torch.bool)
    s2_mask = torch.zeros(clean_tokens.shape, dtype=torch.bool)
    for i, p in enumerate(pairs):
        enc = model.tokenizer(p["clean"], return_offsets_mapping=True)
        n = len(enc["input_ids"])
        assert n == len(model.tokenizer(p["corrupt"])["input_ids"]), p
        offsets = enc["offset_mapping"]
        spans = {k: token_span(offsets, *v) for k, v in p["clean_spans"].items()}
        end_pos.append(n)  # BOS at 0, so last real token is at index n
        io_pos.append(spans["io"][-1])
        s1_last.append(spans["s1"][-1])
        s2_last.append(spans["s2"][-1])
        s1_mask[i, spans["s1"]] = True
        s2_mask[i, spans["s2"]] = True

    as_t = lambda x: torch.tensor(x, device=device)
    return {
        "clean_tokens": clean_tokens,
        "corrupt_tokens": corrupt_tokens,
        "io_toks": as_t([model.to_single_token(" " + p["io_name"]) for p in pairs]),
        "s_toks": as_t([model.to_single_token(" " + p["s_name"]) for p in pairs]),
        "end_pos": as_t(end_pos),
        "io_pos": as_t(io_pos),
        "s1_last": as_t(s1_last),
        "s2_last": as_t(s2_last),
        "s1_mask": s1_mask.to(device),
        "s2_mask": s2_mask.to(device),
    }


# ============================================================================
# Metrics
# ============================================================================

def end_logits(logits: torch.Tensor, end_pos: torch.Tensor) -> torch.Tensor:
    return logits[torch.arange(len(end_pos), device=end_pos.device), end_pos]


def logit_diff(logits: torch.Tensor, d: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Per-example logit(IO) - logit(S) at each example's END position."""
    final = end_logits(logits, d["end_pos"])
    idx = torch.arange(len(final), device=final.device)
    return final[idx, d["io_toks"]] - final[idx, d["s_toks"]]


def behaviour(model: HookedTransformer, d: Dict[str, torch.Tensor]) -> Dict[str, float]:
    with torch.inference_mode():
        clean_logits = model(d["clean_tokens"])
        corrupt_logits = model(d["corrupt_tokens"])
    clean_ld = logit_diff(clean_logits, d)
    corrupt_ld = logit_diff(corrupt_logits, d)
    preds = end_logits(clean_logits, d["end_pos"]).argmax(-1)
    return {
        "clean_logit_diff": clean_ld.mean().item(),
        "clean_logit_diff_std": clean_ld.std().item(),
        "corrupt_logit_diff": corrupt_ld.mean().item(),
        "corruption_effect": (clean_ld - corrupt_ld).mean().item(),
        "p_io_over_s": (clean_ld > 0).float().mean().item(),
        "top1_accuracy": (preds == d["io_toks"]).float().mean().item(),
    }


def attention_scores(model: HookedTransformer, d: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Mean attention [n_layers, n_heads] for each IOI-relevant query/key pair."""
    n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads
    scores = {k: torch.zeros(n_layers, n_heads) for k in
              ("s2_to_s1", "s2_to_after_s1", "end_to_s2", "end_to_io")}
    b = torch.arange(len(d["end_pos"]), device=d["end_pos"].device)

    with torch.inference_mode():
        _, cache = model.run_with_cache(
            d["clean_tokens"], names_filter=lambda name: name.endswith("hook_pattern")
        )
    for layer in range(n_layers):
        pattern = cache[f"blocks.{layer}.attn.hook_pattern"]  # [batch, head, q, k]
        from_s2 = pattern[b, :, d["s2_last"], :]  # [batch, head, k]
        from_end = pattern[b, :, d["end_pos"], :]
        scores["s2_to_s1"][layer] = (from_s2 * d["s1_mask"][:, None, :]).sum(-1).mean(0).cpu()
        scores["s2_to_after_s1"][layer] = from_s2[b, :, d["s1_last"] + 1].mean(0).cpu()
        scores["end_to_s2"][layer] = (from_end * d["s2_mask"][:, None, :]).sum(-1).mean(0).cpu()
        scores["end_to_io"][layer] = from_end[b, :, d["io_pos"]].mean(0).cpu()
    return scores


def head_effects(model: HookedTransformer, d: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Head -> logits path patching (same method as PathPatchingAnalyzer):
    patch one head's clean output into the corrupt run, measure the change in
    logit diff relative to the corrupt baseline.
    """
    n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads
    effects = torch.zeros(n_layers, n_heads)

    with torch.inference_mode():
        _, clean_cache = model.run_with_cache(
            d["clean_tokens"], names_filter=lambda name: name.endswith("attn.hook_z")
        )
        corrupt_ld = logit_diff(model(d["corrupt_tokens"]), d).mean().item()

    for layer in range(n_layers):
        print(f"Layer {layer:2d}...", end=" ", flush=True)
        clean_z = clean_cache[f"blocks.{layer}.attn.hook_z"]
        for head in range(n_heads):
            def patch_z(activation, hook):
                activation[:, :, head, :] = clean_z[:, :, head, :]
                return activation

            with torch.inference_mode():
                patched = model.run_with_hooks(
                    d["corrupt_tokens"], fwd_hooks=[(f"blocks.{layer}.attn.hook_z", patch_z)]
                )
            effects[layer, head] = logit_diff(patched, d).mean().item() - corrupt_ld
        print("✓")
    return effects


def compare_to_exact(effects: Dict[str, torch.Tensor], top_k: int = 10) -> Dict[str, Dict]:
    """Pearson r and top-k |effect| head overlap between each condition and exact."""
    ref = effects["exact"].flatten()
    ref_top = set(torch.topk(ref.abs(), top_k).indices.tolist())
    out = {}
    for cond, eff in effects.items():
        flat = eff.flatten()
        top = set(torch.topk(flat.abs(), top_k).indices.tolist())
        out[cond] = {
            "pearson_r_vs_exact": float(np.corrcoef(ref.numpy(), flat.numpy())[0, 1]),
            f"top{top_k}_overlap_vs_exact": len(ref_top & top) / top_k,
        }
    return out


# ============================================================================
# Plotting
# ============================================================================

def _style_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#b5b4ae")
    ax.spines["bottom"].set_color("#b5b4ae")
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax.grid(axis="y", color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)


def _bar_panel(ax, conditions, values, title, fmt="{:.2f}"):
    bars = ax.bar(conditions, values, color=SERIES_COLOR, width=0.6)
    ax.axhline(0, color="#8a8983", linewidth=0.8)
    ax.set_title(title, fontsize=11, loc="left")
    _style_axes(ax)
    for bar, v in zip(bars, values):
        ax.annotate(fmt.format(v), (bar.get_x() + bar.get_width() / 2, v),
                    xytext=(0, 3 if v >= 0 else -11), textcoords="offset points",
                    ha="center", fontsize=8, color=TEXT_SECONDARY)


def plot_behaviour(results: Dict, conditions: List[str], out_dir: Path, target: str):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    b = results["behaviour"]
    _bar_panel(axes[0], conditions, [b[c]["clean_logit_diff"] for c in conditions],
               "Logit diff, IO − S (clean)")
    _bar_panel(axes[1], conditions, [100 * b[c]["p_io_over_s"] for c in conditions],
               "% examples with IO > S", fmt="{:.0f}")
    _bar_panel(axes[2], conditions, [results["similarity"][c]["pearson_r_vs_exact"] for c in conditions],
               "Head-effect map: Pearson r vs exact")
    fig.suptitle(f"Typo IOI behaviour — typo at {target.upper()}", fontsize=13, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out_dir / "behaviour_by_condition.png", dpi=160)
    plt.close(fig)


def plot_effect_grid(effects: Dict[str, torch.Tensor], conditions: List[str], out_dir: Path, target: str):
    vmax = max(e.abs().max().item() for e in effects.values())
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True, sharey=True)
    for ax, cond in zip(axes.flat, conditions):
        im = ax.imshow(effects[cond].numpy(), cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_title(cond, fontsize=11, loc="left")
        ax.set_xticks(range(0, 12, 2))
        ax.set_yticks(range(0, 12, 2))
    for ax in axes[-1]:
        ax.set_xlabel("Head")
    for ax in axes[:, 0]:
        ax.set_ylabel("Layer")
    fig.colorbar(im, ax=axes, shrink=0.8, label="Δ logit diff (patched − corrupt)")
    fig.suptitle(f"Head → logits path patching by condition — typo at {target.upper()}",
                 fontsize=13, x=0.01, ha="left")
    fig.savefig(out_dir / "effects_by_condition.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_head_classes(results: Dict, conditions: List[str], out_dir: Path, target: str):
    """Small multiples: one panel per IOI head class, its signature attention by condition."""
    classes = list(KNOWN_HEADS)
    fig, axes = plt.subplots(1, len(classes), figsize=(4 * len(classes), 3.8), sharey=True)
    for ax, cls in zip(axes, classes):
        metric = CLASS_METRIC[cls]
        vals = [results["head_classes"][c][cls]["attention"] for c in conditions]
        _bar_panel(ax, conditions, vals, f"{cls}\n({metric.replace('_', ' ')})")
        ax.tick_params(axis="x", rotation=45)
    axes[0].set_ylabel("Mean attention (class heads)")
    fig.suptitle(f"IOI head-class attention by condition — typo at {target.upper()}",
                 fontsize=13, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out_dir / "head_class_attention.png", dpi=160)
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================

def run(target: str, n_examples: int, model_name: str, output_dir: Path, seed: int = 42) -> Dict:
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {model_name} on {device}...")
    model = HookedTransformer.from_pretrained(model_name, device=device)

    data = load_typo_pairs(target)
    conditions = data["conditions"]
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {"target": target, "n_examples": None, "behaviour": {}, "head_classes": {}}
    effects, attn = {}, {}
    for cond in conditions:
        pairs = data["pairs"][cond][:n_examples]
        results["n_examples"] = len(pairs)
        print("\n" + "=" * 80)
        print(f"CONDITION: {cond}  (typo at {target.upper()}, n={len(pairs)})")
        print(f"  e.g. {pairs[0]['clean']}")
        print("=" * 80)

        d = prepare_condition(model, pairs)
        stats = behaviour(model, d)
        stats["mean_variant_n_tokens"] = float(np.mean([p["variant_n_tokens"] for p in pairs]))
        results["behaviour"][cond] = stats
        print(f"  clean logit diff {stats['clean_logit_diff']:.3f} | IO>S {stats['p_io_over_s']*100:.1f}% "
              f"| top-1 {stats['top1_accuracy']*100:.1f}% | corrupt {stats['corrupt_logit_diff']:.3f}")

        attn[cond] = attention_scores(model, d)
        effects[cond] = head_effects(model, d)
        print_top_heads(effects[cond], top_k=10, label=f"Top heads ({cond})")

        results["head_classes"][cond] = {
            cls: {
                "attention": float(np.mean([attn[cond][CLASS_METRIC[cls]][l, h].item() for l, h in heads])),
                "effect": float(np.mean([effects[cond][l, h].item() for l, h in heads])),
            }
            for cls, heads in KNOWN_HEADS.items()
        }
        save_path_patching_heatmap(
            effects=effects[cond], output_dir=str(output_dir),
            filename=f"effects_{cond}.png",
            title=f"Direct Effect: Head → Logits (Typo IOI, {cond}, typo at {target.upper()}, n={len(pairs)})",
        )

    results["similarity"] = compare_to_exact(effects)

    torch.save(effects, output_dir / "effects.pt")
    torch.save(attn, output_dir / "attention_scores.pt")
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    plot_behaviour(results, conditions, output_dir, target)
    plot_effect_grid(effects, conditions, output_dir, target)
    plot_head_classes(results, conditions, output_dir, target)

    print("\n" + "=" * 80)
    print(f"SUMMARY (typo at {target.upper()})")
    print("=" * 80)
    print(f"{'condition':10s} {'logit diff':>10s} {'IO>S':>7s} {'top-1':>7s} {'r vs exact':>11s} {'top10 ovl':>10s}")
    for c in conditions:
        b, s = results["behaviour"][c], results["similarity"][c]
        print(f"{c:10s} {b['clean_logit_diff']:10.3f} {b['p_io_over_s']*100:6.1f}% {b['top1_accuracy']*100:6.1f}% "
              f"{s['pearson_r_vs_exact']:11.3f} {s['top10_overlap_vs_exact']*100:9.0f}%")
    print(f"\n✅ Results saved to: {output_dir}/")
    return results


def main():
    parser = argparse.ArgumentParser(description="Typo IOI path patching analysis")
    parser.add_argument("--target", choices=["s2", "s1"], default="s2",
                        help="Which subject mention is misspelled")
    parser.add_argument("--n", type=int, default=300, help="Examples per condition")
    parser.add_argument("--model", default="gpt2-small")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir or REPO_ROOT / "results" / f"typo_ioi_{args.target}" / f"{args.n}_examples")
    run(args.target, args.n, args.model, output_dir)


if __name__ == "__main__":
    main()
