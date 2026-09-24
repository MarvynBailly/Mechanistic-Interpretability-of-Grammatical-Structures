"""
Where (if anywhere) does GPT-2 repair a typo'd name?

Follow-up to typo_ioi_analysis.py. For each typo condition we take the exact
(correctly spelled) run of the same base example as the source and the typo'd
run as the target, and ask three questions layer by layer:

1. Residual stream patching (exact -> typo). Patch the exact run's residual
   stream into the typo'd run at one position and one layer, and measure how
   much of the logit diff comes back:
       recovery = (patched - typo) / (exact - typo)
   Two positions: the typo'd name's last token, and END.
   If patching the name position stops helping after layer L, the information
   at that position has already been read by later positions by layer L.

2. Name retrieval. At each layer, is the typo'd name's representation (last
   token) closest to the right name? We build one prototype per name from the
   exact runs (mean-centred residuals at the name position) and classify each
   typo'd representation by nearest prototype (cosine). Accuracy above chance
   that grows with depth = the model is rebuilding the word ("repair").

3. Component patching. Patch just one layer's attention output or MLP output
   (exact -> typo) at the typo'd name, to see which components carry the
   name's identity there.

The `unrelated` condition (a different name) is the reference: no repair is
possible there, so a typo curve that matches it means no repair happened.

Usage:
    python general_analysis/typo_repair_analysis.py --target s2 --n 300
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

from typo_ioi_analysis import REPO_ROOT, load_typo_pairs, logit_diff, prepare_condition, _style_axes
from utils import set_seed


TYPO_CONDITIONS = ["swap", "drop", "double", "sub", "unrelated"]

# Categorical slots in fixed order; the unrelated control is a neutral reference
CONDITION_STYLE = {
    "swap": dict(color="#2a78d6"),
    "drop": dict(color="#eb6834"),
    "double": dict(color="#1baf7a"),
    "sub": dict(color="#eda100"),
    "unrelated": dict(color="#8a8983", linestyle="--"),
}
TEXT_SECONDARY = "#52514e"


def resid_hook_names(n_layers: int) -> List[str]:
    """Residual stream entering each block, plus the final residual stream."""
    return [f"blocks.{l}.hook_resid_pre" for l in range(n_layers)] + [f"blocks.{n_layers - 1}.hook_resid_post"]


def recovery(patched: float, typo: float, exact: float) -> float:
    return (patched - typo) / (exact - typo)


def patch_positions(
    model: HookedTransformer,
    target: Dict[str, torch.Tensor],
    hook_name: str,
    source_acts: torch.Tensor,
    src_pos: torch.Tensor,
    tgt_pos: torch.Tensor,
) -> float:
    """Run the typo'd prompts with source activations written at one position per example."""
    b = torch.arange(len(tgt_pos), device=tgt_pos.device)
    vectors = source_acts[b, src_pos]

    def hook(activation, hook):
        activation[b, tgt_pos] = vectors
        return activation

    with torch.inference_mode():
        logits = model.run_with_hooks(target["clean_tokens"], fwd_hooks=[(hook_name, hook)])
    return logit_diff(logits, target).mean().item()


def name_retrieval_accuracy(
    exact_reps: torch.Tensor,
    typo_reps: torch.Tensor,
    names: List[str],
) -> float:
    """
    Nearest-prototype accuracy: does each typo'd rep sit closest to its own
    name's (mean-centred) exact-run prototype?
    """
    mu = exact_reps.mean(0, keepdim=True)
    exact_c, typo_c = exact_reps - mu, typo_reps - mu
    vocab = sorted(set(names))
    labels = torch.tensor([vocab.index(n) for n in names])
    protos = torch.stack([exact_c[labels == k].mean(0) for k in range(len(vocab))])
    sims = torch.nn.functional.normalize(typo_c, dim=-1) @ torch.nn.functional.normalize(protos, dim=-1).T
    return (sims.argmax(-1) == labels).float().mean().item()


def run(target_slot: str, n_examples: int, model_name: str, output_dir: Path, seed: int = 42) -> Dict:
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {model_name} on {device}...")
    model = HookedTransformer.from_pretrained(model_name, device=device)
    n_layers = model.cfg.n_layers
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_typo_pairs(target_slot)
    slot = f"{target_slot}_last"  # the typo'd mention's last token
    exact_pairs = data["pairs"]["exact"][:n_examples]
    names = [p["s_name"] for p in exact_pairs]
    exact = prepare_condition(model, exact_pairs)

    resid_names = resid_hook_names(n_layers)
    comp_names = [f"blocks.{l}.hook_attn_out" for l in range(n_layers)] + \
                 [f"blocks.{l}.hook_mlp_out" for l in range(n_layers)]
    wanted = set(resid_names + comp_names)
    with torch.inference_mode():
        exact_logits, exact_cache = model.run_with_cache(
            exact["clean_tokens"], names_filter=lambda name: name in wanted
        )
    exact_ld = logit_diff(exact_logits, exact).mean().item()
    b = torch.arange(len(exact_pairs), device=device)
    exact_name_reps = [exact_cache[h][b, exact[slot]].float().cpu() for h in resid_names]

    results = {"target": target_slot, "n_examples": len(exact_pairs), "exact_logit_diff": exact_ld,
               "chance_retrieval": 1 / len(set(names)), "conditions": {}}

    for cond in TYPO_CONDITIONS:
        pairs = data["pairs"][cond][:n_examples]
        assert [p["example_id"] for p in pairs] == [p["example_id"] for p in exact_pairs]
        typo = prepare_condition(model, pairs)
        print(f"\n=== {cond} (typo at {target_slot.upper()}) ===")

        with torch.inference_mode():
            typo_logits, typo_cache = model.run_with_cache(
                typo["clean_tokens"], names_filter=lambda name: name in resid_names
            )
        typo_ld = logit_diff(typo_logits, typo).mean().item()

        res = {"typo_logit_diff": typo_ld, "resid_name": [], "resid_end": [],
               "attn_name": [], "mlp_name": [], "retrieval": []}
        for i, h in enumerate(resid_names):
            res["resid_name"].append(recovery(
                patch_positions(model, typo, h, exact_cache[h], exact[slot], typo[slot]), typo_ld, exact_ld))
            res["resid_end"].append(recovery(
                patch_positions(model, typo, h, exact_cache[h], exact["end_pos"], typo["end_pos"]), typo_ld, exact_ld))
            typo_reps = typo_cache[h][b, typo[slot]].float().cpu()
            res["retrieval"].append(name_retrieval_accuracy(exact_name_reps[i], typo_reps, names))

        for l in range(n_layers):
            for kind in ("attn", "mlp"):
                h = f"blocks.{l}.hook_{kind}_out"
                res[f"{kind}_name"].append(recovery(
                    patch_positions(model, typo, h, exact_cache[h], exact[slot], typo[slot]), typo_ld, exact_ld))

        results["conditions"][cond] = res
        fmt = lambda xs: " ".join(f"{x:+.2f}" for x in xs)
        print(f"  logit diff: exact {exact_ld:.2f}, typo {typo_ld:.2f}")
        print(f"  recovery, resid @ name (L0..final): {fmt(res['resid_name'])}")
        print(f"  recovery, resid @ END  (L0..final): {fmt(res['resid_end'])}")
        print(f"  recovery, attn_out @ name (L0..11): {fmt(res['attn_name'])}")
        print(f"  recovery, mlp_out  @ name (L0..11): {fmt(res['mlp_name'])}")
        print(f"  name retrieval acc (L0..final):     {fmt(res['retrieval'])}  (chance {results['chance_retrieval']:.2f})")

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    plot_all(results, output_dir, n_layers)
    print(f"\n✅ Results saved to: {output_dir}/")
    return results


# ============================================================================
# Plotting
# ============================================================================

def _lines(ax, results: Dict, key: str, xs, title: str, ylabel: str):
    for cond in TYPO_CONDITIONS:
        ax.plot(xs, results["conditions"][cond][key], label=cond, linewidth=2,
                marker="o", markersize=4, **CONDITION_STYLE[cond])
    ax.set_title(title, fontsize=11, loc="left")
    ax.set_ylabel(ylabel, color=TEXT_SECONDARY)
    _style_axes(ax)
    ax.grid(axis="x", visible=False)


def plot_all(results: Dict, out_dir: Path, n_layers: int):
    target = results["target"].upper()
    resid_x = list(range(n_layers + 1))
    resid_labels = [str(l) for l in range(n_layers)] + ["final"]

    # 1. Residual stream patching recovery
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)
    _lines(axes[0], results, "resid_name", resid_x, f"Patch at the typo'd name ({target})",
           "Recovery of logit diff\n(0 = typo, 1 = exact)")
    _lines(axes[1], results, "resid_end", resid_x, "Patch at END", "")
    for ax in axes:
        ax.set_xticks(resid_x)
        ax.set_xticklabels(resid_labels)
        ax.set_xlabel("Residual stream entering layer", color=TEXT_SECONDARY)
        ax.axhline(1, color="#b5b4ae", linewidth=0.8, linestyle=":")
    axes[1].legend(frameon=False, fontsize=9)
    fig.suptitle(f"Residual stream patching, exact → typo'd run — typo at {target}", fontsize=13, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out_dir / "resid_patching_recovery.png", dpi=160)
    plt.close(fig)

    # 2. Name retrieval
    fig, ax = plt.subplots(figsize=(8, 4.5))
    _lines(ax, results, "retrieval", resid_x, "Is the typo'd name closest to the right name?",
           "Nearest-prototype accuracy")
    ax.axhline(results["chance_retrieval"], color="#b5b4ae", linewidth=1, linestyle=":")
    ax.annotate("chance", (resid_x[-1], results["chance_retrieval"]), xytext=(4, 2),
                textcoords="offset points", fontsize=8, color=TEXT_SECONDARY)
    ax.set_xticks(resid_x)
    ax.set_xticklabels(resid_labels)
    ax.set_xlabel("Residual stream entering layer", color=TEXT_SECONDARY)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=9)
    fig.suptitle(f"Name identity at the typo'd name's last token — typo at {target}", fontsize=13, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out_dir / "name_retrieval_by_layer.png", dpi=160)
    plt.close(fig)

    # 3. Component patching
    comp_x = list(range(n_layers))
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)
    _lines(axes[0], results, "attn_name", comp_x, "Attention output at the typo'd name",
           "Recovery of logit diff")
    _lines(axes[1], results, "mlp_name", comp_x, "MLP output at the typo'd name", "")
    for ax in axes:
        ax.set_xticks(comp_x)
        ax.set_xlabel("Layer", color=TEXT_SECONDARY)
    axes[1].legend(frameon=False, fontsize=9)
    fig.suptitle(f"Component patching, exact → typo'd run — typo at {target}", fontsize=13, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out_dir / "component_patching_recovery.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Where does GPT-2 repair a typo'd name?")
    parser.add_argument("--target", choices=["s2", "s1"], default="s2",
                        help="Which subject mention is misspelled")
    parser.add_argument("--n", type=int, default=300, help="Examples per condition")
    parser.add_argument("--model", default="gpt2-small")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir or REPO_ROOT / "results" / f"typo_repair_{args.target}" / f"{args.n}_examples")
    run(args.target, args.n, args.model, output_dir)


if __name__ == "__main__":
    main()
