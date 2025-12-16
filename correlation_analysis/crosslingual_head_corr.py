"""
Compute correlation between two head activation-frequency matrices.

Inputs:
- Two .npy files with shape [n_layers, n_heads] (as produced by activation_frequencies.py)

Outputs:
- Prints Pearson & Spearman correlations
- Optionally saves a scatter plot and/or per-layer correlation plot

How to run this code, with an example below:
python crosslingual_head_corr.py \
  --a results/correlation/actfreq_en.npy \ # "a" argument uses the first npz as input, saved by the activation_frequencies_ifr.py
  --b results/correlation/actfreq_zh.npy \ # "b" argument uses the second npz as input, saved by the activation_frequencies_ifr.py
  --out_json results/correlation/corr_en_zh.json \ #saves the correlation results in a json file
  --scatter_png results/correlation/corr_en_zh_scatter.png  #saves a scatter plot as a png. This is an optional argument.
  
Example usage:
python3 crosslingual_head_corr.py --a results/correlation/actfreq_en_ifr_bloom_mps.npy --b results/correlation/actfreq_zh_ifr_bloom_mps.npy
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    from scipy.stats import pearsonr, spearmanr
except ImportError as e:
    raise ImportError("Install scipy: `pip install scipy`") from e

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def _load(path: str) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim != 2:
        raise ValueError(f"{path} must be a 2D array [n_layers, n_heads], got shape {arr.shape}")
    return arr


def _apply_layer_slice(arr: np.ndarray, layer_slice: Optional[str]) -> np.ndarray:
    if not layer_slice:
        return arr
    # supports "start:end" like python slices; end optional
    parts = layer_slice.split(":")
    if len(parts) != 2:
        raise ValueError("--layers must be in 'start:end' format (e.g. 0:12, 5:11, :6, 6:)")
    start = int(parts[0]) if parts[0] != "" else None
    end = int(parts[1]) if parts[1] != "" else None
    return arr[start:end, :]


def _mask_flat(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Remove NaNs/Infs in either vector and return filtered vectors + stats.
    """
    finite = np.isfinite(x) & np.isfinite(y)
    removed = int((~finite).sum())
    return x[finite], y[finite], {"removed_nonfinite": removed, "kept": int(finite.sum())}


def _corr_safe(x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """
    Pearson + Spearman with safeguards:
    - drop non-finite values
    - handle constant vectors (pearsonr/spearmanr may error or return nan)
    """
    x = x.astype(np.float64).reshape(-1)
    y = y.astype(np.float64).reshape(-1)

    x, y, mask_stats = _mask_flat(x, y)

    out: Dict[str, Any] = {"n": int(x.size), **mask_stats}

    if x.size < 2:
        out.update({"pearson_r": float("nan"), "pearson_p": float("nan"),
                    "spearman_r": float("nan"), "spearman_p": float("nan"),
                    "note": "Not enough data points after filtering"})
        return out

    # Constant checks
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        out.update({"pearson_r": float("nan"), "pearson_p": float("nan"),
                    "spearman_r": float("nan"), "spearman_p": float("nan"),
                    "note": "One of the vectors is constant; correlation undefined"})
        return out

    p = pearsonr(x, y)
    s = spearmanr(x, y)

    out.update({
        "pearson_r": float(p.statistic),
        "pearson_p": float(p.pvalue),
        "spearman_r": float(s.statistic),
        "spearman_p": float(s.pvalue),
    })
    return out


def _layerwise_corr(a: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
    """
    Per-layer Pearson/Spearman across heads (each row is a layer).
    Returns lists length n_layers.
    """
    n_layers = a.shape[0]
    pear = []
    spear = []
    for L in range(n_layers):
        stats = _corr_safe(a[L], b[L])
        pear.append(stats["pearson_r"])
        spear.append(stats["spearman_r"])
    return {"pearson_r": pear, "spearman_r": spear}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True, help="First activation frequency .npy (e.g., English)")
    p.add_argument("--b", required=True, help="Second activation frequency .npy (e.g., Chinese)")

    p.add_argument("--layers", default=None, help="Optional layer slice 'start:end' (e.g. 0:12, :6, 6:)")
    p.add_argument("--out_json", default=None, help="Optional path to save correlation summary as JSON")
    p.add_argument("--scatter_png", default=None, help="Optional scatter plot output PNG")
    p.add_argument("--layer_png", default=None, help="Optional per-layer correlation plot PNG")
    p.add_argument("--no_layerwise", action="store_true", help="Skip layerwise correlation computation")

    args = p.parse_args()

    a = _load(args.a)
    b = _load(args.b)

    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {args.a} has {a.shape}, {args.b} has {b.shape}")

    a = _apply_layer_slice(a, args.layers)
    b = _apply_layer_slice(b, args.layers)

    x = a.reshape(-1)
    y = b.reshape(-1)

    summary: Dict[str, Any] = {
        "a_path": args.a,
        "b_path": args.b,
        "shape_loaded": list(np.load(args.a).shape),
        "shape_used": list(a.shape),
        "layers": args.layers,
        "overall": _corr_safe(x, y),
    }

    if not args.no_layerwise:
        summary["layerwise"] = _layerwise_corr(a, b)

    print("Overall correlation (flattened over all selected heads):")
    print(f"  N={summary['overall']['n']} (kept={summary['overall']['kept']}, removed_nonfinite={summary['overall']['removed_nonfinite']})")
    pr = summary["overall"]["pearson_r"]
    pp = summary["overall"]["pearson_p"]
    sr = summary["overall"]["spearman_r"]
    sp = summary["overall"]["spearman_p"]
    print(f"  Pearson  r={pr:.4f}  p={pp:.3e}" if np.isfinite(pr) else "  Pearson  r=nan  p=nan")
    print(f"  Spearman r={sr:.4f}  p={sp:.3e}" if np.isfinite(sr) else "  Spearman r=nan  p=nan")
    if "note" in summary["overall"]:
        print(f"  Note: {summary['overall']['note']}")

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved JSON: {args.out_json}")

    if args.scatter_png:
        if plt is None:
            raise ImportError("matplotlib is required for plots: `pip install matplotlib`")

        # Use filtered finite points for plotting too
        xf, yf, _ = _mask_flat(x.astype(np.float64), y.astype(np.float64))

        os.makedirs(os.path.dirname(args.scatter_png) or ".", exist_ok=True)
        plt.figure()
        plt.scatter(xf, yf, s=8, alpha=0.5)
        plt.xlabel("Activation frequency A")
        plt.ylabel("Activation frequency B")
        title_r = summary["overall"].get("pearson_r", float("nan"))
        if np.isfinite(title_r):
            plt.title(f"Head activation frequency correlation (Pearson r={title_r:.3f})")
        else:
            plt.title("Head activation frequency correlation (Pearson r=nan)")
        plt.tight_layout()
        plt.savefig(args.scatter_png, dpi=200)
        plt.close()
        print(f"Saved scatter: {args.scatter_png}")

    if args.layer_png and (not args.no_layerwise):
        if plt is None:
            raise ImportError("matplotlib is required for plots: `pip install matplotlib`")

        os.makedirs(os.path.dirname(args.layer_png) or ".", exist_ok=True)
        pear_L = np.array(summary["layerwise"]["pearson_r"], dtype=np.float64)
        spear_L = np.array(summary["layerwise"]["spearman_r"], dtype=np.float64)

        plt.figure()
        plt.plot(pear_L, label="Pearson r (per layer)")
        plt.plot(spear_L, label="Spearman r (per layer)")
        plt.xlabel("Layer (after slicing)")
        plt.ylabel("Correlation")
        plt.title("Per-layer correlation across heads")
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.layer_png, dpi=200)
        plt.close()
        print(f"Saved per-layer plot: {args.layer_png}")


if __name__ == "__main__":
    main()




"""
Compute correlation between two head activation-frequency matrices.

Inputs:
- Two .npy files with shape [n_layers, n_heads] (as produced by activation_frequencies.py)

Outputs:
- Prints Pearson & Spearman correlations
- Optionally saves a scatter plot and/or per-layer correlation plot

Usage:
python general_analysis/crosslingual_head_corr.py \
  --a results/actfreq_en.npy \
  --b results/actfreq_zh.npy \
  --out_json results/corr_en_zh.json \
  --scatter_png results/corr_en_zh_scatter.png
"""

# from __future__ import annotations

# import argparse
# import json
# import os
# from typing import Any, Dict, Tuple

# import numpy as np

# try:
#     from scipy.stats import pearsonr, spearmanr
# except ImportError as e:
#     raise ImportError("Install scipy: `pip install scipy`") from e

# try:
#     import matplotlib.pyplot as plt
# except ImportError:
#     plt = None


# def _load(path: str) -> np.ndarray:
#     arr = np.load(path)
#     if arr.ndim != 2:
#         raise ValueError(f"{path} must be a 2D array [n_layers, n_heads], got shape {arr.shape}")
#     return arr


# def _corr(x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
#     x = x.astype(np.float64)
#     y = y.astype(np.float64)

#     pearson = pearsonr(x, y)
#     spearman = spearmanr(x, y)

#     return {
#         "pearson_r": float(pearson.statistic),
#         "pearson_p": float(pearson.pvalue),
#         "spearman_r": float(spearman.statistic),
#         "spearman_p": float(spearman.pvalue),
#     }


# def _layerwise_corr(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     Returns per-layer Pearson r and Spearman r (shape [n_layers]).
#     """
#     n_layers = a.shape[0]
#     pear = np.zeros((n_layers,), dtype=np.float64)
#     spear = np.zeros((n_layers,), dtype=np.float64)
#     for L in range(n_layers):
#         pear[L] = pearsonr(a[L], b[L]).statistic
#         spear[L] = spearmanr(a[L], b[L]).statistic
#     return pear, spear


# def main() -> None:
#     p = argparse.ArgumentParser()
#     p.add_argument("--a", required=True, help="First activation frequency .npy (e.g., English)")
#     p.add_argument("--b", required=True, help="Second activation frequency .npy (e.g., Chinese)")
#     p.add_argument("--out_json", default=None, help="Optional path to save correlation summary as JSON")
#     p.add_argument("--scatter_png", default=None, help="Optional scatter plot output PNG")
#     p.add_argument("--layer_png", default=None, help="Optional per-layer correlation plot PNG")
#     p.add_argument("--no_layerwise", action="store_true", help="Skip layerwise correlation computation")

#     args = p.parse_args()

#     a = _load(args.a)
#     b = _load(args.b)

#     if a.shape != b.shape:
#         raise ValueError(f"Shape mismatch: {args.a} has {a.shape}, {args.b} has {b.shape}")

#     x = a.reshape(-1)
#     y = b.reshape(-1)

#     summary = {
#         "a_path": args.a,
#         "b_path": args.b,
#         "shape": list(a.shape),
#         "overall": _corr(x, y),
#     }

#     if not args.no_layerwise:
#         pear_L, spear_L = _layerwise_corr(a, b)
#         summary["layerwise"] = {
#             "pearson_r": pear_L.tolist(),
#             "spearman_r": spear_L.tolist(),
#         }

#     print("Overall correlation (flattened over all heads):")
#     print(f"  Pearson  r={summary['overall']['pearson_r']:.4f}  p={summary['overall']['pearson_p']:.3e}")
#     print(f"  Spearman r={summary['overall']['spearman_r']:.4f}  p={summary['overall']['spearman_p']:.3e}")

#     if args.out_json:
#         os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
#         with open(args.out_json, "w", encoding="utf-8") as f:
#             json.dump(summary, f, indent=2)
#         print(f"Saved JSON: {args.out_json}")

#     if args.scatter_png:
#         if plt is None:
#             raise ImportError("matplotlib is required for plots: `pip install matplotlib`")
#         os.makedirs(os.path.dirname(args.scatter_png) or ".", exist_ok=True)
#         plt.figure()
#         plt.scatter(x, y, s=8, alpha=0.5)
#         plt.xlabel("Activation frequency A")
#         plt.ylabel("Activation frequency B")
#         plt.title(f"Head activation frequency correlation (Pearson r={summary['overall']['pearson_r']:.3f})")
#         plt.tight_layout()
#         plt.savefig(args.scatter_png, dpi=200)
#         plt.close()
#         print(f"Saved scatter: {args.scatter_png}")

#     if args.layer_png and not args.no_layerwise:
#         if plt is None:
#             raise ImportError("matplotlib is required for plots: `pip install matplotlib`")
#         os.makedirs(os.path.dirname(args.layer_png) or ".", exist_ok=True)
#         pear_L = np.array(summary["layerwise"]["pearson_r"], dtype=np.float64)
#         spear_L = np.array(summary["layerwise"]["spearman_r"], dtype=np.float64)

#         plt.figure()
#         plt.plot(pear_L, label="Pearson r (per layer)")
#         plt.plot(spear_L, label="Spearman r (per layer)")
#         plt.xlabel("Layer")
#         plt.ylabel("Correlation")
#         plt.title("Per-layer correlation across heads")
#         plt.legend()
#         plt.tight_layout()
#         plt.savefig(args.layer_png, dpi=200)
#         plt.close()
#         print(f"Saved per-layer plot: {args.layer_png}")


# if __name__ == "__main__":
#     main()
