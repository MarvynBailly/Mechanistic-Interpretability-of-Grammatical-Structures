"""
General Path Patching Framework for Identifying Significant Heads

This module provides a unified interface for:
1. Loading any dataset (IOI, color-object, code completion)
2. Validating task solvability (clean accuracy check)
3. Running path patching analysis
4. Identifying significant heads
5. Generating visualizations and reports

Usage:
    from general_analysis import run_path_patching_analysis
    
    results = run_path_patching_analysis(
        dataset_type="code",
        dataset_params={"var_type": "objects"},
        n_examples=100
    )
"""

import torch
from transformer_lens import HookedTransformer
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import json
import sys

sys.path.insert(0, str(Path(__file__).parent))

from plotting import save_path_patching_heatmap, plot_top_heads, print_top_heads
from utils import set_seed


class PathPatchingAnalyzer:
    """General path patching analyzer for any dataset."""
    
    def __init__(
        self,
        model: HookedTransformer,
        clean_tokens: torch.Tensor,
        corrupt_tokens: torch.Tensor,
        correct_toks: torch.Tensor,
        incorrect_toks: torch.Tensor,
        task_name: str = "task",
        pairs: Optional[List[Dict]] = None,
    ):
        """
        Initialize analyzer with dataset.
        
        Args:
            model: HookedTransformer model
            clean_tokens: Clean input tokens [batch, seq_len]
            corrupt_tokens: Corrupt input tokens [batch, seq_len]
            correct_toks: Correct answer token IDs [batch]
            incorrect_toks: Incorrect answer token IDs [batch]
            task_name: Name of the task for reporting
            pairs: Optional list of example pairs for reference
        """
        self.model = model
        self.clean_tokens = clean_tokens
        self.corrupt_tokens = corrupt_tokens
        self.correct_toks = correct_toks
        self.incorrect_toks = incorrect_toks
        self.task_name = task_name
        self.device = model.cfg.device
        self.pairs = pairs
        
    def validate_task(self, min_accuracy: float = 0.5) -> Dict[str, float]:
        """
        Validate that the model can solve the task on clean data.
        
        Args:
            min_accuracy: Minimum accuracy threshold to consider task solvable
            
        Returns:
            Dictionary with performance metrics
        """
        print("\n" + "="*80)
        print("TASK VALIDATION")
        print("="*80)
        
        batch_idx = torch.arange(len(self.correct_toks), device=self.device)
        
        with torch.inference_mode():
            # Clean performance
            clean_logits = self.model(self.clean_tokens)[:, -1, :]
            clean_correct = clean_logits[batch_idx, self.correct_toks]
            clean_incorrect = clean_logits[batch_idx, self.incorrect_toks]
            clean_diff = (clean_correct - clean_incorrect).mean().item()
            
            # Accuracy: is correct token the top prediction?
            clean_preds = clean_logits.argmax(dim=-1)
            clean_acc = (clean_preds == self.correct_toks).float().mean().item()
            
            # Top-5 accuracy
            top5_preds = torch.topk(clean_logits, k=5, dim=-1).indices
            clean_top5_acc = torch.any(
                top5_preds == self.correct_toks.unsqueeze(-1), dim=-1
            ).float().mean().item()
            
            # Corrupt performance
            corrupt_logits = self.model(self.corrupt_tokens)[:, -1, :]
            corrupt_correct = corrupt_logits[batch_idx, self.correct_toks]
            corrupt_incorrect = corrupt_logits[batch_idx, self.incorrect_toks]
            corrupt_diff = (corrupt_correct - corrupt_incorrect).mean().item()
            
            corrupt_preds = corrupt_logits.argmax(dim=-1)
            corrupt_acc = (corrupt_preds == self.correct_toks).float().mean().item()
        
        stats = {
            'clean_logit_diff': clean_diff,
            'corrupt_logit_diff': corrupt_diff,
            'clean_accuracy': clean_acc,
            'clean_top5_accuracy': clean_top5_acc,
            'corrupt_accuracy': corrupt_acc,
            'corruption_effect': clean_diff - corrupt_diff,
        }
        
        # Report results
        print(f"\n📊 Task: {self.task_name}")
        print(f"   Examples: {len(self.correct_toks)}")
        
        print(f"\n✓ Clean Performance:")
        print(f"   Logit difference: {stats['clean_logit_diff']:>7.3f}")
        print(f"   Top-1 Accuracy:   {stats['clean_accuracy']*100:>6.1f}%")
        print(f"   Top-5 Accuracy:   {stats['clean_top5_accuracy']*100:>6.1f}%")
        
        print(f"\n✗ Corrupt Performance:")
        print(f"   Logit difference: {stats['corrupt_logit_diff']:>7.3f}")
        print(f"   Accuracy:         {stats['corrupt_accuracy']*100:>6.1f}%")
        
        print(f"\n🔄 Corruption Effect:")
        print(f"   Logit diff change:{stats['corruption_effect']:>7.3f}")
        
        # Validation verdict
        print(f"\n{'='*80}")
        if stats['clean_accuracy'] >= 0.9:
            print("✅ EXCELLENT: Model solves this task very well (≥90% accuracy)")
            verdict = "excellent"
        elif stats['clean_accuracy'] >= 0.7:
            print("✅ GOOD: Model solves this task well (≥70% accuracy)")
            verdict = "good"
        elif stats['clean_accuracy'] >= min_accuracy:
            print("⚠️  MODERATE: Model partially solves this task")
            verdict = "moderate"
        else:
            print("❌ POOR: Model does not solve this task well")
            print(f"   Accuracy {stats['clean_accuracy']*100:.1f}% < minimum threshold {min_accuracy*100:.1f}%")
            verdict = "poor"
        
        if stats['corruption_effect'] < 0.5:
            print("⚠️  WARNING: Small corruption effect - clean/corrupt distinction may be weak")
        
        stats['verdict'] = verdict
        return stats
    
    def path_patch_head_to_logits(
        self,
        layer: int,
        head: int,
    ) -> float:
        """
        Path patch from a single head to logits.
        
        Measures direct effect of head on the logit difference by:
        1. Caching head output on clean input
        2. Running corrupt input with patched clean head output
        3. Measuring change in logit difference
        
        Returns:
            Effect on logit difference (positive = head helps task)
        """
        batch_idx = torch.arange(len(self.correct_toks), device=self.device)
        
        # Cache clean head output
        cache = {}
        hook_z_name = f"blocks.{layer}.attn.hook_z"
        
        def cache_clean_z(activation, hook):
            cache['z'] = activation[:, :, head, :].clone()
            return activation
        
        with torch.inference_mode():
            self.model.run_with_hooks(
                self.clean_tokens,
                fwd_hooks=[(hook_z_name, cache_clean_z)]
            )
        
        clean_z = cache['z'].detach()
        
        # Get corrupt baseline
        with torch.inference_mode():
            corrupt_logits = self.model(self.corrupt_tokens)[:, -1, :]
            corrupt_correct = corrupt_logits[batch_idx, self.correct_toks]
            corrupt_incorrect = corrupt_logits[batch_idx, self.incorrect_toks]
            corrupt_diff = (corrupt_correct - corrupt_incorrect).mean().item()
        
        # Run corrupt with patched head
        def patch_z(activation, hook):
            activation[:, :, head, :] = clean_z
            return activation
        
        with torch.inference_mode():
            patched_logits = self.model.run_with_hooks(
                self.corrupt_tokens,
                fwd_hooks=[(hook_z_name, patch_z)]
            )[:, -1, :]
            
            patched_correct = patched_logits[batch_idx, self.correct_toks]
            patched_incorrect = patched_logits[batch_idx, self.incorrect_toks]
            patched_diff = (patched_correct - patched_incorrect).mean().item()
        
        effect = patched_diff - corrupt_diff
        return effect
    
    def compute_all_effects(self) -> torch.Tensor:
        """
        Compute path patching effects for all attention heads.
        
        Returns:
            Tensor of shape [n_layers, n_heads] with effects
        """
        n_layers = self.model.cfg.n_layers
        n_heads = self.model.cfg.n_heads
        
        effects = torch.zeros(n_layers, n_heads)
        
        print("\n" + "="*80)
        print("PATH PATCHING ANALYSIS")
        print("="*80)
        print(f"\nComputing Head → Logits effects for {n_layers} layers × {n_heads} heads...")
        
        for layer in range(n_layers):
            print(f"Layer {layer:2d}...", end=" ", flush=True)
            for head in range(n_heads):
                effect = self.path_patch_head_to_logits(layer, head)
                effects[layer, head] = effect
            print("✓")
        
        return effects
    
    def identify_significant_heads(
        self,
        effects: torch.Tensor,
        threshold_percentile: float = 95,
        top_k: int = 20,
    ) -> Dict[str, List[Tuple[int, int, float]]]:
        """
        Identify significant heads based on effect magnitude.
        
        Args:
            effects: Tensor of shape [n_layers, n_heads]
            threshold_percentile: Percentile threshold for significance
            top_k: Number of top heads to return
            
        Returns:
            Dictionary with 'positive', 'negative', and 'all_significant' heads
        """
        print("\n" + "="*80)
        print("IDENTIFYING SIGNIFICANT HEADS")
        print("="*80)
        
        # Statistics
        print(f"\nEffect Statistics:")
        print(f"  Mean:   {effects.mean():.4f}")
        print(f"  Std:    {effects.std():.4f}")
        print(f"  Max:    {effects.max():.4f}")
        print(f"  Min:    {effects.min():.4f}")
        print(f"  Median: {effects.median():.4f}")
        
        # Significance threshold
        threshold = torch.quantile(torch.abs(effects), threshold_percentile / 100.0).item()
        significant_mask = torch.abs(effects) >= threshold
        n_significant = significant_mask.sum().item()
        
        print(f"\nSignificance Threshold ({threshold_percentile}th percentile):")
        print(f"  |Effect| ≥ {threshold:.4f}")
        print(f"  Significant heads: {n_significant} / {effects.numel()}")
        
        # Get top positive and negative heads
        n_layers, n_heads = effects.shape
        flat_effects = effects.flatten()
        
        # Top positive (help task)
        top_pos_indices = torch.topk(flat_effects, k=min(top_k, len(flat_effects))).indices
        positive_heads = [
            (idx.item() // n_heads, idx.item() % n_heads, flat_effects[idx].item())
            for idx in top_pos_indices
        ]
        
        # Top negative (hurt task)
        top_neg_indices = torch.topk(-flat_effects, k=min(top_k, len(flat_effects))).indices
        negative_heads = [
            (idx.item() // n_heads, idx.item() % n_heads, flat_effects[idx].item())
            for idx in top_neg_indices
        ]
        
        # All significant heads
        significant_indices = torch.where(significant_mask.flatten())[0]
        all_significant = [
            (idx.item() // n_heads, idx.item() % n_heads, flat_effects[idx].item())
            for idx in significant_indices
        ]
        all_significant.sort(key=lambda x: abs(x[2]), reverse=True)
        
        # Print results
        print_top_heads(effects, top_k=top_k, label=f"Top {top_k} Positive Effect Heads")
        print_top_heads(-effects, top_k=10, label="Top 10 Negative Effect Heads")
        
        return {
            'positive': positive_heads,
            'negative': negative_heads,
            'all_significant': all_significant,
            'threshold': threshold,
        }
    
    def save_results(
        self,
        effects: torch.Tensor,
        stats: Dict,
        significant_heads: Dict,
        output_dir: str,
    ):
        """Save all results to disk."""
        print("\n" + "="*80)
        print("SAVING RESULTS")
        print("="*80)
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save heatmap
        save_path_patching_heatmap(
            effects=effects,
            output_dir=output_dir,
            filename="direct_effect_heatmap.png",
            title=f"Direct Effect: Head → Logits ({self.task_name})",
            # vmin=-0.6,
            # vmax=0.6,
        )
        
        # Save top heads plot
        plot_top_heads(
            effects=effects,
            top_k=20,
            output_dir=output_dir,
            filename="top_heads.png",
            title=f"Top Heads by Effect ({self.task_name})"
        )
        
        # Save effects tensor
        torch.save(effects, output_path / "direct_effects.pt")
        print(f"Saved effects tensor to: {output_path / 'direct_effects.pt'}")
        
        # Get example sentences
        example = {}
        if self.pairs is not None and len(self.pairs) > 0:
            first_pair = self.pairs[0]
            
            # Decode tokens to get actual text
            clean_text = self.model.to_string(self.clean_tokens[0])
            corrupt_text = self.model.to_string(self.corrupt_tokens[0])
            correct_token = self.model.to_string(self.correct_toks[0])
            incorrect_token = self.model.to_string(self.incorrect_toks[0])
            
            example = {
                'clean_sentence': clean_text,
                'corrupt_sentence': corrupt_text,
                'correct_token': correct_token,
                'incorrect_token': incorrect_token,
                'raw_pair': first_pair  # Include original pair data
            }
        
        # Save summary
        summary = {
            'task_name': self.task_name,
            'n_examples': len(self.correct_toks),
            'example': example,
            'statistics': stats,
            'significant_heads': {
                'threshold': significant_heads['threshold'],
                'n_significant': len(significant_heads['all_significant']),
                'top_10_positive': [
                    {'layer': l, 'head': h, 'effect': float(e)}
                    for l, h, e in significant_heads['positive'][:10]
                ],
                'top_10_negative': [
                    {'layer': l, 'head': h, 'effect': float(e)}
                    for l, h, e in significant_heads['negative'][:10]
                ],
            }
        }
        
        with open(output_path / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Saved summary to: {output_path / 'summary.json'}")
        
        print(f"\n✅ All results saved to: {output_dir}/")


def load_dataset(
    dataset_type: str,
    model: HookedTransformer,
    dataset_params: Optional[Dict] = None,
    n_examples: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, List[Dict]]:
    """
    Load dataset of specified type.
    
    Args:
        dataset_type: One of 'ioi', 'color', 'code'
        model: HookedTransformer model
        dataset_params: Additional parameters for dataset loading
        n_examples: Number of examples to use
        
    Returns:
        Tuple of (clean_tokens, corrupt_tokens, correct_toks, incorrect_toks, task_name, pairs)
    """
    dataset_params = dataset_params or {}
    
    if dataset_type == "ioi":
        from data_loader import load_dataset_for_patching
        size = dataset_params.get('size', 'large')
        clean, corrupt, correct, incorrect, pairs = load_dataset_for_patching(
            model, size=size, n_examples=n_examples
        )
        task_name = f"IOI ({size})"
        
    elif dataset_type == "color":
        from data_loader import load_dataset_for_patching
        size = dataset_params.get('size', 'small')
        clean, corrupt, correct, incorrect, pairs = load_dataset_for_patching(
            model, size=size, n_examples=n_examples
        )
        task_name = f"Color-Object ({size})"
        
    elif dataset_type == "code":
        from data_loader_code import load_code_dataset_for_patching
        var_type = dataset_params.get('var_type', 'objects')
        clean, corrupt, correct, incorrect, pairs = load_code_dataset_for_patching(
            model, var_type=var_type, n_examples=n_examples
        )
        task_name = f"Code ({var_type})"
        
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    return clean, corrupt, correct, incorrect, task_name, pairs


def run_path_patching_analysis(
    dataset_type: str,
    dataset_params: Optional[Dict] = None,
    n_examples: int = 100,
    model_name: str = "gpt2-small",
    output_dir: Optional[str] = None,
    min_accuracy: float = 0.5,
    seed: int = 42,
) -> Dict:
    """
    Run complete path patching analysis on any dataset.
    
    Args:
        dataset_type: One of 'ioi', 'color', 'code'
        dataset_params: Additional parameters for dataset loading
        n_examples: Number of examples to use
        model_name: Name of model to load
        output_dir: Output directory (auto-generated if None)
        min_accuracy: Minimum accuracy to consider task solvable
        seed: Random seed
        
    Returns:
        Dictionary with all results
    """
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("="*80)
    print("GENERAL PATH PATCHING ANALYSIS")
    print("="*80)
    print(f"\nDataset: {dataset_type}")
    print(f"Model: {model_name}")
    print(f"Examples: {n_examples}")
    print(f"Device: {device}")
    
    # Load model
    print(f"\nLoading model...")
    model = HookedTransformer.from_pretrained(model_name, device=device)
    
    # Load dataset
    print(f"\nLoading dataset...")
    clean, corrupt, correct, incorrect, task_name, pairs = load_dataset(
        dataset_type, model, dataset_params, n_examples
    )
    
    # Create analyzer
    analyzer = PathPatchingAnalyzer(
        model=model,
        clean_tokens=clean,
        corrupt_tokens=corrupt,
        correct_toks=correct,
        incorrect_toks=incorrect,
        task_name=task_name,
        pairs=pairs,
    )
    
    # Validate task
    stats = analyzer.validate_task(min_accuracy=min_accuracy)
    
    # Check if task is solvable enough to continue
    if stats['verdict'] == 'poor':
        print("\n⚠️  Task validation failed. Stopping analysis.")
        print("   The model does not solve this task well enough for meaningful path patching.")
        return {'status': 'failed', 'stats': stats}
    
    # Run path patching
    effects = analyzer.compute_all_effects()
    
    # Identify significant heads
    significant_heads = analyzer.identify_significant_heads(effects, top_k=20)
    
    # Save results
    if output_dir is None:
        output_dir = f"results/{dataset_type}_{n_examples}_examples"
    
    analyzer.save_results(effects, stats, significant_heads, output_dir)
    
    return {
        'status': 'success',
        'stats': stats,
        'effects': effects,
        'significant_heads': significant_heads,
        'output_dir': output_dir,
    }


if __name__ == "__main__":
    # Example: Run on code completion task
    results = run_path_patching_analysis(
        dataset_type="code",
        dataset_params={"var_type": "objects"},
        n_examples=200,
        output_dir="results/general_analysis_code_objects"
    )
    
    if results['status'] == 'success':
        print("\n" + "="*80)
        print("✅ ANALYSIS COMPLETE")
        print("="*80)
