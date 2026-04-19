"""
Experiment Runner and Comparison Utilities
Provides tools to run both experiments and compare results
"""

import subprocess
import sys
import os
from pathlib import Path
from typing import Dict, List
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class ExperimentRunner:
    """Utility for running and managing experiments"""
    
    @staticmethod
    def run_fixed_beta_sweep(num_prompts: int = 5, generations: int = 5) -> Path:
        """
        Run fixed-β sweep experiment
        
        Args:
            num_prompts: Number of prompts to use
            generations: Generations per prompt
            
        Returns:
            Path to results directory
        """
        print("\n" + "="*70)
        print("LAUNCHING FIXED-β SWEEP EXPERIMENT")
        print("="*70)
        
        try:
            from fixed_beta_sweep import FixedBetaSweepExperiment
            exp = FixedBetaSweepExperiment()
            exp.run_sweep(num_prompts=num_prompts, generations_per_prompt=generations)
            return exp.output_dir
        except Exception as e:
            print(f"Error running fixed-β sweep: {e}")
            raise
    
    @staticmethod
    def run_adaptive_controller(num_prompts: int = 5, num_steps: int = 10) -> Path:
        """
        Run adaptive β controller experiment
        
        Args:
            num_prompts: Number of prompts to use
            num_steps: Number of optimization steps
            
        Returns:
            Path to results directory
        """
        print("\n" + "="*70)
        print("LAUNCHING ADAPTIVE β CONTROLLER EXPERIMENT")
        print("="*70)
        
        try:
            from adaptive_beta_controller import AdaptiveOptimizationExperiment
            exp = AdaptiveOptimizationExperiment()
            exp.run_experiment(num_prompts=num_prompts, num_steps=num_steps)
            return exp.output_dir
        except Exception as e:
            print(f"Error running adaptive controller: {e}")
            raise


class ResultsComparator:
    """Utilities for comparing experimental results"""
    
    @staticmethod
    def load_fixed_beta_results(results_dir: Path) -> Dict[float, pd.DataFrame]:
        """Load and aggregate fixed-β sweep results"""
        results = {}
        
        for file in results_dir.glob("beta_*.csv"):
            # Extract β value from filename. Support names like beta_0.10_results.csv.
            beta_str = file.stem.replace("beta_", "")
            if beta_str.endswith("_results"):
                beta_str = beta_str[: -len("_results")]
            beta_val = float(beta_str)
            results[beta_val] = pd.read_csv(file)
        
        return dict(sorted(results.items()))
    
    @staticmethod
    def load_adaptive_history(results_dir: Path) -> pd.DataFrame:
        """Load adaptive controller optimization history"""
        history_file = results_dir / "adaptive_optimization_history.csv"
        if history_file.exists():
            return pd.read_csv(history_file)
        return None
    
    @staticmethod
    def compare_reward_sentiment_tradeoff(fixed_results: Dict[float, pd.DataFrame], adaptive_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Compare sentiment vs. reward across different approaches
        
        Returns:
            DataFrame with comparison metrics
        """
        comparison_data = []
        
        for beta, df in fixed_results.items():
            avg_sentiment = df['top_sentiment'].mean()
            avg_kl = df['top_kl'].mean()
            avg_reward = df['top_reward'].mean()
            
            comparison_data.append({
                'approach': f'Fixed β={beta}',
                'beta': beta,
                'avg_sentiment': avg_sentiment,
                'avg_kl': avg_kl,
                'avg_reward': avg_reward,
                'type': 'fixed',
            })
        
        if adaptive_df is not None:
            final_beta = adaptive_df['beta'].iloc[-1]
            avg_sentiment = adaptive_df['sentiment_score'].mean()
            avg_kl = adaptive_df['kl_divergence'].mean()
            avg_reward = adaptive_df['reward'].mean()
            
            comparison_data.append({
                'approach': f'Adaptive (final β={final_beta:.3f})',
                'beta': final_beta,
                'avg_sentiment': avg_sentiment,
                'avg_kl': avg_kl,
                'avg_reward': avg_reward,
                'type': 'adaptive',
            })
        
        return pd.DataFrame(comparison_data)
    
    @staticmethod
    def plot_comparison(comparison_df: pd.DataFrame, output_path: Path = None):
        """
        Plot comparison of approaches
        
        Args:
            comparison_df: DataFrame from compare_reward_sentiment_tradeoff
            output_path: Path to save plot (optional)
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Plot 1: Sentiment vs KL
        fixed_data = comparison_df[comparison_df['type'] == 'fixed']
        adaptive_data = comparison_df[comparison_df['type'] == 'adaptive']
        
        axes[0].scatter(fixed_data['avg_kl'], fixed_data['avg_sentiment'], s=100, alpha=0.6, label='Fixed β', color='blue')
        if not adaptive_data.empty: axes[0].scatter(adaptive_data['avg_kl'], adaptive_data['avg_sentiment'], s=200, marker='*', label='Adaptive β', color='red')
        
        axes[0].set_xlabel('Average KL Divergence')
        axes[0].set_ylabel('Average Sentiment Score')
        axes[0].set_title('Sentiment vs. KL Divergence')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Reward across approaches
        approaches = comparison_df['approach']
        rewards = comparison_df['avg_reward']
        
        colors = ['blue' if t == 'fixed' else 'red' for t in comparison_df['type']]
        axes[1].bar(range(len(approaches)), rewards, color=colors, alpha=0.6)
        axes[1].set_xticks(range(len(approaches)))
        axes[1].set_xticklabels(approaches, rotation=45, ha='right')
        axes[1].set_ylabel('Average Reward')
        axes[1].set_title('RLHF Reward Comparison')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # Plot 3: β values
        axes[2].scatter(fixed_data['beta'], fixed_data['avg_reward'], s=100, alpha=0.6, label='Fixed β', color='blue')
        if not adaptive_data.empty: axes[2].scatter(adaptive_data['beta'], adaptive_data['avg_reward'], s=200, marker='*', label='Adaptive β', color='red')
        
        axes[2].set_xlabel('β Value')
        axes[2].set_ylabel('Average Reward')
        axes[2].set_title('Reward vs. β Coefficient')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        axes[2].set_xscale('log')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"\nPlot saved to {output_path}")
        
        plt.show()
    
    @staticmethod
    def print_comparison_report(comparison_df: pd.DataFrame):
        """Print detailed comparison report"""
        print("\n" + "="*70)
        print("EXPERIMENTAL COMPARISON REPORT")
        print("="*70)
        
        print("\n" + comparison_df.to_string(index=False))
        
        # Find best approach by reward
        best_idx = comparison_df['avg_reward'].idxmax()
        best = comparison_df.loc[best_idx]
        
        print(f"\n✓ Best approach by reward: {best['approach']}")
        print(f"  - Sentiment: {best['avg_sentiment']:.4f}")
        print(f"  - KL: {best['avg_kl']:.4f}")
        print(f"  - Reward: {best['avg_reward']:.4f}")
        
        # Find best alignment (low KL)
        best_align_idx = comparison_df['avg_kl'].idxmin()
        best_align = comparison_df.loc[best_align_idx]
        
        print(f"\n✓ Best model alignment (low KL): {best_align['approach']}")
        print(f"  - Sentiment: {best_align['avg_sentiment']:.4f}")
        print(f"  - KL: {best_align['avg_kl']:.4f}")
        print(f"  - Reward: {best_align['avg_reward']:.4f}")
        
        print("\n" + "="*70)


def run_full_comparison(num_prompts: int = 5):
    """
    Run both experiments and generate comparison report
    
    Args:
        num_prompts: Number of prompts to use in experiments
    """
    print("\n" + "="*70)
    print("FULL EXPERIMENTAL COMPARISON PROTOCOL")
    print("="*70)
    
    runner = ExperimentRunner()
    
    fixed_dir = runner.run_fixed_beta_sweep(num_prompts=num_prompts, generations=5)
    adaptive_dir = runner.run_adaptive_controller(num_prompts=num_prompts, num_steps=10)
    comparator = ResultsComparator()
    fixed_results = comparator.load_fixed_beta_results(fixed_dir)
    adaptive_history = comparator.load_adaptive_history(adaptive_dir)
    
    comparison_df = comparator.compare_reward_sentiment_tradeoff(
        fixed_results,
        adaptive_history
    )
    
    comparator.print_comparison_report(comparison_df)

    output_plot = fixed_dir.parent / "experiment_comparison.png"
    comparator.plot_comparison(comparison_df, output_plot)
    
    print("\n" + "="*70)
    print("COMPARISON COMPLETE")
    print(f"Results saved to:")
    print(f"  Fixed-β: {fixed_dir}")
    print(f"  Adaptive: {adaptive_dir}")
    print(f"  Comparison plot: {output_plot}")
    print("="*70)


if __name__ == "__main__":
    # Run full comparison with default settings
    run_full_comparison(num_prompts=3)
