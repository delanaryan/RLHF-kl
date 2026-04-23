"""
Experiment Runner and Comparison Utilities
"""

from pathlib import Path
from typing import Dict, Optional
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import plot

class ExperimentRunner:
    """Utility for running and managing experiments"""
    @staticmethod
    def build_reference_responses(prompts, model_fn):
        reference_responses = []

        for prompt_row in prompts[1:]:
            prompt_text = prompt_row[1]

            response = model_fn(prompt_text)
            reference_responses.append(response)

        return reference_responses

    @staticmethod
    def run_fixed_beta_sweep(num_prompts: int = 5, generations: int = 5, reference=None) -> Path:
        """
        Run fixed-β sweep experiment.

        Args:  num_prompts — number of prompts to use
                generations — generations per prompt
        Returns: Path to results directory
        """
        print("\n" + "=" * 70)
        print("LAUNCHING FIXED-β SWEEP EXPERIMENT")
        print("=" * 70)

        from fixed_beta_sweep import FixedBetaSweepExperiment
        exp = FixedBetaSweepExperiment()
        exp.run_sweep(num_prompts=num_prompts, generations_per_prompt=generations, reference=reference)
        return exp.output_dir

    @staticmethod
    def run_adaptive_controller(num_prompts: int = 5, num_steps: int = 10, reference=None) -> Path:
        """
        Run adaptive β controller experiment.

        Args:  num_prompts — number of prompts to use
                num_steps — number of optimization steps
        Returns: Path to results directory
        """
        print("\n" + "=" * 70)
        print("LAUNCHING ADAPTIVE β CONTROLLER EXPERIMENT")
        print("=" * 70)

        from adaptive_beta_controller import AdaptiveOptimizationExperiment
        exp = AdaptiveOptimizationExperiment()
        exp.run_experiment(num_prompts=num_prompts, num_steps=num_steps, reference=reference)
        return exp.output_dir

class ResultsComparator:

    @staticmethod
    def load_fixed_beta_results(results_dir: Path) -> Dict[float, pd.DataFrame]:
        """
        Load individual beta result CSVs from the sweep results directory.

        Returns: dict mapping beta float → per-prompt DataFrame
        """
        results = {}
        for file in results_dir.glob("beta_*.csv"):
            beta_str = file.stem.replace("beta_", "")
            if beta_str.endswith("_results"):
                beta_str = beta_str[: -len("_results")]
            results[float(beta_str)] = pd.read_csv(file)
        return dict(sorted(results.items()))

    @staticmethod
    def load_sweep_summary(results_dir: Path) -> pd.DataFrame:
        """
        Load the pre-aggregated sweep_summary.csv produced by FixedBetaSweepExperiment.

        Returns: DataFrame with columns: beta, avg_sentiment_score, avg_kl_divergence,
                avg_reward, repetitive_responses, num_prompts
        """
        summary_file = results_dir / "sweep_summary.csv"
        if not summary_file.exists():
            raise FileNotFoundError(f"sweep_summary.csv not found in {results_dir}")
        return pd.read_csv(summary_file)

    @staticmethod
    def load_adaptive_history(results_dir: Path) -> Optional[pd.DataFrame]:
        """
        Load the optimization history CSV from the adaptive controller run.

        Returns: DataFrame with columns: step, beta, kl_divergence,
                sentiment_score, reward, beta_action
        """
        history_file = results_dir / "adaptive_optimization_history.csv"
        if history_file.exists():
            return pd.read_csv(history_file)
        return None

    @staticmethod
    def build_comparison_df(
            fixed_results: Dict[float, pd.DataFrame],
            adaptive_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Aggregate per-beta metrics into a single comparison DataFrame.
        One row per approach (fixed betas + optional adaptive).

        Returns: DataFrame with columns:
                approach, beta, avg_sentiment, avg_kl, avg_reward, type
        """
        rows = []

        for beta, df in fixed_results.items():
            rows.append({
                'approach':      f'Fixed β={beta}',
                'beta':          beta,
                'avg_sentiment': df['top_sentiment'].mean(),
                'avg_kl':        df['top_kl'].mean(),
                'avg_reward':    df['top_reward'].mean(),
                'type':          'fixed',
            })

        if adaptive_df is not None:
            rows.append({
                'approach':      f'Adaptive (final β={adaptive_df["beta"].iloc[-1]:.3f})',
                'beta':          adaptive_df['beta'].iloc[-1],
                'avg_sentiment': adaptive_df['sentiment_score'].mean(),
                'avg_kl':        adaptive_df['kl_divergence'].mean(),
                'avg_reward':    adaptive_df['reward'].mean(),
                'type':          'adaptive',
            })

        return pd.DataFrame(rows)

    @staticmethod
    def print_comparison_report(comparison_df: pd.DataFrame):
        """
        Print a text report identifying best approach by reward and by alignment.

        Args: comparison_df from build_comparison_df()
        """
        print("\n" + "=" * 70)
        print("EXPERIMENTAL COMPARISON REPORT")
        print("=" * 70)
        print("\n" + comparison_df.to_string(index=False))

        best       = comparison_df.loc[comparison_df['avg_reward'].idxmax()]
        best_align = comparison_df.loc[comparison_df['avg_kl'].idxmin()]

        print(f"\n✓ Best approach by reward: {best['approach']}")
        print(f"  - Sentiment : {best['avg_sentiment']:.4f}")
        print(f"  - KL        : {best['avg_kl']:.4f}")
        print(f"  - Reward    : {best['avg_reward']:.4f}")

        print(f"\n✓ Best model alignment (lowest KL): {best_align['approach']}")
        print(f"  - Sentiment : {best_align['avg_sentiment']:.4f}")
        print(f"  - KL        : {best_align['avg_kl']:.4f}")
        print(f"  - Reward    : {best_align['avg_reward']:.4f}")

        print("\n" + "=" * 70)

def run_full_comparison(num_prompts: int = 5):
    """
    Run both experiments then generate the full comparison report and plots.

    Args: num_prompts — number of prompts to use in both experiments
    """
    print("\n" + "=" * 70)
    print("FULL EXPERIMENTAL COMPARISON PROTOCOL")
    print("=" * 70)

    from src import utils, generate
    import config
    promptArr = utils.csvToArr(config.PROMPT_PATH)
    prompts   = [promptArr[0]] + promptArr[1:num_prompts + 1]
    runner = ExperimentRunner()
    reference_responses = runner.build_reference_responses(prompts, generate.generateSingleResponse)

    fixed_dir = runner.run_fixed_beta_sweep(num_prompts=num_prompts, generations=5, reference=reference_responses)
    adaptive_dir = runner.run_adaptive_controller(num_prompts=num_prompts, num_steps=10, reference=reference_responses)

    comparator = ResultsComparator()
    fixed_results = comparator.load_fixed_beta_results(fixed_dir)
    sweep_summary = comparator.load_sweep_summary(fixed_dir)
    adaptive_hist = comparator.load_adaptive_history(adaptive_dir)
    comparison_df = comparator.build_comparison_df(fixed_results, adaptive_hist)

    comparator.print_comparison_report(comparison_df)

    # Plots generated by plot.py
    plot.plotComparisonSummary(comparison_df)
    plot.plotFixedBetaComparison(sweep_summary)
    plot.plotRepetitionRate(sweep_summary)

    if adaptive_hist is not None:
        plot.plotAdaptiveBetaTrace(adaptive_hist)
        plot.plotAdaptiveVsFixed(adaptive_hist, sweep_summary, best_fixed_beta=0.1)

    print("\n" + "=" * 70)
    print("COMPARISON COMPLETE")
    print(f"  Fixed-β results  : {fixed_dir}")
    print(f"  Adaptive results : {adaptive_dir}")
    print("=" * 70)


if __name__ == "__main__":
    run_full_comparison(num_prompts=3)