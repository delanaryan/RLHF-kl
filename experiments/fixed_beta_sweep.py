"""
Fixed-β Sweeps Experiment
Runs the optimization loop with Fixed KL Penalties to observe where the model starts "hacking"
(e.g., repeating "Happy" to get high scores).

This experiment sweeps through different β values (0.01, 0.1, 0.5, 1.0) and monitors:
- Sentiment score trends
- KL divergence from base model
- Response diversity/repetition patterns
- Degradation curves
"""

import pandas as pd
import csv
import math
from typing import List, Dict, Tuple
from pathlib import Path
import sys
import os

# parent directory path to import src modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import generate, score, utils
import config


class FixedBetaSweepExperiment:
    """Runs optimization with fixed KL penalties to identify model hacking behavior"""

    def __init__(self, output_dir: str = "experiments/results/fixed_beta_sweep"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.beta_values = [0.01, 0.1, 0.5, 1.0]
        self.results = {}

    def run_optimization_loop(self, beta: float) -> Dict:
        """
        Run a single optimization loop with fixed β value.
        
        Args:
            beta: Fixed KL penalty coefficient
            
        Returns:
            Dictionary with results for this β value
        """
        results = {
            'beta': beta,
            'prompts_data': []
        }

        all_responses = []

        best_candidates = generate.getAllBestOfN(beta)
        for candidate in best_candidates:
            prompt_id = candidate[0]
            response = candidate[3]
            sentiment = candidate[4]
            n = candidate[1]
            kl_div = candidate[6]
            reward = candidate[7]
            perplexity = candidate[5]

            results['prompts_data'].append({
                'prompt_id': prompt_id,
                'response': response,
                'sentiment': sentiment,
                'perplexity': perplexity,
                'kl_divergence': kl_div,
                'reward': reward,
                'beta': beta,
                'N': n
            })

            all_responses.append(response)

        return results

    def detect_hacking_behavior(self, results: Dict) -> Dict:
        """
        Analyze results to detect potential model hacking (e.g., repetition, shortcuts).
        
        Returns:
            Dictionary with hacking metrics
        """
        hacking_metrics = {
            'beta': results['beta'],
            'high_sentiment_low_diversity': 0,
            'repetitive_responses': 0,
            'reward_vs_sentiment': [],
        }

        for prompt_data in results['prompts_data']:
            top_response = prompt_data['response']
            top_sentiment = prompt_data['sentiment']
            top_kl = prompt_data['kl_divergence']

            # Check for repetition in top response
            words = top_response.lower().split()
            if len(words) > 0:
                unique_ratio = len(set(words)) / len(words)
                if unique_ratio < 0.8:  # More than 50% repeated words
                    hacking_metrics['repetitive_responses'] += 1

            if top_sentiment > 0.8 and top_kl < 0.1: # High sentiment but low diversity could indicate hacking
                hacking_metrics['high_sentiment_low_diversity'] += 1

            hacking_metrics['reward_vs_sentiment'].append({
                'sentiment': top_sentiment,
                'kl': top_kl,
                'reward': prompt_data['reward']
            })

        return hacking_metrics

    def save_results(self, results: Dict, beta: float):
        """Save results for a single β value to CSV"""
        output_file = self.output_dir / f"beta_{beta:.2f}_results.csv"

        rows = []
        for prompt_data in results['prompts_data']:
            rows.append({
                'prompt_id':    prompt_data['prompt_id'],
                'top_response': prompt_data['response'],
                'top_sentiment':prompt_data['sentiment'],
                'top_perplexity':prompt_data['perplexity'],
                'top_kl':       prompt_data['kl_divergence'],
                'top_reward':   prompt_data['reward'],
                'beta': prompt_data['beta'],
                'n': prompt_data['n']
            })

        df = pd.DataFrame(rows)
        df.to_csv(output_file, index=False)
        print(f"\nSaved results for β={beta} to {output_file}")

    def save_summary(self):
        """Save comprehensive summary of all β sweep results"""
        summary_file = self.output_dir / "sweep_summary.csv"

        summary_rows = []
        for beta, results in self.results.items():
            hacking = self.detect_hacking_behavior(results)

            avg_sent   = sum(p['sentiment']    for p in results['prompts_data']) / len(results['prompts_data'])
            avg_kl     = sum(p['kl_divergence'] for p in results['prompts_data']) / len(results['prompts_data'])
            avg_reward = sum(p['reward']        for p in results['prompts_data']) / len(results['prompts_data'])

            summary_rows.append({
                'beta': beta,
                'avg_sentiment_score': avg_sent,
                'avg_kl_divergence': avg_kl,
                'avg_reward': avg_reward,
                'repetitive_responses': hacking['repetitive_responses'],
                'high_sentiment_low_diversity': hacking['high_sentiment_low_diversity'],
                'num_prompts': len(results['prompts_data']),
            })

        df = pd.DataFrame(summary_rows)
        df.to_csv(summary_file, index=False)
        print(f"\nSaved sweep summary to {summary_file}")
        print("\n=== FIXED-β SWEEP SUMMARY ===")
        print(df.to_string(index=False))

    def run_sweep(self):
        """
        Execute the full fixed-β sweep experiment.
        """
        print("=" * 60)
        print("FIXED-β SWEEP EXPERIMENT")
        print("=" * 60)

        print(f"Testing β values: {self.beta_values}")
        print()

        for beta in self.beta_values:
            print(f"\n{'='*60}")
            print(f"Running optimization with β={beta}")
            print(f"{'='*60}")

            results = self.run_optimization_loop(beta)
            self.results[beta] = results
            self.save_results(results, beta)

        self.save_summary()

        print("\n" + "=" * 60)
        print("EXPERIMENT COMPLETE")
        print(f"Results saved to: {self.output_dir}")
        print("=" * 60)


if __name__ == "__main__":
    experiment = FixedBetaSweepExperiment()

    # Run with fewer prompts for testing (adjust for full experiment)
    experiment.run_sweep()