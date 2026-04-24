"""
Adaptive β Controller Implementation
Implements threshold-based dynamic β adjustment that monitors KL divergence
and "tightens/loosens the leash" to keep the model aligned with the base distribution.

The controller uses a step function approach:
- If KL divergence drifts too far above target: increase β (tighten constraint)
- If KL divergence is too low: decrease β (loosen constraint)
"""

import pandas as pd
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass
import sys
import os
import numpy as np

# Add parent directory to path to import src modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import generate, score, utils
import config


@dataclass
class AdaptiveControllerConfig:
    """Configuration for the Adaptive β Controller"""
    initial_beta: float = 0.01
    target_kl: float = 0.5  # Target KL divergence threshold
    kl_upper_threshold: float = 0.70
    kl_lower_threshold: float = 0.45
    beta_increase_factor: float = 1.2  # Tighten leash
    beta_decrease_factor: float = 0.8  # Loosen leash
    beta_min: float = 0.01  # Minimum β to prevent over-loosening
    beta_max: float = 2.0   # Maximum β to prevent over-tightening
    batch_size: int = 5     # Number of generations per batch


class AdaptiveKLController:
    """
    Adaptive β controller that adjusts KL penalty dynamically based on actual KL divergence.

    This implements a feedback control system that monitors KL divergence and adjusts β
    to keep the model's outputs close to the base model distribution.
    """

    def __init__(self, config: AdaptiveControllerConfig = None):
        self.config = config or AdaptiveControllerConfig()
        self.beta = self.config.initial_beta
        self.history = {
            'step': [],
            'beta': [],
            'kl_divergence': [],
            'sentiment_score': [],
            'reward': [],
            'beta_action': [],  # 'no_change', 'increase', 'decrease'
        }
        self.all_reference_responses = []

    def normalize_kl(self, kl_value: float) -> float:
        """
        Maps KL [0, inf) to [0, 1] using exponential squashing.
        This ensures 1.0 is the theoretical maximum penalty.
        """ 
        # to change how 'aggressive' the normalization is, add scale factor in exp
        return 1 - np.exp(-kl_value)


    def adjust_beta(self, current_kl: float) -> Tuple[float, str]:
        """
        Threshold-based step function for β adjustment.

        Step function logic:
        - If current_kl > target_kl * 1.5: β = β * 1.2 (tighten leash)
        - If current_kl < target_kl * 0.5: β = β * 0.8 (loosen leash)
        - Otherwise: keep β unchanged

        Input: current_kl: Measured KL divergence from batch
        Returns: Tuple of (new_beta, action_taken)
        """
        current_kl = current_kl
        old_beta = self.beta
        action = 'no_change'

        if current_kl > self.config.kl_upper_threshold:
            self.beta = self.beta * self.config.beta_increase_factor
            action = 'increase'
        elif current_kl < self.config.kl_lower_threshold:
            self.beta = self.beta * self.config.beta_decrease_factor
            action = 'decrease'

        self.beta = max(self.config.beta_min, min(self.config.beta_max, self.beta)) # Ensure β stays within bounds

        return self.beta, action


    def process_batch(self, batch_responses: List[str], batch_sentiments: List[float], batch_rewards: List[float], step_num: int, kl: 0.0) -> Dict:
        """
        Process a batch of generated responses and adjust β accordingly.

        Inputs:
        batch_responses: Generated responses for this batch
        batch_sentiments: Sentiment scores for each response
        step_num: Current optimization step
        kl: Average kl for this batch

        Returns: Dictionary with batch results and controller state
        """

        #urrent_kl = kl
        norm_kl = self.normalize_kl(0.5*kl)
        new_beta, action = self.adjust_beta(norm_kl) # Adjust β based on KL divergence and get the action taken

        avg_sentiment = sum(float(x) for x in batch_sentiments) / len(batch_sentiments) if batch_sentiments else 0
        avg_reward = sum(float(x) for x in batch_rewards) / len(batch_rewards) if batch_rewards else 0

        # history
        self.history['step'].append(step_num)
        self.history['beta'].append(self.beta)
        self.history['kl_divergence'].append(norm_kl)
        self.history['sentiment_score'].append(avg_sentiment)
        self.history['reward'].append(avg_reward)
        self.history['beta_action'].append(action)

        self.all_reference_responses.extend(batch_responses)

        return {
            'step': step_num,
            'old_beta': new_beta / self.config.beta_increase_factor if action == 'increase' else (new_beta / self.config.beta_decrease_factor if action == 'decrease' else new_beta),
            'new_beta': self.beta,
            'current_kl': norm_kl,
            'target_kl': self.config.target_kl,
            'avg_sentiment': avg_sentiment,
            'avg_reward': avg_reward,
            'action': action,
            'batch_size': len(batch_responses),
        }

    def get_history_dataframe(self) -> pd.DataFrame:
        """Return optimization history as DataFrame"""
        return pd.DataFrame(self.history)

    def print_status(self, batch_result: Dict):
        """Pretty-print the controller status for a batch"""
        print(f"\n--- Step {batch_result['step']} ---")
        print(f"KL Divergence: {batch_result['current_kl']:.4f} (target: {batch_result['target_kl']:.4f})")
        print(f"β: {batch_result['old_beta']:.4f} → {batch_result['new_beta']:.4f} (action: {batch_result['action']})")
        print(f"Avg Sentiment: {batch_result['avg_sentiment']:.4f}")
        print(f"Avg Reward: {batch_result['avg_reward']:.4f}")


class AdaptiveOptimizationExperiment:
    """
    Full adaptive optimization experiment that uses the KL controller.
    """

    def __init__(self, config: Optional[AdaptiveControllerConfig] = None, output_dir: str = "experiments/results/adaptive_beta"):
        self.config = config or AdaptiveControllerConfig()
        self.controller = AdaptiveKLController(config)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_optimization(self, num_steps: int = 30) -> Dict:
        """
        Run adaptive optimization across multiple steps.

        Inputs:
        prompts: List of prompts to optimize on
        num_steps: Number of optimization steps
        generations_per_batch: Generations per batch per prompt

        Returns:
        Optimization results
        """
        print("=" * 70)
        print("ADAPTIVE β CONTROLLER EXPERIMENT")
        print("=" * 70)
        print(f"Initial β: {self.controller.beta}")
        print(f"Target KL: {self.config.target_kl}")
        print(f"Number of steps: {num_steps}")
        print("=" * 70)

        all_prompt_results = []
        beta = 0.01

        for step in range(num_steps):
            print(f"\n{'='*70}")
            print(f"OPTIMIZATION STEP {step + 1}/{num_steps}")
            print(f"{'='*70}")

            batch_responses = []
            batch_sentiments = []
            batch_rewards = []
            batch_kl = []

            best_candidates = generate.getAllBestOfN(beta)
            for candidate in best_candidates:
                prompt_id = candidate[0]
                response = candidate[3]
                sentiment = candidate[4]
                n = candidate[1]
                kl_div = candidate[6]
                reward = candidate[7]
                perplexity = candidate[5]

                batch_responses.append(response)
                batch_sentiments.append(sentiment)
                batch_rewards.append(reward)
                batch_kl.append(kl_div)

                all_prompt_results.append({
                    'step': step + 1,
                    'prompt_id': prompt_id,
                    'response': response,
                    'sentiment': sentiment,
                    'perplexity': perplexity,
                    'N': n,
                    'kl_divergence': kl_div,
                    'reward': reward
                })


            avg_kl = np.mean([float(x) for x in batch_kl])
            batch_result = self.controller.process_batch(
                batch_responses,
                batch_sentiments,
                batch_rewards,
                step + 1,
                avg_kl
            )
            beta = batch_result['new_beta']

            self.controller.print_status(batch_result)

        return {
            'prompt_results': all_prompt_results,
            'history': self.controller.get_history_dataframe(),
        }

    def save_results(self, results: Dict):
        """Save detailed results of adaptive optimization"""
        # Save prompt-level results
        prompt_df = pd.DataFrame(results['prompt_results'])
        prompt_file = self.output_dir / "adaptive_prompt_generations.csv"
        prompt_df.to_csv(prompt_file, index=False)
        print(f"\nSaved prompt generations to {prompt_file}")

        # Save optimization history
        history_df = results['history']
        history_file = self.output_dir / "adaptive_optimization_history.csv"
        history_df.to_csv(history_file, index=False)
        print(f"Saved optimization history to {history_file}")

        # Save summary statistics
        self._save_summary_statistics(history_df)

    def _save_summary_statistics(self, history_df: pd.DataFrame):
        """Generate and save summary statistics"""
        summary = {
            'Metric': [
                'Initial β',
                'Final β',
                'Min β',
                'Max β',
                'Average KL Divergence',
                'Final KL Divergence',
                'Average Sentiment',
                'Average Reward',
                'Times β Increased',
                'Times β Decreased',
            ],
            'Value': [
                f"{history_df['beta'].iloc[0]:.4f}",
                f"{history_df['beta'].iloc[-1]:.4f}",
                f"{history_df['beta'].min():.4f}",
                f"{history_df['beta'].max():.4f}",
                f"{history_df['kl_divergence'].mean():.4f}",
                f"{history_df['kl_divergence'].iloc[-1]:.4f}",
                f"{history_df['sentiment_score'].mean():.4f}",
                f"{history_df['reward'].mean():.4f}",
                f"{(history_df['beta_action'] == 'increase').sum()}",
                f"{(history_df['beta_action'] == 'decrease').sum()}",
            ]
        }

        summary_df = pd.DataFrame(summary)
        summary_file = self.output_dir / "adaptive_summary_statistics.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"Saved summary statistics to {summary_file}")
        print("\n=== OPTIMIZATION SUMMARY ===")
        print(summary_df.to_string(index=False))

    def run_experiment(self, num_steps: int = 10):
        """Execute the full adaptive optimization experiment"""

        results = self.run_optimization(num_steps=num_steps)
        self.save_results(results)

        print("\n" + "=" * 70)
        print("ADAPTIVE OPTIMIZATION EXPERIMENT COMPLETE")
        print(f"Results saved to: {self.output_dir}")
        print("=" * 70)


if __name__ == "__main__":
    config = AdaptiveControllerConfig(initial_beta=0.1, target_kl=0.5) # default config

    experiment = AdaptiveOptimizationExperiment(config=config)
    experiment.run_experiment(num_steps=8)