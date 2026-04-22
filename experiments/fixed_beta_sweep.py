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
        
    def compute_rlhf_reward(self, sentiment_score: float, kl_divergence: float, beta: float) -> float:
        """
        Compute RLHF reward with KL penalty.
        
        Reward = sentiment_score - β * KL_divergence
        where β controls the strength of the KL constraint
        
        Args:
            sentiment_score: Reward signal from reward model (0-1)
            kl_divergence: KL divergence from base model
            beta: KL penalty coefficient
            
        Returns:
            Combined RLHF reward
        """
        return sentiment_score - beta * kl_divergence
    
    def run_optimization_loop(self, prompts: List[List[str]], beta: float, num_generations: int = 5) -> Dict:
        """
        Run a single optimization loop with fixed β value.
        
        Args:
            prompts: List of prompts to optimize on
            beta: Fixed KL penalty coefficient
            num_generations: Number of generations per prompt
            
        Returns:
            Dictionary with results for this β value
        """
        results = {
            'beta': beta,
            'prompts_data': [],
            'avg_sentiment': [],
            'avg_kl': [],
            'avg_reward': [],
            'responses_sampled': []
        }
        
        all_responses = []
        
        for prompt_row in prompts[1:]:  # Skip header
            prompt_id = prompt_row[0]
            prompt_text = prompt_row[1]
            
            prompt_results = {
                'prompt_id': prompt_id,
                'prompt': prompt_text,
                'generations': [],
                'sentiments': [],
                'kl_divergences': [],
                'rewards': [],
            }
            
            for gen_idx in range(num_generations): # generate multiple responses per prompt to observe trends
                response = generate.generateSingleResponse(prompt_text)
                sentiment = score.getSentimentScore(response)
                kl_div = score.calculate_kl_divergence(response, all_responses)
                reward = self.compute_rlhf_reward(sentiment, kl_div, beta)
                
                prompt_results['generations'].append(response)
                prompt_results['sentiments'].append(sentiment)
                prompt_results['kl_divergences'].append(kl_div)
                prompt_results['rewards'].append(reward)
                all_responses.append(response)
                
                print(f"β={beta} | Prompt {prompt_id} | Gen {gen_idx+1}: "f"sentiment={sentiment:.3f}, kl={kl_div:.3f}, reward={reward:.3f}")
            
            top_response_idx = prompt_results['rewards'].index(max(prompt_results['rewards']))
            top_response = prompt_results['generations'][top_response_idx]
            
            prompt_results['top_response'] = top_response
            prompt_results['top_sentiment'] = prompt_results['sentiments'][top_response_idx]
            prompt_results['top_kl'] = prompt_results['kl_divergences'][top_response_idx]
            prompt_results['top_reward'] = prompt_results['rewards'][top_response_idx]
            
            results['prompts_data'].append(prompt_results)
            results['avg_sentiment'].append(sum(prompt_results['sentiments']) / len(prompt_results['sentiments']))
            results['avg_kl'].append(sum(prompt_results['kl_divergences']) / len(prompt_results['kl_divergences']))
            results['avg_reward'].append(sum(prompt_results['rewards']) / len(prompt_results['rewards']))
            results['responses_sampled'].extend(prompt_results['generations'])
        
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
            top_response = prompt_data['top_response']
            top_sentiment = prompt_data['top_sentiment']
            top_kl = prompt_data['top_kl']
            
            # Check for repetition in top response
            words = top_response.lower().split()
            if len(words) > 0:
                unique_ratio = len(set(words)) / len(words)
                if unique_ratio < 0.5:  # More than 50% repeated words
                    hacking_metrics['repetitive_responses'] += 1
            
            if top_sentiment > 0.8 and top_kl < 0.1: # High sentiment but low diversity could indicate hacking
                hacking_metrics['high_sentiment_low_diversity'] += 1
            
            hacking_metrics['reward_vs_sentiment'].append({
                'sentiment': top_sentiment,
                'kl': top_kl,
                'reward': prompt_data['top_reward']
            })
        
        return hacking_metrics
    
    def save_results(self, results: Dict, beta: float):
        """Save results for a single β value to CSV"""
        output_file = self.output_dir / f"beta_{beta:.2f}_results.csv"
        
        rows = []
        for prompt_data in results['prompts_data']:
            rows.append({
                'prompt_id': prompt_data['prompt_id'],
                'prompt': prompt_data['prompt'],
                'top_response': prompt_data['top_response'],
                'top_sentiment': prompt_data['top_sentiment'],
                'top_kl': prompt_data['top_kl'],
                'top_reward': prompt_data['top_reward'],
                'avg_sentiment': prompt_data['sentiments'][0] if prompt_data['sentiments'] else 0,  # Simplified
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
            
            avg_sent = sum(results['avg_sentiment']) / len(results['avg_sentiment']) if results['avg_sentiment'] else 0
            avg_kl = sum(results['avg_kl']) / len(results['avg_kl']) if results['avg_kl'] else 0
            avg_reward = sum(results['avg_reward']) / len(results['avg_reward']) if results['avg_reward'] else 0
            
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
    
    def run_sweep(self, num_prompts: int = 5, generations_per_prompt: int = 5):
        """
        Execute the full fixed-β sweep experiment.
        
        Args:
            num_prompts: Number of prompts to use in sweep
            generations_per_prompt: Number of generations per prompt per β value
        """
        print("=" * 60)
        print("FIXED-β SWEEP EXPERIMENT")
        print("=" * 60)
        
        prompts = utils.csvToArr(config.PROMPT_PATH)
        prompts = [prompts[0]] + prompts[1:num_prompts+1] # Limit to specified number of prompts
        
        print(f"Loaded {len(prompts)-1} prompts")
        print(f"Testing β values: {self.beta_values}")
        print(f"Generations per prompt: {generations_per_prompt}")
        print()
        
        for beta in self.beta_values:
            print(f"\n{'='*60}")
            print(f"Running optimization with β={beta}")
            print(f"{'='*60}")
            
            results = self.run_optimization_loop(prompts, beta, num_generations=generations_per_prompt)
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
    experiment.run_sweep(num_prompts=3, generations_per_prompt=4)
