"""
Smoke Test — validates every function in the pipeline with minimal compute.
Generates 2 responses for 2 prompts only. Does not save to your real data files.
Run this before any full experiment to catch broken imports or schema mismatches.
"""

import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import score, plot, utils
from src.generate import (
    generateSingleResponse,
    generateBestOfN,
    getAllBestOfN,
)

import matplotlib
matplotlib.use('Agg')   # non-interactive — must be set before pyplot import
import matplotlib.pyplot as plt

FAKE_PROMPTS = [
    ["prompt_id", "prompt"],
    ["1", "What is the capital of France?"],
    ["2", "Explain how photosynthesis works."],
]

FAKE_RESPONSES_ARR = [
    ["prompt_id", "candidate_id", "response"],
    ["1", "1", "Paris is the capital of France and a wonderful city."],
    ["1", "2", "France's capital is Paris, known for the Eiffel Tower."],
    ["2", "1", "Photosynthesis is the process by which plants make food."],
    ["2", "2", "Plants use sunlight to convert CO2 into glucose."],
]

PASS = "  ✓"
FAIL = "  ✗"


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def test_score_functions():
    section("1. Score functions (score.py)")
    test_response = "This is a great day and everything is wonderful."

    try:
        sentiment = score.getSentimentScore(test_response)
        assert 0 <= sentiment <= 1
        print(f"{PASS} getSentimentScore → {sentiment:.4f}")
    except Exception as e:
        print(f"{FAIL} getSentimentScore → {e}")

    try:
        perplexity = score.getPerplexity(test_response)
        assert perplexity > 0
        print(f"{PASS} getPerplexity → {perplexity:.2f}")
    except Exception as e:
        print(f"{FAIL} getPerplexity → {e}")

    try:
        kl = score.calculate_batch_kl(test_response)
        assert kl > 0
        print(f"{PASS} getKLDivergence → {kl:.4f}")
    except Exception as e:
        print(f"{FAIL} getKLDivergence → {e}")

    try:
        reward = score.getPenalizedReward(test_response, beta=0.1)
        assert all(k in reward for k in ['sentiment_score', 'kl_divergence', 'penalized_reward'])
        print(f"{PASS} getPenalizedReward (β=0.1) → {reward}")
    except Exception as e:
        print(f"{FAIL} getPenalizedReward → {e}")


def test_generation_functions():
    section("2. Generation functions (generate.py)")

    try:
        response = generateSingleResponse(FAKE_PROMPTS[1][1])
        assert isinstance(response, str) and len(response) > 0
        print(f"{PASS} generateSingleResponse → '{response[:60]}...'")
    except Exception as e:
        print(f"{FAIL} generateSingleResponse → {e}")

    try:
        best = generateBestOfN(FAKE_PROMPTS[1][1], N=2, verbose=False)
        assert all(k in best for k in ['response', 'sentiment_score'])
        print(f"{PASS} generateBestOfN (N=2) → sentiment={best['sentiment_score']:.4f}")
    except Exception as e:
        print(f"{FAIL} generateBestOfN → {e}")

    try:
        bon_path = "/tmp/smoke_test_bonN.csv"
        getAllBestOfN(FAKE_PROMPTS, N=2, outputCSVPath=bon_path, verbose=False)
        bon_df = pd.read_csv(bon_path)
        assert len(bon_df) == 2
        assert 'perplexity' not in bon_df.columns  # added manually in main.py, not here
        print(f"{PASS} getAllBestOfN → {len(bon_df)} rows, columns: {list(bon_df.columns)}")
    except Exception as e:
        print(f"{FAIL} getAllBestOfN → {e}")


def test_batch_scoring():
    section("3. Batch scoring (score.py — array functions)")

    sentiments = None
    perplexities = None

    try:
        sentiments = score.getAllSentimentScores(FAKE_RESPONSES_ARR)
        assert len(sentiments) == 4
        print(f"{PASS} getAllSentimentScores → {len(sentiments)} rows")
    except Exception as e:
        print(f"{FAIL} getAllSentimentScores → {e}")

    try:
        perplexities = score.getAllPerplexities(FAKE_RESPONSES_ARR)
        assert len(perplexities) == 4
        print(f"{PASS} getAllPerplexities → {len(perplexities)} rows")
    except Exception as e:
        print(f"{FAIL} getAllPerplexities → {e}")

    if sentiments and perplexities:
        try:
            scored_path = "/tmp/smoke_test_scored.csv"
            score.fillScoredGenerations(scored_path, sentiments, perplexities)
            scored_df = pd.read_csv(scored_path)
            assert list(scored_df.columns) == [
                "prompt_id", "candidate_id", "response", "sentiment_score", "perplexity"
            ]
            print(f"{PASS} fillScoredGenerations → {len(scored_df)} rows, columns OK")
        except Exception as e:
            print(f"{FAIL} fillScoredGenerations → {e}")


def test_plot_functions():
    section("4. Plot functions (plot.py — schema check only, no display)")

    # Patch show so no windows open
    plt.show = lambda: None

    scored_df = pd.DataFrame({
        'sentiment_score': [0.8, 0.6, 0.9, 0.5],
        'perplexity':      [50,  80,  45,  120],
    })
    bon_df = pd.DataFrame({
        'N':               [1,    2,    4,    8   ],
        'sentiment_score': [0.6,  0.7,  0.8,  0.85],
        'perplexity':      [60,   70,   90,   130 ],
    })
    sweep_summary_df = pd.DataFrame({
        'beta':                 [0.01, 0.1,  0.5,  1.0 ],
        'avg_sentiment_score':  [0.85, 0.80, 0.70, 0.60],
        'avg_kl_divergence':    [0.80, 0.50, 0.25, 0.10],
        'avg_reward':           [0.77, 0.75, 0.68, 0.59],
        'repetitive_responses': [3,    1,    0,    0   ],
    })
    all_results_df = pd.DataFrame({
        'top_sentiment': [0.9,  0.8,  0.7,  0.6 ],
        'top_kl':        [0.8,  0.5,  0.3,  0.1 ],
        'beta':          [0.01, 0.1,  0.5,  1.0 ],
    })
    adaptive_history_df = pd.DataFrame({
        'step':            [1,    2,    3,    4,    5   ],
        'beta':            [0.1,  0.12, 0.10, 0.11, 0.10],
        'kl_divergence':   [0.8,  0.6,  0.5,  0.52, 0.50],
        'sentiment_score': [0.7,  0.75, 0.78, 0.77, 0.79],
        'reward':          [0.62, 0.68, 0.73, 0.71, 0.74],
        'target_kl':       [0.5,  0.5,  0.5,  0.5,  0.5 ],
    })
    comparison_df = pd.DataFrame({
        'approach':      ['Fixed β=0.01', 'Fixed β=0.1', 'Adaptive (final β=0.10)'],
        'beta':          [0.01, 0.1,  0.10],
        'avg_sentiment': [0.85, 0.80, 0.79],
        'avg_kl':        [0.80, 0.50, 0.50],
        'avg_reward':    [0.77, 0.75, 0.74],
        'type':          ['fixed', 'fixed', 'adaptive'],
    })

    plot_tests = [
        ("plotSentimentVsPerplexity",         lambda: plot.plotSentimentVsPerplexity(scored_df)),
        ("plotSentimentVsPerplexity (with N)", lambda: plot.plotSentimentVsPerplexity(bon_df)),
        ("plotMetricByN (sentiment)",          lambda: plot.plotMetricByN(bon_df, 'sentiment_score')),
        ("plotMetricByN (perplexity)",         lambda: plot.plotMetricByN(bon_df, 'perplexity')),
        ("plotElbow",                          lambda: plot.plotElbow(bon_df)),
        ("plotFixedBetaComparison",            lambda: plot.plotFixedBetaComparison(sweep_summary_df)),
        ("plotRewardSurface",                  lambda: plot.plotRewardSurface(all_results_df)),
        ("plotAdaptiveBetaTrace",              lambda: plot.plotAdaptiveBetaTrace(adaptive_history_df)),
        ("plotAdaptiveVsFixed",                lambda: plot.plotAdaptiveVsFixed(adaptive_history_df, sweep_summary_df, best_fixed_beta=0.1)),
        ("plotRepetitionRate",                 lambda: plot.plotRepetitionRate(sweep_summary_df)),
        ("plotComparisonSummary",              lambda: plot.plotComparisonSummary(comparison_df)),
    ]

    for name, fn in plot_tests:
        try:
            fn()
            print(f"{PASS} {name}")
        except Exception as e:
            print(f"{FAIL} {name} → {e}")


def main():
    print("\n" + "=" * 60)
    print("  SMOKE TEST — reward overoptimization pipeline")
    print("=" * 60)

    test_score_functions()
    test_generation_functions()
    test_batch_scoring()
    test_plot_functions()

    section("Smoke test complete")
    print("  If all checks show ✓, your pipeline is ready for a full run.")
    print("  Any ✗ lines show exactly which function to fix before running.")
    print()


if __name__ == "__main__":
    main()