# plot sentiment vs perplexity
# plot average reward by N
# plot average perplexity by N
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {
    'beta_0.01': '#378ADD',
    'beta_0.1':  '#1D9E75',
    'beta_0.5':  '#EF9F27',
    'beta_1.0':  '#D85A30',
    'adaptive':  '#7F77DD',
    'sentiment': '#185FA5',
    'perplexity': '#993C1D',
}

def plotSentimentVsPerplexity(df: pd.DataFrame):
    '''
    Scatter of sentiment score vs perplexity. Shows "delusion".
    Color-coded by N.
    The higher the perplexity, the less coherent.
    Perplexity is based on GPT-2
    Sentiment is based on RoBERTa

    Args: DataFrame with columns: sentiment_score, perplexity, (optionally N)
    '''
    fig, ax = plt.subplots(figsize=(10, 6))

    if 'N' in df.columns:
        n_values = sorted(df['N'].unique())
        cmap = plt.cm.get_cmap('Blues', len(n_values) + 2)
        for i, n in enumerate(n_values):
            subset = df[df['N'] == n]
            ax.scatter(subset['perplexity'], subset['sentiment_score'],
                       alpha=0.6, label=f'N={n}', color=cmap(i + 2), s=40)
        ax.legend(title='N (candidates)')
    else:
        ax.scatter(df['perplexity'], df['sentiment_score'],
                   alpha=0.5, color=COLORS['sentiment'], s=40)

    ax.set_title('Sentiment Score vs Perplexity')
    ax.set_xlabel('Perplexity')
    ax.set_ylabel('Sentiment Score')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plotMetricByN(df: pd.DataFrame, metric: str):
    '''
    Line plot showing average of <metric> grouped by N.
    <metric> can be 'sentiment_score' or 'perplexity'.

    Args: DataFrame with columns: N, <metric>
    '''
    grouped = df.groupby('N')[metric].mean().reset_index()
    color = COLORS['sentiment'] if 'sentiment' in metric else COLORS['perplexity']

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(grouped['N'], grouped[metric], marker='o', color=color, linewidth=2)
    ax.set_title(f'Average {metric.replace("_", " ").title()} by N')
    ax.set_xlabel('N (Best-of-N candidates)')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_xticks(grouped['N'])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plotElbow(df: pd.DataFrame):
    '''
    Overlays sentiment and perplexity vs N.
    The point where perplexity rises while sentiment plateaus = elbow.

    Args: DataFrame with columns: N, sentiment_score, perplexity
    '''
    grouped = df.groupby('N').agg(
        avg_sentiment=('sentiment_score', 'mean'),
        avg_perplexity=('perplexity', 'mean')
    ).reset_index()

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    ax1.plot(grouped['N'], grouped['avg_sentiment'],
             marker='o', color=COLORS['sentiment'], linewidth=2, label='Sentiment')
    ax2.plot(grouped['N'], grouped['avg_perplexity'],
             marker='s', color=COLORS['perplexity'], linewidth=2,
             linestyle='--', label='Perplexity')

    ax1.set_xlabel('N (Best-of-N candidates)')
    ax1.set_ylabel('Avg Sentiment Score', color=COLORS['sentiment'])
    ax2.set_ylabel('Avg Perplexity (GPT-2)', color=COLORS['perplexity'])
    ax1.set_xticks(grouped['N'])

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    ax1.set_title('Elbow Detection: Sentiment vs Perplexity by N')
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plotFixedBetaComparison(summary_df: pd.DataFrame):
    '''
    Grouped bar chart: sentiment, KL, and penalized reward per beta value.

    Args: sweep_summary.csv DataFrame with columns:
           beta, avg_sentiment_score, avg_kl_divergence, avg_reward
    '''
    betas = summary_df['beta'].astype(str)
    x = np.arange(len(betas))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, summary_df['avg_sentiment_score'], width,
           label='Avg Sentiment', color=COLORS['sentiment'], alpha=0.85)
    ax.bar(x,          summary_df['avg_kl_divergence'],  width,
           label='Avg KL Divergence', color=COLORS['perplexity'], alpha=0.85)
    ax.bar(x + width,  summary_df['avg_reward'],         width,
           label='Avg Penalized Reward', color=COLORS['adaptive'], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([f'β={b}' for b in betas])
    ax.set_title('Fixed β Sweep: Sentiment, KL, and Penalized Reward')
    ax.set_ylabel('Score')
    ax.legend()
    ax.grid(True, alpha=0.2, axis='y')
    plt.tight_layout()
    plt.show()

def plotRewardSurface(all_results_df: pd.DataFrame):
    '''
    Scatter of sentiment vs KL divergence, one point per generation,
    colored by beta. Shows how each penalty level shifts the distribution.
    KL-divergence is from based model.

    Args: merged DataFrame from all beta CSVs with columns:
           top_sentiment, top_kl, beta
    '''
    beta_vals = sorted(all_results_df['beta'].unique())
    color_list = [COLORS['beta_0.01'], COLORS['beta_0.1'],
                  COLORS['beta_0.5'], COLORS['beta_1.0']]

    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, b in enumerate(beta_vals):
        subset = all_results_df[all_results_df['beta'] == b]
        ax.scatter(subset['top_kl'], subset['top_sentiment'],
                   label=f'β={b}', color=color_list[idx % len(color_list)],
                   alpha=0.7, s=50)

    ax.set_title('Reward Surface: Sentiment vs KL Divergence by β')
    ax.set_xlabel('KL Divergence')
    ax.set_ylabel('Sentiment Score')
    ax.legend(title='β value')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plotComparisonSummary(comparison_df: pd.DataFrame):
    '''
    Plot 1: Sentiment vs KL scatter (fixed = circles, adaptive = star)
    Plot 2: Average reward bar chart per approach
    Plot 3: Reward vs beta coefficient (log x-axis)

    Args: comparison_df with columns:
           approach, beta, avg_sentiment, avg_kl, avg_reward, type
    '''
    fixed_data    = comparison_df[comparison_df['type'] == 'fixed']
    adaptive_data = comparison_df[comparison_df['type'] == 'adaptive']

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Plot 1
    axes[0].scatter(fixed_data['avg_kl'], fixed_data['avg_sentiment'],
                    s=100, alpha=0.7, label='Fixed β', color=COLORS['sentiment'])
    if not adaptive_data.empty:
        axes[0].scatter(adaptive_data['avg_kl'], adaptive_data['avg_sentiment'],
                        s=200, marker='*', label='Adaptive β', color=COLORS['adaptive'])
    axes[0].set_xlabel('Average KL Divergence')
    axes[0].set_ylabel('Average Sentiment Score')
    axes[0].set_title('Sentiment vs KL Divergence')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2
    colors = [COLORS['adaptive'] if t == 'adaptive' else COLORS['sentiment']
              for t in comparison_df['type']]
    axes[1].bar(range(len(comparison_df)), comparison_df['avg_reward'],
                color=colors, alpha=0.85)
    axes[1].set_xticks(range(len(comparison_df)))
    axes[1].set_xticklabels(comparison_df['approach'], rotation=45, ha='right')
    axes[1].set_ylabel('Average Reward')
    axes[1].set_title('RLHF Reward Comparison')
    axes[1].grid(True, alpha=0.3, axis='y')

    # Plot 3
    axes[2].scatter(fixed_data['beta'], fixed_data['avg_reward'],
                    s=100, alpha=0.7, label='Fixed β', color=COLORS['sentiment'])
    if not adaptive_data.empty:
        axes[2].scatter(adaptive_data['beta'], adaptive_data['avg_reward'],
                        s=200, marker='*', label='Adaptive β', color=COLORS['adaptive'])
    axes[2].set_xlabel('β Value')
    axes[2].set_ylabel('Average Reward')
    axes[2].set_title('Reward vs β Coefficient')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xscale('log')

    plt.tight_layout()
    plt.show()

def plotAdaptiveBetaTrace(history_df: pd.DataFrame):
    '''
    Tracks beta value and KL divergence over optimization steps.
    Shows whether the controller stabilizes KL toward the target.

    Args: adaptive_optimization_history.csv DataFrame with columns:
           step, beta, kl_divergence, (optionally target_kl)
    '''
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    ax1.plot(history_df['step'], history_df['beta'],
             marker='o', color=COLORS['adaptive'], linewidth=2, label='β (adaptive)')
    ax2.plot(history_df['step'], history_df['kl_divergence'],
             marker='s', color=COLORS['perplexity'], linewidth=2,
             linestyle='--', label='KL Divergence')

    if 'target_kl' in history_df.columns:
        ax2.axhline(y=history_df['target_kl'].iloc[0], color='gray',
                    linestyle=':', linewidth=1.5, label='KL target')

    ax1.set_xlabel('Optimization Step')
    ax1.set_ylabel('β value', color=COLORS['adaptive'])
    ax2.set_ylabel('KL Divergence', color=COLORS['perplexity'])

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    ax1.set_title('Adaptive β Controller: Beta and KL Divergence over Steps')
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plotAdaptiveVsFixed(adaptive_df: pd.DataFrame, fixed_df: pd.DataFrame,
                        best_fixed_beta: float = 0.1):
    '''
    Key paper comparison: adaptive β vs best fixed β.
    Comparing sentiment and KL trajectories over equivalent steps.

    Args: adaptive history DataFrame, sweep_summary DataFrame,
           best_fixed_beta: which fixed beta row to compare against
    '''
    fixed_subset = fixed_df[fixed_df['beta'] == best_fixed_beta]
    beta_color   = COLORS.get(f'beta_{best_fixed_beta}', COLORS['sentiment'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(adaptive_df['step'], adaptive_df['sentiment_score'],
             color=COLORS['adaptive'], linewidth=2, marker='o', label='Adaptive β')
    ax1.plot(range(len(fixed_subset)), fixed_subset['avg_sentiment_score'].values,
             color=beta_color, linewidth=2, marker='s', linestyle='--',
             label=f'Fixed β={best_fixed_beta}')
    ax1.set_title('Sentiment Score: Adaptive vs Fixed β')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Avg Sentiment Score')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(adaptive_df['step'], adaptive_df['kl_divergence'],
             color=COLORS['adaptive'], linewidth=2, marker='o', label='Adaptive β')
    ax2.plot(range(len(fixed_subset)), fixed_subset['avg_kl_divergence'].values,
             color=beta_color, linewidth=2, marker='s', linestyle='--',
             label=f'Fixed β={best_fixed_beta}')
    ax2.set_title('KL Divergence: Adaptive vs Fixed β')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('KL Divergence')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Adaptive β vs Best Fixed β', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()

def plotRepetitionRate(summary_df: pd.DataFrame):
    '''
    Bar chart of repetitive response count per beta.
    An illustration of reward hacking.

    Args: sweep_summary.csv DataFrame with columns:
           beta, repetitive_responses
    '''
    fig, ax = plt.subplots(figsize=(8, 5))
    betas = summary_df['beta'].astype(str)
    ax.bar(betas, summary_df['repetitive_responses'],
           color=COLORS['perplexity'], alpha=0.85)

    ax.set_title('Repetitive Responses by β Value')
    ax.set_xlabel('β value')
    ax.set_ylabel('Number of repetitive responses')
    ax.grid(True, alpha=0.2, axis='y')
    plt.tight_layout()
    plt.show()