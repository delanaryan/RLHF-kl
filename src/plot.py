# plot sentiment vs perplexity
# plot average reward by N
# plot average perplexity by N
import matplotlib.pyplot as plt
import numpy as np

def plotSentimentVsPerplexity(sentimentScores, perplexityScores) :
    sentimentScores = np.array(sentimentScores)
    perplexityScores = np.array(perplexityScores)

    plt.figure(figsize=(10, 6))
    plt.scatter(perplexityScores[:, 3], sentimentScores[:, 3], alpha=0.5)
    plt.title('Sentiment Score vs Perplexity')
    plt.xlabel('Perplexity')
    plt.ylabel('Sentiment Score')
    plt.grid()
    plt.show()