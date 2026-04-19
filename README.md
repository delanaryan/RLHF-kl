# Project description 
This project investigates the phenomenon of Reward Overoptimization (Reward Hacking) in Large Language Models by analyzing the linguistic degradation that occurs when a model is forced to maximize a specific sentiment. We will use Llama-3 (8B) as our target policy, generating varied responses to a fixed set of prompts through Best-of-N sampling. Our evaluation pipeline utilizes a multi-model tech stack: Ollama serves the primary LLM, DistilRoBERTa acts as a "Proxy Reward Model" to drive the positivity score, and a frozen GPT-2 (Small) serves as an objective auditor to measure Perplexity and detect "delusional" output. By mapping these scores, we aim to identify the specific "Elbow Point" where optimization pressure breaks the model's linguistic coherence, providing a practical case study in AI alignment failure on consumer-grade hardware. 

In other words, this project explores why AI models sometimes become "delusional" (repetitive or nonsensical) when they are pushed to be too positive. We want to find the exact point where an AI stops being helpful and starts "hacking" its reward system just to get a high score. 


# How to run

## Installing Ollama
We're using Ollama's Mistral model to generate the responses we'll be analyzing. Run `curl -fsSL https://ollama.com/install.sh | sh` to download Ollama.

To run Mistral locally, we need to follow the follow the following commands :
```
ollama pull mistral
ollama run mistral
```

To make sure that Ollama can run using our python code, we need to do : `pip install ollama pandas`

## Installing RoBERTa 
We need RoBERTa to do the sentiment analysis of Ollama's responses! To use it locally, we need to run the following commands :
```
pip install transformers
pip install torch
```

# File structure 
```
RLHF-KL/
│
├── .gitignore
├── requirements.txt
├── README.md
├── config.py
├── main.py
│
├── data/
│   ├── prompts.csv
│   ├── rawGenerations.csv     (stores outputs before scoring)
│   └── scoredGenerations.csv  (stores outputs after scoring + their scores)
│   └── selectedGenerations.csv  (stores outputs with the highest sentiment score (per N-value))
│
├── src/
│   ├── __init__.py
│   ├── generate.py
│   ├── score.py
│   ├── utils.py
│   └── plot.py
│
├── experiments/
│   └── run_experiment.py
│
└── results/
```
