# run the sentiment model
# compute perplexity
# combine scores into a dictionary or DataFrame

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from scipy.special import softmax
from collections import Counter
from typing import List, Optional
import torch
import math
import csv

PERPLEXITY_MODEL = "gpt2"   # for the perplexity score
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment" # for the sentiment score

# Sentiment tokenizer's purpose is to convert the input text into a format that the sentiment model can understand, while the sentiment model's purpose is to analyze the sentiment of the input text and provide a sentiment score.
sentiment_tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL) 
sentiment_model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL)

perplexity_tokenizer = AutoTokenizer.from_pretrained(PERPLEXITY_MODEL)
perplexity_model = AutoModelForCausalLM.from_pretrained(PERPLEXITY_MODEL)

# calculating the sentiment score

def getSentimentScore (response) : 
    '''
    Calculates the sentiment score for a given response using the sentiment model.
    input: a single response
    output: the sentiment score for the response
    '''
    encoded_input = sentiment_tokenizer(response, return_tensors="pt", truncation=True, max_length=512)
    output = sentiment_model(**encoded_input)

    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    positive_score = float(scores[2])   # LABEL_2 = positive
    return positive_score


def getAllSentimentScores (responsesArr) : 
    '''
    Calculates the sentiment scores for all responses in the responses array.
    input: an array of responses
    output: an array of sentiment scores for each response
    '''
    sentimentScores = []

    for row in responsesArr[1:] : 
        curPromptId = row[0]
        curCandidateId = row[1]
        curResponse = row[2]
        curSentimentScore = getSentimentScore(curResponse)
        curRow = [curPromptId, curCandidateId, curResponse, curSentimentScore]
        sentimentScores.append(curRow)

        # debugging
        # print(curPromptId, curCandidateId)

    return sentimentScores  # find the sentiment score for each response


# calculating the perplexity score

def getPerplexity(response):
    '''
    Calculates the perplexity score for a given response using the perplexity model.
    input: a single response
    output: the perplexity score for the response
    '''
    text = str(response).strip()
    if not text:
        return float("nan")

    encodings = perplexity_tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    input_ids = encodings["input_ids"]

    with torch.no_grad():
        outputs = perplexity_model(input_ids, labels=input_ids)
        loss = outputs.loss

    perplexity = math.exp(loss.item())
    return perplexity


def getAllPerplexities(responsesArr) :
    '''
    Calculates the perplexity scores for all responses in the responses array.
    input: an array of responses
    output: an array of perplexity scores for each response
    '''
    perplexities = []

    for row in responsesArr[1:] : 
        curPromptId = row[0]
        curCandidateId = row[1]
        curResponse = row[2]
        curPerplexity = getPerplexity(curResponse)
        curRow = [curPromptId, curCandidateId, curResponse, curPerplexity]
        perplexities.append(curRow)
        # print(curRow)

    return perplexities  # find the perplexity for each response


def _token_counts(text: str) -> Counter:
    """Tokenize text and return token id frequencies."""
    if not text:
        return Counter()

    encodings = perplexity_tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    token_ids = encodings["input_ids"][0].tolist()
    return Counter(token_ids)


def _aggregate_counts(responses: List[str]) -> Counter:
    """Aggregate token counts across multiple responses."""
    aggregate = Counter()
    for text in responses:
        aggregate.update(_token_counts(text))
    return aggregate


def calculate_kl_divergence(response: str, reference_responses: Optional[List[str]] = None, epsilon: float = 1e-12) -> float:
    """Compute KL(P || Q) between a response distribution and a reference distribution."""
    if not response:
        return 0.0

    p_counts = _token_counts(response)
    if not p_counts:
        return 0.0

    if not reference_responses:
        return 0.0

    q_counts = _aggregate_counts(reference_responses)
    if not q_counts:
        return 0.0

    vocab = set(p_counts) | set(q_counts)
    p_total = sum(p_counts.values())
    q_total = sum(q_counts.values())
    if p_total == 0 or q_total == 0:
        return 0.0

    vocab_size = len(vocab)
    kl = 0.0
    for token in vocab:
        p_prob = (p_counts.get(token, 0) + epsilon) / (p_total + epsilon * vocab_size)
        q_prob = (q_counts.get(token, 0) + epsilon) / (q_total + epsilon * vocab_size)
        kl += p_prob * math.log(p_prob / q_prob)
    return float(kl)


def calculate_batch_kl(responses: List[str], reference_responses: Optional[List[str]] = None) -> float:
    """Compute average KL divergence for a batch of responses."""
    if not responses:
        return 0.0

    if reference_responses is None:
        if len(responses) == 1:
            return 0.0
        kl_values = []
        for idx, response in enumerate(responses):
            other_responses = [r for j, r in enumerate(responses) if j != idx]
            kl_values.append(calculate_kl_divergence(response, other_responses))
        return sum(kl_values) / len(kl_values)

    kl_values = [calculate_kl_divergence(response, reference_responses) for response in responses]
    return sum(kl_values) / len(kl_values)


def fillScoredGenerations(fileName, sentimentArr, perplexities) :
    '''
    Combines the sentiment scores and perplexity scores into a single CSV file.
    input: the path to save the scored generations CSV file, an array of sentiment scores, and an array of perplexity scores
    output: a CSV file with the prompt id, candidate id, response, sentiment score and perplexity score for each response
    '''
    with open(fileName, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["prompt_id", "candidate_id", "response", "sentiment_score", "perplexity"])
        
        for rowSentiment in sentimentArr : 
            curPromptId = rowSentiment[0]
            curCandidateId = rowSentiment[1]
            curResponse = rowSentiment[2]
            curSentimentScore = rowSentiment[3]

            for rowPerplexity in perplexities : 
                if (rowPerplexity[0] == curPromptId) &  (rowPerplexity[1] == curCandidateId) :
                    curPerplexity = rowPerplexity[3]

            curRow = [curPromptId, curCandidateId, curResponse, curSentimentScore, curPerplexity]
            writer.writerow(curRow)

    return

