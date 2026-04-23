from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from scipy.special import softmax
from collections import Counter
from typing import List
import torch
import math
import csv

SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment" # for the sentiment score
sentiment_tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL) 
sentiment_model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL)

PERPLEXITY_MODEL = "gpt2"   # for the perplexity score
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
    '''
    tokenizes a text using the perplexity tokenizer and returns token frequencies.
    input: a single response
    output: Counter of token ids
    '''
    if not text:
        return Counter()

    encodings = perplexity_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )
    token_ids = encodings["input_ids"][0].tolist()
    return Counter(token_ids)


def _aggregate_counts(responses: List[str]) -> Counter:
    '''
    Aggregates token counts across multiple responses.
    input: list of reference responses
    output: Counter of token ids across all reference responses
    '''
    aggregate = Counter()

    for text in responses:
        aggregate.update(_token_counts(text))

    return aggregate


def calculate_kl_divergence(response, reference_responses, epsilon=1e-12):
    '''
    calculates the KL divergence between one response and a set of reference responses.
    input: 
        response : a single response,
        reference_responses : array of reference responses
    output: 
        float(kl) : the KL divergence score for the response
    '''
    if not response:
        return 0.0
    if not reference_responses:
        return 0.0

    p_counts = _token_counts(response)
    q_counts = _aggregate_counts(reference_responses)

    if not p_counts or not q_counts:
        return 0.0

    vocab = set(p_counts.keys()) | set(q_counts.keys())
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


def getAllKLdivergences(promptsArr, responsesArr):
    '''
    calculates the KL divergence scores for all responses in the responses array.
    input: 
        responsesArr : an array of responses
    output: 
        klScores : an array of KL divergence scores for each response
    '''
    klScores = []

    # skip header
    from src import generate
    reference_responses = generate.build_reference_responses(promptsArr, generate.generateSingleResponse)
    dataRows = responsesArr[1:]

    for row in dataRows:
        curPromptId = row[0]
        curCandidateId = row[1]
        curResponse = row[2]

        curKL = calculate_kl_divergence(curResponse, reference_responses)
        curRow = [curPromptId, curCandidateId, curResponse, curKL]
        klScores.append(curRow)

    return klScores

def writeScoresToCSV(fileName, sentimentArr, perplexities, klScores) :
    '''
    Combines the sentiment scores and perplexity scores into a single CSV file.
    input: 
        the path to save the scored generations CSV file, an array of sentiment scores, and an array of perplexity scores
    output: 
        a CSV file with the prompt id, candidate id, response, sentiment score and perplexity score for each response
    '''
    with open(fileName, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["prompt_id", "candidate_id", "response", "sentiment_score", "perplexity", "kl_divergence"])
        
        for rowSentiment in sentimentArr : 
            curPromptId = rowSentiment[0]
            curCandidateId = rowSentiment[1]
            curResponse = rowSentiment[2]
            curSentimentScore = rowSentiment[3]

            for rowPerplexity in perplexities : 
                if (rowPerplexity[0] == curPromptId) &  (rowPerplexity[1] == curCandidateId) :
                    curPerplexity = rowPerplexity[3]

            for rowKLscores in klScores : 
                # curRow = [curPromptId, curCandidateId, curResponse, curKL]
                if (rowKLscores[0] == curPromptId) &  (rowKLscores[1] == curCandidateId) :
                    curKLscore = rowKLscores[3]

            curRow = [curPromptId, curCandidateId, curResponse, curSentimentScore, curPerplexity, curKLscore]
            writer.writerow(curRow)
    return

def getScoredGenerations (fileName, promptsArr, responsesArr) :
    sentimentArr = getAllSentimentScores(responsesArr)
    perplexities = getAllPerplexities(responsesArr)
    klScores = getAllKLdivergences(promptsArr, responsesArr) 
    writeScoresToCSV(fileName, sentimentArr, perplexities, klScores)


def getPenalizedReward(response, beta):
    '''
    Computes KL-penalized reward.
    Reward = sentiment_score - beta * kl_divergence
    input:  response string, beta float
    output: dict with sentiment_score, kl_divergence, penalized_reward
    '''
    sentiment = getSentimentScore(response)
    kl        = getKLDivergence(response)
    return {
        'sentiment_score':   sentiment,
        'kl_divergence':     kl,
        'penalized_reward':  sentiment - beta * kl,
    }
