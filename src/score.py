# run the sentiment model
# compute perplexity
# combine scores into a dictionary or DataFrame

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from scipy.special import softmax
import torch
import math
import csv

PERPLEXITY_MODEL = "gpt2"   # for the perplexity score
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment" # for the sentiment score

sentiment_tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL)

perplexity_tokenizer = AutoTokenizer.from_pretrained(PERPLEXITY_MODEL)
perplexity_model = AutoModelForCausalLM.from_pretrained(PERPLEXITY_MODEL)

# calculating the sentiment score

def getSentimentScore (response) : 
    encoded_input = sentiment_tokenizer(response, return_tensors="pt", truncation=True, max_length=512)
    output = sentiment_model(**encoded_input)

    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    positive_score = float(scores[2])   # LABEL_2 = positive
    return positive_score


def getAllSentimentScores (responsesArr) : 
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


def fillScoredGenerations(fileName, sentimentArr, perplexities) :
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

