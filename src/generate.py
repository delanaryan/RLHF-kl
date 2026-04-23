# send prompts to Llama
# generate multiple candidates per prompt
# return/save outputs

import pandas as pd
import ollama
import csv
from src.score import getSentimentScore
import statistics

def generateSingleResponse(prompt):
    '''
    Generates a response using the ollama API for a given prompt.
    input: a single prompt
    output: a single response from the model
    '''
    try:
        response = ollama.chat(
            model='mistral', 
            messages=[{'role': 'user', 'content': prompt}],
        )
        return response['message']['content']
    
    except Exception as e:
        return f"Error: {e}"


def getAllResponses(promptArray, maxN, rawGenerationCSVPath) :
    '''
    Generates responses for all prompts in the prompt array and saves them to a CSV file.
    input: an array of prompts, the number of candidates to generate per prompt, and the path to save the raw generations CSV file
    output: a CSV file with all generated responses for each prompt
    '''
    with open(rawGenerationCSVPath, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["prompt_id", "candidate_id", "response"])
        
        for row in promptArray[1:]:
            prompt_id = row[0]
            prompt = row[1]

            for i in range(maxN):
                curResponse = generateSingleResponse(prompt)
                writer.writerow([prompt_id, i + 1, curResponse])
                
                # debugging 
                print([prompt_id, i + 1, curResponse])
    return

def getResponsesChunk (promptArray, start_prompt_id, end_prompt_id, maxN) : 
    '''
    Generates responses for a chunk of prompts and saves them to a CSV file.
    input: an array of prompts, the start and end indices of the chunk, the number of candidates to generate per prompt, and the path to save the raw generations CSV file
    output: a CSV file with all generated responses for each prompt in the chunk
    '''
    nameOfNewFile = "data/rawGenerations" + str(start_prompt_id) + "to" + str(end_prompt_id) + ".csv"
    
    with open(nameOfNewFile, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["prompt_id", "candidate_id", "response"])
        
        for row in promptArray[start_prompt_id : end_prompt_id + 1]:
            prompt_id = row[0]
            prompt = row[1]

            # debugging
            # print(prompt_id)
            # print(prompt)

            for i in range(maxN):
                curResponse = generateSingleResponse(prompt)
                writer.writerow([prompt_id, i + 1, curResponse])
                
                # debugging 
                print(prompt_id, i + 1, curResponse)
    return


# def getBestOfN (responsesCurPrompt, curN, beta=0.0) :
#     '''
#     Selects the best candidate for the current prompt_id and N value (the one with highest sentiment)
    
#     Input :
#         responsesCurPrompt : array containing the prompt generations and their score for a specific prompt_id
#         curN : current value of N (in the Best-of-N sampling)
#         beta: beta value

#     Output : 
#         bestCandidate : single array, the generation with the highest sentiment score
#     '''


#     candidates = responsesCurPrompt[:curN]
#     bestCandidate = max(candidates, key=lambda x: float(x[3]))

#     print("The best candidate for N = ", curN, " is : ")
#     print("\t- ", bestCandidate[2])
#     print("\t- Sentiment score : ", bestCandidate[3])
#     print("\t- Perplexity : ", bestCandidate[4])

#     return bestCandidate


def compute_rlhf_reward(sentiment_score: float, kl_divergence: float, beta: float) -> float:
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


# separate function using beta value
def getBestOfN (prompt_id, curBeta, curN, scoredArr) :
    '''
    For the specified prompt_id, beta, and N value : 
        we find all potential 'candidates' using the scoredArr
        calculate the reward for each candidate
        find the 'best' one, meaning the one 
    
    :param prompt_ID: Description
    :param curBeta: Description
    :param curN: Description
    
    '''

    responsesCurPrompt = []
    for response in scoredArr[1:] : # skipping the header
        if (response[0] == str(prompt_id)) : 
            responsesCurPrompt.append(response)

    candidatesWithReward = []

    candidates = responsesCurPrompt[:curN]
    allSentiments = []
    allPerplexities = []
    allKLdivergences = []
    allRewards = []

    for candidate in candidates : 
        candidateID = candidate[1]
        curResponse = candidate[2]
        sentiment = candidate[3]
        perplexity = candidate[4]
        kl_divergence = candidate[4]  

        reward = compute_rlhf_reward(sentiment, kl_divergence, curBeta)

        # append all scores
        allSentiments.append(sentiment)
        allPerplexities.append(perplexity)
        allKLdivergences.append(kl_divergence)
        allRewards.append(reward)

        newCandidate = [prompt_id, curN, candidateID, curResponse, sentiment, perplexity, kl_divergence, reward]
        candidatesWithReward.append(newCandidate)

    # calculate mean scores of all 16 prompts
    meanSentiment = statistics.mean(allSentiments)
    meanPerplexity = statistics.mean(allPerplexities)
    meanKL = statistics.mean(allKLdivergences)
    meanReward = statistics.mean(allRewards)
    
    # using the reward as determining factor
    bestCandidate = max(candidatesWithReward, key=lambda x: float(x[7]))   

    return bestCandidate, meanSentiment, meanPerplexity, meanKL, meanReward


def getAllBestOfN(nValuesArr, beta, scoredArr) :
    '''
    For all prompts, it selects the best response with a certain value of N
    (in regards to its reward) and groups all selected responses in an array. 
    
    Input : 
        nValuesArr: array containing the values of N used for the best-of-n sampling
        beta : 
        scoredArr: array formed from the (alternate_)scoredGenerations.csv  

    Output : 
        selectedCandidates : array containing the selected responses 
    '''

    for i in range(20) :    # there are 20 prompts
        curPromptId = i+1

        allSelected = []
        meanSentiments = []
        meanPerplexities = []
        meanKLs = []
        meanRewards = []
        for curN in nValuesArr : 
            curSelection, meanSentiment, meanPerplexity, meanKL, meanReward = getBestOfN(curPromptId, beta, curN, scoredArr)
            
            allSelected.append(curSelection)
            meanSentiments.append(meanSentiment)
            meanPerplexities.append(meanPerplexity)
            meanKLs.append(meanKL)
            meanRewards.append(meanReward)

    return allSelected, meanSentiments, meanPerplexities, meanKLs, meanRewards


def build_reference_responses(prompts, model_fn):
        reference_responses = []

        for prompt_row in prompts[1:]:
            prompt_text = prompt_row[1]

            response = model_fn(prompt_text)
            reference_responses.append(response)

        return reference_responses
