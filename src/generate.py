# send prompts to Llama
# generate multiple candidates per prompt
# return/save outputs

import pandas as pd
import ollama
import csv
from src.score import getSentimentScore

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

def generateBestOfN(prompt, N, verbose=False):
    '''
    Implements Best-of-N sampling with proxy scoring using RoBERTa sentiment.
    Generates N distinct responses and selects the one with highest positivity score.
    
    input: a single prompt, the number of candidates N to generate, verbose flag for debugging (false by default, making it true will print out the generated candidates and their scores)
    output: a dictionary with the best response and its sentiment score
    '''
    candidates = []
    
    for i in range(N):
        response = generateSingleResponse(prompt)
        sentiment_score = getSentimentScore(response)
        candidates.append({
            'candidate_id': i + 1,
            'response': response,
            'sentiment_score': sentiment_score
        })
        
        if verbose:
            print(f"Generated candidate {i+1}/{N} - Sentiment Score: {sentiment_score:.4f}")
    
    best_candidate = max(candidates, key=lambda x: x['sentiment_score']) # Select best response based on highest positivity score
    
    if verbose:
        print(f"\nBest candidate selected: #{best_candidate['candidate_id']} with score {best_candidate['sentiment_score']:.4f}")
        print(f"Response: {best_candidate['response']}\n")
    
    return best_candidate


def getAllBestOfN(promptArray, N, outputCSVPath, verbose=False):
    '''
    Applies Best-of-N sampling to all prompts and saves results to CSV.
    
    input: an array of prompts, the number of candidates N to generate per prompt, and the path to save the best-of-N selections
    output: a CSV file with the best response selected for each prompt
    '''
    with open(outputCSVPath, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["prompt_id", "prompt", "best_response", "sentiment_score", "N"])
        
        for row in promptArray[1:]:
            prompt_id = row[0]
            prompt = row[1]
            
            best_candidate = generateBestOfN(prompt, N, verbose=verbose)
            
            writer.writerow([
                prompt_id,
                prompt,
                best_candidate['response'],
                best_candidate['sentiment_score'],
                N
            ])
            
            if verbose:
                print(f"Saved best-of-{N} for prompt {prompt_id}")
    
    return