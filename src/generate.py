# send prompts to Llama
# generate multiple candidates per prompt
# return/save outputs

import pandas as pd
import ollama
import csv

def generateSingleResponse(prompt):
    '''
    Generaates a response using the ollama API for a given prompt.
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