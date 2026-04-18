# send prompts to Llama
# generate multiple candidates per prompt
# return/save outputs

import pandas as pd
import ollama
import csv

def generateSingleResponse(prompt):
    try:
        response = ollama.chat(
            model='mistral', 
            messages=[{'role': 'user', 'content': prompt}],
        )
        return response['message']['content']
    
    except Exception as e:
        return f"Error: {e}"


def getAllResponses(promptArray, maxN, rawGenerationCSVPath) :
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