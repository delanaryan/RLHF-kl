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

def getResponses(promptArray, n, rawGenerationCSVPath) :
    with open(rawGenerationCSVPath, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["prompt_id", "n_value", "candidate_id", "response"])

        for row in promptArray[1:]:
            prompt_id = row[0]
            prompt = row[1]

            for i in range(n):
                curResponse = generateSingleResponse(prompt)
                writer.writerow([prompt_id, n, i + 1, curResponse])

    return