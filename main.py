# load prompts
# generate responses
# score responses
# save results
# make plots

import config

from src import generate
from src import score
from src import plot
from src import utils

import pandas as pd

if __name__ == "__main__":
    promptArr = utils.csvToArr(config.PROMPT_PATH)
    alternatePromptArr = utils.csvToArr(config.ALTERNATE_PROMPT_PATH)
    # generate.getAllResponses(promptArr, config.N, config.RAW_GENERATIONS_PATH)
    # generate.getResponsesChunk(promptArr, 1, 20, 16)
    # generate.getAllResponses(alternatePromptArr, config.MAX_N, config.ALTERNATE_RAW_GENERATIONS_PATH)

    # --- Best-of-N Sampling ---
    # Uncomment to use Best-of-N sampling with RoBERTa proxy scoring
    # for the factual prompts
    generate.getAllBestOfN(promptArr, config.BEST_OF_N, config.BEST_OF_N_SELECTIONS_PATH, verbose=True)
    responsesArr = utils.csvToArr(config.RAW_GENERATIONS_PATH)
    sentimentArr = score.getAllSentimentScores(responsesArr)
    perplexities = score.getAllPerplexities(responsesArr)
    score.fillScoredGenerations(config.SCORED_GENERATIONS_PATH, sentimentArr, perplexities)

    # for the alternate prompts 
    # generate.getAllBestOfN(alternatePromptArr, config.BEST_OF_N, config.ALTERNATE_BEST_OF_N_SELECTIONS_PATH, verbose=True)
    # alternate_responsesArr = utils.csvToArr(config.ALTERNATE_RAW_GENERATIONS_PATH)
    # alternate_sentimentArr = score.getAllSentimentScores(alternate_responsesArr)
    # alternate_perplexities = score.getAllPerplexities(alternate_responsesArr)
    # score.fillScoredGenerations(config.ALTERNATE_SCORED_GENERATIONS_PATH, alternate_sentimentArr, alternate_perplexities)