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
    # generate.getAllResponses(promptArr, config.N, config.RAW_GENERATIONS_PATH)
    # generate.getResponsesChunk(promptArr, 1, 5, 16)
    # generate.getResponsesChunk(promptArr, 6, 10, 16)
    # generate.getResponsesChunk(promptArr, 11, 14, 16)
    # generate.getResponsesChunk(promptArr, 15, 20, 16)

    # --- Best-of-N Sampling ---
    # Uncomment to use Best-of-N sampling with RoBERTa proxy scoring
    # generate.getAllBestOfN(promptArr, config.BEST_OF_N, config.BEST_OF_N_SELECTIONS_PATH, verbose=True)

    responsesArr = utils.csvToArr(config.RAW_GENERATIONS_PATH)

    sentimentArr = score.getAllSentimentScores(responsesArr)
    perplexities = score.getAllPerplexities(responsesArr)
    score.fillScoredGenerations(config.SCORED_GENERATIONS_PATH, sentimentArr, perplexities)

    # debugging
    # df = pd.read_csv(config.RAW_GENERATIONS_PATH)
    # print(df.head())
    # print(df.shape)