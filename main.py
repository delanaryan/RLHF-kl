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
    # TRANSFORM THE PROMPT CSVs TO ARRAYS
    promptArr = utils.csvToArr(config.PROMPT_PATH)
    alternate_PromptArr = utils.csvToArr(config.ALTERNATE_PROMPT_PATH)

    # GENERATE THE RESPONSES 
    # generate.getAllResponses(promptArr, config.MAX_N, config.RAW_GENERATIONS_PATH)
    # generate.getAllResponses(alternatePromptArr, config.MAX_N, config.ALTERNATE_RAW_GENERATIONS_PATH)

    # TRANSFORM THE RESPONSES' CSVs INTO ARRAYS
    responsesArr = utils.csvToArr(config.RAW_GENERATIONS_PATH)
    alternate_responsesArr = utils.csvToArr(config.ALTERNATE_RAW_GENERATIONS_PATH)


    # BEST OF N SAMPLING
        # for the factual prompts :
    score.getScoredGenerations(config.SCORED_GENERATIONS_PATH, promptArr, responsesArr)
    scoredArr = utils.csvToArr(config.SCORED_GENERATIONS_PATH)

    # generate.getAllBestOfN(20, config.N_VALUES, scoredArr, config.BEST_OF_N_SELECTIONS_PATH)

        # for the alternate prompts :
    score.getScoredGenerations(config.ALTERNATE_SCORED_GENERATIONS_PATH, alternate_PromptArr, alternate_responsesArr)
    alternate_scoredArr = utils.csvToArr(config.ALTERNATE_SCORED_GENERATIONS_PATH)
    # generate.getAllBestOfN(20, config.N_VALUES, alternate_scoredArr, config.ALTERNATE_BEST_OF_N_SELECTIONS_PATH)
