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
    generate.getResponses(promptArr, config.N, config.RAW_GENERATIONS_PATH)

    df = pd.read_csv(config.RAW_GENERATIONS_PATH)
    print(df.head())
    print(df.shape)