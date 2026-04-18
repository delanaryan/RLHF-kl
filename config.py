# file for constants/settings
# keeps track of all adjustable settings 
    # model name
    # N values
    # max tokens
    # temperature
    # file paths
    # which sentiment model to use

# N_VALUES = [1, 2, 4, 8]     # if we do N=16, then it would take too long to run
MAX_N = 16
PROMPT_PATH = 'data/prompts.csv'
RAW_GENERATIONS_PATH = 'data/rawGenerations.csv'
SCORED_GENERATIONS_PATH = 'data/scoredGenerations.csv'