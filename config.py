# file for constants/settings
# keeps track of all adjustable settings 
    # model name
    # N values
    # max tokens
    # temperature
    # file paths
    # which sentiment model to use

from pathlib import Path

# Get the directory where config.py is located (project root)
PROJECT_ROOT = Path(__file__).parent

N_VALUES = [1, 2, 4, 8, 16]
MAX_N = 16
BEST_OF_N = 8  # Number of candidates to generate in Best-of-N sampling

PROMPT_PATH = PROJECT_ROOT / 'data' / 'prompts.csv'
RAW_GENERATIONS_PATH = PROJECT_ROOT / 'data' / 'rawGenerations.csv'
SCORED_GENERATIONS_PATH = PROJECT_ROOT / 'data' / 'scoredGenerations.csv'
BEST_OF_N_SELECTIONS_PATH = PROJECT_ROOT / 'data' / 'bestOfNSelections.csv'

ALTERNATE_PROMPT_PATH = PROJECT_ROOT / 'data' / 'alternate_prompts.csv'
ALTERNATE_RAW_GENERATIONS_PATH = PROJECT_ROOT / 'data' / 'alternate_rawGenerations.csv'
ALTERNATE_SCORED_GENERATIONS_PATH = PROJECT_ROOT / 'data' / 'alternate_scoredGenerations.csv'
ALTERNATE_BEST_OF_N_SELECTIONS_PATH = PROJECT_ROOT / 'data' / 'alternate_bestOfNSelections.csv'