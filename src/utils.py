# helper functions that don't fit anywhere else 
    # load CSV
    # save CSV
    # clean text
    # seed randomness
    # make folders if missing

import csv

def csvToArr (fileName) : 
    with open(fileName, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        array = list(reader)
    return array

    