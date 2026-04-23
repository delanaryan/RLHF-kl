# helper functions that don't fit anywhere else 
    # load CSV
    # save CSV
    # clean text
    # seed randomness
    # make folders if missing

import csv

def csvToArr (fileName) : 
    # Convert Path to string if needed
    filePath = str(fileName)
    with open(filePath, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        array = list(reader)
    return array

    