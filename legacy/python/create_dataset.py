import pandas as pd
import nltk
import re

import config as cfg

def load_raw_data(language):
    fname = cfg.datafolder + language + ".txt"
    with open(fname) as f:
        raw_data = f.read()
    return raw_data

def clean_str(string):
    """ Clean string """

    ## Only use lower case
    filtered_string = string.lower()
    ## filter all that is not in the char_set
    filtered_string = re.sub(r'[^' + cfg.char_set + ']', ' ', filtered_string)
    ## remove double spaces
    filtered_string = re.sub(' +', ' ', filtered_string)

    return filtered_string

def get_documents_from_raw_data(raw_data):

    ## Filter Lengths of sentences
    MIN_LENGTH = 10
    MAX_LENGTH = 300

    raw_data = raw_data.split('\n')
    documents = []
    c=0
    for datapoint in raw_data:
        if c<cfg.nr_datapoints:
            sentences = nltk.sent_tokenize(datapoint)
            for sentence in sentences:

                ## Clean strings
                sentence = clean_str(sentence)
                if len(sentence)>MIN_LENGTH and len(sentence)<MAX_LENGTH:
                    if c<cfg.nr_datapoints:
                        documents.append(sentence)
                    c+=1
    return documents
def combine_data_to_csv():

    all_documents = []
    all_labels = []
    for language in cfg.languages:
        print("Now doing ", language)
        # load data
        raw_data = load_raw_data(language)

        # replace newlines by period
        raw_data = raw_data.replace("\n", ". ")

        # split into sentences
        documents = get_documents_from_raw_data(raw_data)

        # construct the dataframe
        langid = [language]*len(documents)

        all_documents += documents
        all_labels += langid

    ## make pandas dataframe
    data = {"sentence": all_documents, "language": all_labels}
    df = pd.DataFrame(data)

    fname = cfg.datafolder + "dataset"+str(cfg.nr_datapoints)+".csv"
    df.to_csv(fname)
    print("Dataset saved to", fname, "Number of datapoints:", len(df))
    fname = cfg.datafolder + "alldata"+str(cfg.nr_datapoints)+".txt"
    with open(fname, "w+") as f:
        f.write("\n".join(all_documents))
    print("Text file with all data saved to", fname)

if __name__ == '__main__':
    combine_data_to_csv()
