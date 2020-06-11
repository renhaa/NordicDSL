import config as cfg
import os
import sys
import argparse
import glob
import re
import pandas as pd
from sklearn.model_selection import train_test_split

import nltk

nltk.download('punkt')

import config as cfg


def load_raw_data(fname):
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


def get_sentences(raw_data):

    ## Filter Lengths of sentences
    MIN_LENGTH = 10
    MAX_LENGTH = 300

    raw_data = raw_data.split('\n')
    documents = []
    for datapoint in raw_data:
            sentences = nltk.sent_tokenize(datapoint)
            for sentence in sentences:
                ## Clean strings
                sentence = clean_str(sentence)
                if len(sentence) > MIN_LENGTH and len(sentence) < MAX_LENGTH:
                        documents.append(sentence)
                   
    return documents


def preprocess(src_dir, out_dir):
    fs = glob.glob(f"{src_dir}*")
    langs = [f.split("/")[-1].split(".")[0] for f in fs]
 
    all_documents = []
    all_labels = []
    for lang, fname in zip(langs, fs):
        print("Now doing ", lang)
        # load data
        raw_data = load_raw_data(fname)

        # replace newlines by period
        raw_data = raw_data.replace("\n", ". ")

        # split into sentences
        documents = get_sentences(raw_data)

        # construct the dataframe
        langid = [lang]*len(documents)

        all_documents += documents
        all_labels += langid
    
    ## test,train split
    X_train, X_test, y_train, y_test = train_test_split(all_documents,
                                                        all_labels,
                                                        test_size=0.05)

    ## Save to .csv and .txt format
    data = {"language": y_train,"sentence": X_train, }
    df = pd.DataFrame(data)
    df.to_csv(f"{out_dir}train.csv", mode='w', index=False)

    ## Save to .csv and .txt format
    data = {"language": y_test, "sentence": X_test, }
    df = pd.DataFrame(data)
    df.to_csv(f"{out_dir}test.csv", mode='w', index=False)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Preprocess raw data. Extracts sentences from each file in src_dir and produces a training and test set.')
    parser.add_argument('src_dir', help='Source dir. Path to folder with raw data files. The script will process all .txt files in the folder an regard the filename as the language label. E.g dk.txt is a file with danish text.')
    parser.add_argument('out_dir', help='Output dir. Path to folder where the processed data will be saved. The script will generate two files train.csv and test.csv.')
    parser.add_argument('--train_test_ratio', default=0.05, help=' Test / Train split ratio. Size of the test set as a fraction of the total data.')

    args, other_args = parser.parse_known_args()

    if not os.path.isdir(args.out_dir):
        os.mkdir(args.out_dir)
    preprocess(args.src_dir, args.out_dir)


