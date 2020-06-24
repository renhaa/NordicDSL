import os
import argparse
import pandas as pd 
import fasttext
import glob

def prepare_data_for_fasttext(data):

    for lang, text in zip(data["language"], data["sentence"]):
        line = "__langid__" + lang + " " + text
        with open("fasttext.tmp", "a+") as f:
            f.write(line + "\n")

    print("Fasttext format of data saved to ", "fasttext.tmp")


def train_fasttext_model(args):
    
    fs = glob.glob(f"{args.src_dir}*")
    trainfile = [f for f in fs if "train" in f][0]
    data = pd.read_csv(trainfile)
    prepare_data_for_fasttext(data) 

    fastTextmodel = fasttext.train_supervised("fasttext.tmp",
                                              dim=args.dim,
                                              ws=args.ws,
                                              wordNgrams=args.wordNgrams,
                                              minn=args.minn,
                                              maxn=args.maxn,
                                              label = "__langid__")

    fastTextmodel.save_model(args.src_dir + "fasttextmodel.ftz")
    print("fasttext model saved to", args.src_dir + "fasttextmodel.ftz")
    os.remove("fasttext.tmp")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a FastText model for DSL')
    parser.add_argument('src_dir', help='Path folder with test and train .csv files. We expect the language label in the first column and sentences in the second.')

    parser.add_argument('--lr', default = 0.1, help='learning rate [0.1]')
    parser.add_argument('--dim', default=100, help='size of word vectors [100]')
    parser.add_argument('--ws', default=5, help='size of the context window [5]')
    parser.add_argument('--wordNgrams', default=1, help='max length of word ngram[1]')
    parser.add_argument('--minn', default=2,help='min length of char ngram [2]')
    parser.add_argument('--maxn', default=5,help='max length of char ngram [5]')

    args, other_args = parser.parse_known_args()

    train_fasttext_model(args)
