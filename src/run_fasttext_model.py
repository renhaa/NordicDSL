import os
import argparse
import pandas as pd 
import fasttext
import glob


class FastTextModel:
    def __init__(self, model_path):
        self.model = fasttext.load_model(model_path)

    def predict(self,sentece):
        return self.model.predict(sentece)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Use a trained FastText model for DSL')
    parser.add_argument('model_path', help='Path to trained FastText Model in the format. path/to/fasttextmodel.ftz.')
    parser.add_argument('src', help='Path to a .txt file. The script with read each line and return labels and confidences.')


    args, other_args = parser.parse_known_args()

    ftmodel = FastTextModel(args.model_path)

    with open(args.src) as f:
        lines = f.readlines()
    # remove \n newlines. 
    lines = [l.strip() for l in lines]

    pred = ftmodel.predict(lines)
    print(pred)
  