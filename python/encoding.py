import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
import config as cfg
import numpy as np
import keras
import fastText
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

class Data:
    def __init__(self, encoding = "char",
                 nrNgram = 2,
                 dimension = None,
                 encoding_type = "pad",
                 force_rerun = False,
                 nr_datapoints = 1000):

        self.path_to_data = "data/raw_data/dataset"+str(nr_datapoints)+".csv"
        self.force_rerun = force_rerun
        self.encoding = encoding
        self.nrNgram = nrNgram
        self.dimension = dimension
        self.encoding_type = encoding_type
        #self.nr_datapoints = nr_datapoints
    #    self.path_to_data = path_to_data
        df = pd.read_csv(self.path_to_data)

        self.raw_data = df["sentence"].values
        self.labels = df["language"].values
        self.nr_datapoints = len(self.labels)
        self.num_classes = len(cfg.languages)

        ## Encode the labels to categorical
        self.y = [cfg.lang_to_label[label] for label in self.labels]
        print("Number of labels", len(self.y))

        ## My own code
        if self.encoding in ["char","word"]:
            self.make_ngram()
            self.make_vocab()
            self.max_document_length = max([len(document) for document in self.data])
            print("Max_document_length",self.max_document_length)
            if self.encoding_type == "one_hot":
                self.encode_datapoints_one_hot()
            elif self.encoding_type == "pad":
                self.encode_datapoints()
        ## Fasttext encoding
        elif self.encoding in ["cbow", "skipgram"]:
            self.fasttext_encoder()

        ## sklearn
        # elif self.encoding in ["tfidf", "count"]:
        elif self.encoding == "tfidf":
            vectorizer = TfidfVectorizer(ngram_range=(1, 1), analyzer="word", max_df=1.0, min_df=1,
                                        max_features=self.dimension, vocabulary=None, binary=False)

            X = vectorizer.fit_transform(self.raw_data)
            self.X = X.toarray()
            self.freqlist = vectorizer.get_feature_names()
            self.dimension = self.X.shape[1]
            print("Dataset encoded, shape: ", self.X.shape)

        elif self.encoding == "count":
            vectorizer = CountVectorizer(ngram_range=(1, 1), analyzer="word", max_df=1.0, min_df=1,
                                        max_features=self.dimension , vocabulary=None, binary=False)

            X = vectorizer.fit_transform(self.raw_data)
            self.X = X.toarray()
            self.freqlist = vectorizer.get_feature_names()
            self.dimension = self.X.shape[1]
            print("Dataset encoded, shape: ", self.X.shape)

        else:
            print(self.encoding)
            print("WARNING: Encoding invalid!")

        ## do test train split.
        if self.encoding == None:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.raw_data, self.y, test_size=0.2, random_state=42)
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)


    def make_vocab(self):
        """
        returns a dictionary with each token in the vocabulary
        is mapped to a unique integer ordered after frequency
        """
        from collections import Counter

        cnt = Counter()

        for document in self.data:
            cnt.update(document)

        cntSortedByFreq = sorted(cnt,key=cnt.get,reverse=True)
        self.freqlist = cntSortedByFreq
        self.vocab_size = len(cntSortedByFreq)
        self.vocab_dict = dict(zip(["<PAD>"] + cntSortedByFreq, np.arange(self.vocab_size+1)))
        print("Vocabulary made of size", self.vocab_size)


    def make_ngram(self):

        def make_word_ngram(text, n = 2):
            """ Convert text into word ngrams. """
            tokens = [token for token in text.split(" ") if token != ""]
            ngrams = zip(*[tokens[i:] for i in range(n)])
            return [" ".join(ngram) for ngram in ngrams]

        def make_char_ngram(text, n = 2):
           """ Convert text into character ngrams. """
           return ["".join(j) for j in zip(*[text[i:] for i in range(n)])]

        if self.encoding == "char":
            self.data = [make_char_ngram(document, n=self.nrNgram) for document in self.raw_data]
        if self.encoding == "word":
            self.data = [make_word_ngram(document, n=self.nrNgram) for document in self.raw_data]

        print("Text processed into ", self.encoding, self.nrNgram)

    def encode_datapoints(self):

        def encode_datapoint(vocab_dict , document):
            """Given a vocabulary dict that maps to unique int returns the
            encoded document."""
            return [vocab_dict[d] for d in document]

        encoded_documents = [encode_datapoint(self.vocab_dict,d) for d in self.data]

        self.X = keras.preprocessing.sequence.pad_sequences(encoded_documents,
                                                       value=0,
                                                       padding='post',
                                                       maxlen=self.max_document_length)

        self.dimension = self.X.shape[1]

        print("Encoding made, shape:", self.X.shape)

    def encode_datapoints_one_hot(self):

        X_one_hot = np.zeros((len(self.y), self.vocab_size))

        self.dimension = self.vocab_size

        for i, datapoint in enumerate(self.data):
            for feature in datapoint:
                X_one_hot[i, self.vocab_dict[feature]-1] +=1

        self.X = X_one_hot

    def fasttext_encoder(self, save = True):


        datafolder = "data/datasets/"
        model_name = "fasttext" + self.encoding + str(self.dimension) + "size" + str(self.nr_datapoints)

        fname = datafolder + model_name + ".npy"

        # If this have already meen calculated then load that.
        if os.path.isfile(fname) and not self.force_rerun:
            self.X = np.load(fname)
            print("Dataset loaded", fname)

        else: # Create fasttext encoding
           # print("Creating corpus")
           # corpus = " ".join(data.valu#es)
            if self.dimension == None:
                self.dimension = 50

            print("Now training fasttextmodel:")
            fastTextmodel = fastText.train_unsupervised(cfg.datafolder + "alldata" + str(self.nr_datapoints) + ".txt",
                                            model = self.encoding,
                                            dim = self.dimension,
                                            wordNgrams = self.nrNgram)

            print("Now doing fasttext encoding:")
            n = len(self.y)
            m = fastTextmodel.get_dimension()
            X = np.zeros((n,m))

            for i, datapoint in enumerate(self.raw_data):
                encoding = fastTextmodel.get_sentence_vector(datapoint)
                X[i,:] = encoding

            self.X = X

            print("Fasttext encoding made, shape:", self.X.shape)
            if save:
                ## save fasttext encoding
                np.save(fname, X)
                print("Encoded dataset saved to: ", fname)
