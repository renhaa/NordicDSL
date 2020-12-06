# import re
import itertools
import nltk
import random
import numpy as np
import random
import os.path

import keras
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

import sklearn.metrics

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

import glob

from encoding import *


fasttext_encodings = ["cbow","skipgram"]
keras_encodings = ["binary", "count", "tfidf", "freq"]

def load_dataset(params, filename = "data/raw_data/dataset.csv",
                test_size = 0.2):

    df = pd.read_csv(filename)

    if params["encoding"] in keras_encodings:
        X, y = keras_encoder(df, output_dimension = int(params["dimension"]), mode = params["encoding"])
    elif params["encoding"] in fasttext_encodings:
        X, y = fasttext_encoder(params, save = True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    return X_train, X_test, y_train, y_test

def params_to_str(params):
    paramlist = []
    for key in params.keys():
        paramlist.append(params[key])
    return ",".join(paramlist)

def write_result(params):

    fname = "results/evaluation"
    result = params_to_str(params)

    # Write header
    if not os.path.isfile(fname):
        with open(fname, "a+") as f:
            f.write(",".join(params.keys())+"\n")

    with open(fname, "a+") as f:
        f.write(result +"\n")
    print("result saved to:", fname)


def SVM(X_train, y_train, X_test):
    """
    https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC
    """

    clf = LinearSVC(random_state=0, tol=1e-5)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return y_pred

def LogRegression(X_train, y_train, X_test):

    clf = LogisticRegression(random_state=0,
                              solver='lbfgs',
                              multi_class='multinomial')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return y_pred

def NeuralNetwork(X_train, y_train, X_test, verbose = False):

    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train))

    MLP = [keras.layers.Dense(64 , input_shape=(input_dim,), activation="relu"),
           keras.layers.Dense(64, input_shape=(input_dim,), activation="relu"),
           keras.layers.Dense(num_classes, activation=tf.nn.softmax)]



    model = keras.Sequential(MLP)

    model.compile(loss="mean_squared_error",
                  optimizer="adam",
                  metrics=['acc'],
                  #metrics=self.metrics)
                 )
    if verbose:
        model.summary()

    nrEpochs = 10

    y_train_categorical = keras.utils.to_categorical(y_train)

    history = model.fit(X_train, y_train_categorical,
                        epochs = nrEpochs,
                        #batch_size=20,
                        validation_split=0.2,
                        verbose = verbose)


    model.predict(X_test)

    y_pred = model.predict_classes(X_test)

    return y_pred

def plot_keras_training(history):

    history_dict = history.history
    history_dict.keys()


    acc = history_dict['acc']
    val_acc = history_dict['val_acc']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

def evaluation(y_test, y_pred, show = False):

    confusion_matrix = sklearn.metrics.confusion_matrix(y_test,y_pred)
    accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)
    f1_score = sklearn.metrics.f1_score(y_test, y_pred, average = "weighted")
    precision = sklearn.metrics.precision_score(y_test, y_pred,average = "weighted")
    recall = sklearn.metrics.recall_score(y_test, y_pred, average = "weighted")

    if show:
        sn.set(font_scale=1)#for label size
        sn.heatmap(confusion_matrix, annot=True,annot_kws={"size": 16})# font
        plt.title("accuracy = " + str(accuracy),)
        plt.show()

    metrics = {"confusion_matrix": confusion_matrix,
               "accuracy": accuracy,
               "f1_score": f1_score,
               "precision": precision,
               "recall": recall}

    return metrics

def run_model(params):
    print("Now doing: \n")
    print(params)
    X_train, X_test, y_train, y_test = load_dataset(params)

    if params["model_name"] == "logreg":
        y_pred= LogRegression(X_train, y_train, X_test)
    if params["model_name"] == "svm":
        y_pred = SVM(X_train, y_train, X_test)
    if params["model_name"] == "nn":
        y_pred = NeuralNetwork(X_train, y_train, X_test)

    metrics = evaluation(y_test,y_pred)

    params['accuracy'] =  str(metrics['accuracy'])
    write_result(params)
