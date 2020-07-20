import sklearn
import pandas as pd
import config as cfg
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np

def evaluation(y_test, y_pred, fname = False, show = True, verbose = True):
    confusion_matrix = sklearn.metrics.confusion_matrix(y_test,y_pred)
    accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)
    f1_score = sklearn.metrics.f1_score(y_test, y_pred, average = "weighted")
    precision = sklearn.metrics.precision_score(y_test, y_pred,average = "weighted")
    recall = sklearn.metrics.recall_score(y_test, y_pred, average = "weighted")

    if fname:
        df_cm = pd.DataFrame(confusion_matrix, index = [i for i in cfg.languages],
                  columns = [i for i in cfg.languages])
        sn.set(font_scale=1)#for label size
        sn.heatmap(df_cm, annot=True,annot_kws={"size": 12}, fmt='g')# font
        plt.title("Accuracy = " + str(accuracy),size = 18)
        plt.ylabel("Actual Label")
        plt.xlabel("Predicted Label")
        plt.savefig(fname)
        plt.show()

    metrics = {"confusion_matrix": confusion_matrix,
               "accuracy": accuracy,
               "f1_score": f1_score,
               "precision": precision,
               "recall": recall}
    wrong_ixd = [i for i,pred in enumerate(y_pred) if not pred == y_test[i]]
    wrong_pred = np.array(y_pred)[wrong_ixd]

    metrics["wrong_ixd"] = wrong_ixd
    metrics["wrong_pred"] = wrong_pred
    # if verbose:
    #     print(metrics)
    #     print("wrong_ixd",wrong_ixd)
    #     print("wrong_pred",wrong_pred)


    return metrics
