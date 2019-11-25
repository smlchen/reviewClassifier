from os.path import isfile

import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

import matplotlib.pyplot as plt
import pandas as pd
from tensorflow_core.python.keras.models import load_model


def findBestModel(X_test, y_test, results):
    # Find best model to work with against the 1st fold of data.
    best_model = ""
    most_AUC = 0
    for _, row in results.iterrows():
        file_name = r'grid_search_results/' + row['file_name']
        if not isfile(file_name):
            continue

        model = load_model(file_name)
        y_predict = model.predict(X_test)
        fpr, tpr, auc = get_roc_auc(y_test, y_predict)

        if most_AUC < sum(auc.values()):
            most_AUC = sum(auc.values())
            best_model = file_name
    return best_model


def getData(fold="1"):
    if fold == "all":
        df_test = pd.read_csv("doc2vec.csv")
    else:
        df_test = pd.read_csv("split_doc2vec/" + fold + "/test_data.csv")

    # FIXME: explode lists (vectors) to different columns
    """ Currently, the vector = [-0.3, 1.7, 6.6 ...] so panda only reads in the vector as a single feature. It should 
    really separate the vector into the corresponding features (i.e. for sample_0: X_test[0] = -.3, X_test[1] = 1.7 ..."""
    X_test = df_test.iloc[:, -1]
    y_test = df_test.iloc[:, 1]
    y_test = label_binarize(y_test, classes=[1, 2, 3, 4, 5])

    results = pd.read_csv(r'doc2vec_grid_search_results/' + 'results.csv')

    return X_test, y_test, results


def plotROC(fpr, tpr, auc):
    plt.figure()
    for rating in range(5):
        plt.plot(fpr[rating], tpr[rating], label='ROC curve: Rating {0} (area = {1:0.2f})'.format(rating + 1, auc[rating]))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Ratings')
    plt.legend(loc="lower right")
    plt.show()


def get_roc_auc(y_test, y_predict):
    fpr = dict(); tpr = dict(); roc_auc = dict()
    for rating in range(5):
        fpr[rating], tpr[rating], _ = roc_curve(y_test[:, rating], y_predict[:, rating])
        roc_auc[rating] = auc(fpr[rating], tpr[rating])
    return fpr, tpr, roc_auc


def cross_validate_tfidf():
    """ Referenced implementation: https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html """
    X_test, y_test, results = getData(fold="all")

    # best_model = findBestModel(X_test, y_test, results)
    model = load_model("doc2vec_grid_search_results/1574409738.h5")

    y_predict = model.predict(X_test)
    fpr, tpr, auc = get_roc_auc(y_test, y_predict)
    plotROC(fpr, tpr, auc)


if __name__ == '__main__':
    cross_validate_tfidf()
