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
        df_test = pd.read_csv("tfidf.csv")
    else:
        df_test = pd.read_csv("split_tfidf/" + fold + "/test_data.csv")

    X_test = np.asarray(df_test.iloc[:, :-2])
    y_test = df_test.iloc[:, -1]

    y_test = label_binarize(y_test, classes=[1, 2, 3, 4, 5])

    results = pd.read_csv(r'grid_search_results/' + 'results.csv')
    results = results[results.predictor_type != 'log_regression']

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
    plt.savefig("plots/ROC_tfidf.pdf")


def get_roc_auc(y_test, y_predict):
    fpr = dict(); tpr = dict(); roc_auc = dict()
    for rating in range(5):
        fpr[rating], tpr[rating], _ = roc_curve(y_test[:, rating], y_predict[:, rating])
        roc_auc[rating] = auc(fpr[rating], tpr[rating])
    return fpr, tpr, roc_auc


def cross_validate_tfidf():
    """ Referenced implementation: https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html """
    X_test, y_test, results = getData(fold="2")

    # best_model = findBestModel(X_test, y_test, results)
    # best_model = 'grid_search_results/1574183552.h5'
    model = load_model("grid_search_results/1574183552.h5")

    y_predict = model.predict(X_test)
    fpr, tpr, auc = get_roc_auc(y_test, y_predict)
    plotROC(fpr, tpr, auc)


if __name__ == '__main__':
    cross_validate_tfidf()
