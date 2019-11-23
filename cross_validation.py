from os.path import isfile

import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

import matplotlib.pyplot as plt
import pandas as pd
from tensorflow_core.python.keras.models import load_model


def getData(fold="1"):
    df_train = pd.read_csv("split_tfidf/" + fold + "/training_data.csv")
    df_test = pd.read_csv("split_tfidf/" + fold + "/test_data.csv")

    X_train = np.asarray(df_train.iloc[:, :-2])
    y_train = df_train.iloc[:, -1]
    X_test = np.asarray(df_test.iloc[:, :-2])
    y_test = df_test.iloc[:, -1]

    y_train = label_binarize(y_train, classes=[1, 2, 3, 4, 5])
    y_test = label_binarize(y_test, classes=[1, 2, 3, 4, 5])

    results = pd.read_csv(r'grid_search_results/' + 'results.csv')
    results = results[results.predictor_type != 'log_regression']

    return X_train, y_train, X_test, y_test, results


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


def calcAvgOfDiffROCAUC(fpr_list, tpr_list, auc_list):
    # FIXME: stub
    return avg_fpr, avg_tpr, avg_auc


def getROCForAllFolds(best_model):
    model = load_model(best_model)

    fpr_list = []; tpr_list = []; auc_list = []
    for fold in range(1, 6):
        _, _, X_test, y_test, results = getData(str(fold))
        y_predict = model.predict(X_test)
        fpr, tpr, auc = get_roc_auc(y_test, y_predict)

        # fpr = dict of ndarray. auc = dict of floats
        fpr_list.append(fpr.values());
        tpr_list.append(tpr.values());
        auc_list.append(auc.values())

    # FIXME: All fpr, tpr, auc dictionaries from each K-fold are inserted into their corresponding
    # FIXME: list, but I don't know how to take the mean of each index from several dictionaries.
    avg_fpr, avg_tpr, avg_auc = calcAvgOfDiffROCAUC(fpr_list, tpr_list, auc_list)
    plotROC(avg_fpr, avg_tpr, avg_auc)


def cross_validate_tfidf():
    """ Referenced implementation: https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html """
    _, _, X_test, y_test, results = getData(fold="1")

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

    model = load_model(best_model)
    y_predict = model.predict(X_test)
    fpr, tpr, auc = get_roc_auc(y_test, y_predict)
    plotROC(fpr, tpr, auc)
    # FIXME: need to average the ROC / AUC curves with the different folds.
    # FIXME: getROCForAllFolds(best_model)


if __name__ == '__main__':
    cross_validate_tfidf()
