from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

import matplotlib.pyplot as plt
import pandas as pd

from build_model import build_model


def plotROC(fpr, tpr, roc_auc):
    plt.figure()
    for i in range(5):
        plt.plot(fpr[i], tpr[i], label='ROC curve: Rating {0} (area = {1:0.2f})'.format(i + 1, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves of Ratings')
    plt.legend(loc="lower right")
    plt.show()


def cross_validate_tfidf():
    """ Referenced implementation: https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html """

    df_train = pd.read_csv("split_tfidf/1/training_data.csv")  # FIXME: Only using 1st K-Fold. Use all later when working
    df_test = pd.read_csv("split_tfidf/1/test_data.csv")

    X_train = df_train.iloc[:, :-2]
    y_train = df_train.iloc[:, -1]
    X_test = df_test.iloc[:, :-2]
    y_test = df_test.iloc[:, -1]

    y_train = label_binarize(y_train, classes=[1, 2, 3, 4, 5])
    y_test = label_binarize(y_test, classes=[1, 2, 3, 4, 5])

    df = pd.read_csv("grid_search_results/results.csv")
    for para in df.iterrows():
        model = build_model(para['layers'], para['nodes'], X_train.shape[1], 5, loss=para['loss'], clipnorm=1.0)
        # FIXME: NaN values when you put smaller batchsize / more epoches. y_score explodes in value, not sure why?
        # FIXME: However, clipnorm should remove them, so I'm not sure why it stil explodes
        history = model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=2)
        y_score = model.predict(X_test)

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(5):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        plotROC(fpr, tpr, roc_auc)


if __name__ == '__main__':
    cross_validate_tfidf()


# def cross_validate_tfidf():
#     """ Referenced implementation: https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html """
#
#     df_train = pd.read_csv("split_tfidf/1/training_data.csv")  # FIXME: Only using 1st K-Fold. Use all later when working
#     df_test = pd.read_csv("split_tfidf/1/test_data.csv")
#
#     X_train = df_train.iloc[:, :-2]
#     y_train = df_train.iloc[:, -1]
#     X_test = df_test.iloc[:, :-2]
#     y_test = df_test.iloc[:, -1]
#
#     y_train = label_binarize(y_train, classes=[1, 2, 3, 4, 5])
#     y_test = label_binarize(y_test, classes=[1, 2, 3, 4, 5])
#
#     model = build_model(2, 4, X_train.shape[1], 5, clipnorm=1.0)
#     # FIXME: NaN values when you put smaller batchsize / more epoches. y_score explodes in value, not sure why?
#     # FIXME: However, clipnorm should remove them, so I'm not sure why it stil explodes
#     history = model.fit(X_train, y_train, epochs=32, batch_size=32, verbose=2)
#     y_score = model.predict(X_test)
#
#     fpr = dict()
#     tpr = dict()
#     roc_auc = dict()
#     for i in range(5):
#         fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
#         roc_auc[i] = auc(fpr[i], tpr[i])
#
#     plotROC(fpr, tpr, roc_auc)
