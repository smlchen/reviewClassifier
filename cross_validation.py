from os.path import isfile

import numpy as np
from sklearn.metrics import roc_curve
import sklearn.metrics
from sklearn.preprocessing import label_binarize
from scipy import interp
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import load_model
from sklearn.linear_model import LogisticRegression
from keras.models import Sequential
import joblib

#tf-idf or doc2vec
using_tfidf = False

#converts the doc2vec x string vector into a numpy array
def listify(d_frame): 
    clean_var = []
    for sample in d_frame: 
        str1 = sample.replace(']','').replace('[','')
        l = str1.replace('"','').split(",")
        clean_var.append(l)

    df = pd.DataFrame(clean_var)
    return df.astype(float).to_numpy()


def getData(fold="1"):
    
    if using_tfidf: #for tf-idf data
        df_train = pd.read_csv("split/" + fold + "/training_data.csv")
        df_test = pd.read_csv("split/" + fold + "/test_data.csv")

        X_train = np.asarray(df_train.iloc[:, :-2])
        y_train = df_train.iloc[:, -1]
        X_test = np.asarray(df_test.iloc[:, :-2])
        y_test = df_test.iloc[:, -1]
        results = pd.read_csv(r'./grid_search_results/' + 'results.csv')
    
    else: #doc2vec data
        training_data = pd.read_csv('./split_doc2vec/' + fold + '/training_data.csv', header=0)
        test_data = pd.read_csv('./split_doc2vec/' + fold + '/training_data.csv', header=0)
        X_train = listify(training_data['Vector'])
        X_test = listify(test_data['Vector'])
        y_train = training_data['overall']
        y_test = test_data['overall'] 
        results = pd.read_csv(r'doc2vec_grid_search_results/' + 'results.csv')
    
    
    y_train = label_binarize(y_train, classes=[1, 2, 3, 4, 5])
    y_test = label_binarize(y_test, classes=[1, 2, 3, 4, 5])
    
    #results = results[results.loss_function == 'mse']
    #results = results[results.loss_function == 'binary_crossentropy']
    results = results[results.predictor_type == 'log_regression']
    
    return X_train, y_train, X_test, y_test, results


def plotROC(fpr, tpr, auc, tpr_upper = None, tpr_lower = None, file_name = None):
    plt.figure()
    for rating in range(5):
        plt.plot(fpr[rating], tpr[rating], label='Rating {0} (area = {1:0.2f})'.format(rating + 1, auc[rating]))
        if tpr_upper:
            plt.fill_between(fpr[rating], tpr_lower[rating], tpr_upper[rating], alpha=0.2)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Doc2Vec Logistic Regression')
    plt.legend(loc="lower right")
    if file_name: plt.savefig(r'./roc_plots/' + file_name + '.png')
    plt.show()


def get_roc_auc(y_test, y_predict):
    fpr = dict(); tpr = dict(); roc_auc = dict()
    for rating in range(5):
        fpr[rating], tpr[rating], _ = roc_curve(y_test[:, rating], y_predict[:, rating])
        roc_auc[rating] = sklearn.metrics.auc(fpr[rating], tpr[rating])
    return fpr, tpr, roc_auc


def getROCForAllFolds(best_model):
    #referenced from https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html
    
    if best_model[-1] == '5': #saved keras models file name end in .h5
        model = load_model(best_model)
    else: #saved logistic regression model
        model = joblib.load(best_model)
    
    #separate list for each rating to easily find mean and std
    tpr1_list = []; tpr2_list = []; tpr3_list = []; tpr4_list = []; tpr5_list = [];
    auc_list = []
    mean_fpr = np.linspace(0,1,100) #x-axis for ROC curve and for interpolation
    
    # five fold cross validation
    for fold in range(1, 6):
        X_train, y_train, X_test, y_test, results = getData(str(fold))
        
        if best_model[-1] == '5': #saved keras models file name end in .h5
            model = Sequential.from_config(model.get_config()) #reload model using same config as best_model for doing cross validation
            model.compile(optimizer='adam', loss='mse')
            model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=2)
            y_predict = model.predict(X_test)
        else: #saved logistic regression model
            model = LogisticRegression(C=1e5, tol=1e-5, solver='liblinear', multi_class = 'ovr', max_iter=1000)
            y_train = np.argmax(y_train, axis=1)
            model.fit(X_train,y_train)
            y_predict = model.predict(X_test)
            y_predict = label_binarize(y_predict, classes=[1, 2, 3, 4, 5])
            
        fpr, tpr, auc = get_roc_auc(y_test, y_predict)
        auc_list.append(auc)
        
        tpr1_list.append(interp(mean_fpr, fpr[0], tpr[0])); tpr1_list[-1][0] = 0.0;
        tpr2_list.append(interp(mean_fpr, fpr[1], tpr[1])); tpr2_list[-1][0] = 0.0;
        tpr3_list.append(interp(mean_fpr, fpr[2], tpr[2])); tpr3_list[-1][0] = 0.0;
        tpr4_list.append(interp(mean_fpr, fpr[3], tpr[3])); tpr4_list[-1][0] = 0.0;
        tpr5_list.append(interp(mean_fpr, fpr[4], tpr[4])); tpr5_list[-1][0] = 0.0;
    
    #mean_tpr and std_tpr for each fold
    mean_tpr1 = np.mean(tpr1_list, axis=0); std_tpr1 = np.std(tpr1_list, axis=0); 
    mean_tpr2 = np.mean(tpr2_list, axis=0); std_tpr2 = np.std(tpr2_list, axis=0);
    mean_tpr3 = np.mean(tpr3_list, axis=0); std_tpr3 = np.std(tpr3_list, axis=0);
    mean_tpr4 = np.mean(tpr4_list, axis=0); std_tpr4 = np.std(tpr4_list, axis=0);
    mean_tpr5 = np.mean(tpr5_list, axis=0); std_tpr5 = np.std(tpr5_list, axis=0);
    #mean_tpr and std_tpr to dict
    mean_tpr = dict(); std_tpr = dict()
    mean_tpr[0] = mean_tpr1; mean_tpr[1] = mean_tpr2; mean_tpr[2] = mean_tpr3; mean_tpr[3] = mean_tpr4; mean_tpr[4] = mean_tpr5
    std_tpr[0] = std_tpr1; std_tpr[1] = std_tpr2; std_tpr[2] = std_tpr3; std_tpr[3] = std_tpr4; std_tpr[4] = std_tpr5
    
    #mean auc and upper and lower error lines
    mean_auc = dict(); mean_fpr = dict(); tpr_upper = dict(); tpr_lower = dict()
    for i in range(5):
        mean_fpr[i] = np.linspace(0,1,100)
        tpr_upper[i] = np.minimum(mean_tpr[i]+2*std_tpr[i], 1)
        tpr_lower[i] = np.maximum(mean_tpr[i]-2*std_tpr[i], 0)
        mean_auc[i] = sklearn.metrics.auc(mean_fpr[i], mean_tpr[i])
        
    plotROC(mean_fpr, mean_tpr, mean_auc, tpr_upper, tpr_lower, best_model)


def cross_validate_tfidf():
    """ Referenced implementation: https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html """
    _, _, X_test, y_test, results = getData(fold="1")

    best_model = findBestModel(X_test, y_test, results)
    #best_model = "grid_search_results/1574183552.h5"

    if best_model[-1] == '5': #saved keras models file name end in .h5
        model = load_model(best_model)
        y_predict = model.predict(X_test)
    else: #saved logistic regression model
        model = joblib.load(best_model)
        y_predict = model.predict(X_test)
        y_predict = label_binarize(y_predict, classes=[1, 2, 3, 4, 5])
    
    #fpr, tpr, auc = get_roc_auc(y_test, y_predict)
    #plotROC(fpr, tpr, auc, file_name=best_model)
    
    getROCForAllFolds(best_model)


def findBestModel(X_test, y_test, results):
    # Find best model to work with against the 1st fold of data.
    best_model = ""
    most_AUC = 0
    
    if using_tfidf:
        path = r'./grid_search_results/'
    else: #doc2vec path
        path = r'./doc2vec_grid_search_results/'
    
    best_model_idx = 0
    for i, row in results.iterrows():
        file_name = path + row['file_name']
        if not isfile(file_name):
            continue
        
        if file_name[-1] == '5': # load saved keras models; file name end in .h5
            model = load_model(file_name)
            y_predict = model.predict(X_test)
        else: # load saved logistic regression model
            model = joblib.load(file_name)
            y_predict = model.predict(X_test)
            y_predict = label_binarize(y_predict, classes=[1, 2, 3, 4, 5])
            
        
        fpr, tpr, auc = get_roc_auc(y_test, y_predict)
        
        if most_AUC < sum(auc.values()):
            most_AUC = sum(auc.values())
            best_model = file_name
            best_model_idx = i
    
    print(best_model)
    return best_model


if __name__ == '__main__':
    cross_validate_tfidf()
