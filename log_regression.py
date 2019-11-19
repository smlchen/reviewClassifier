import numpy as np
import pandas as pd
import time
from sklearn.linear_model import LogisticRegression
from save_grid_search_results import save_grid_search_results
import joblib

def log_regression(X_train, y_train, X_test, y_test, metrics=['accuracy'], class_weight=None, save_model = False, predict_rating=True):
    
    #class weights
    class_weight = None if not class_weight else 'balanced'
    weighted = 'False' if not class_weight else 'True'
    predict = 'rating' if predict_rating else 'category'

    #create model and fit
    model = LogisticRegression(C=1e5, tol=1e-5, solver='liblinear', multi_class = 'ovr',\
                            random_state=3, max_iter=1000, class_weight=class_weight)
    model.fit(X_train,y_train)
    #evaluate the model
    accuracy = model.score(X_test, y_test)
    
    #save model and results
    file_name = ''
    if save_model:
        path = r'./grid_search_results/'
        file_name = str(int(time.time()))
        joblib.dump(model, path + file_name + '.pkl')
    results = save_grid_search_results(dict(file_name=file_name, predict=predict, predictor_type='log_regression', weighted=weighted, accuracy=accuracy))
    
    return results
