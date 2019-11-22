import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.class_weight import compute_class_weight
from grid_search import grid_search
from log_regression import log_regression
import time
#accuracy and loss are saved into ./grid_search_results/results.csv
#load saved nn models using keras.models.load_model for .h5 files 
#load saved log reg models using joblib.load for .pkl files

hours = 2
time.sleep(hours*30*60)

#tf-idf or doc2vec
using_tfidf = False

#predict category or rating?
predict_rating = True

#save model and history to ./grid_search_results ?
save_model = True

#neural network or logistic regression?
neural_net = True

#weight classes?
weight_class = False

#function that converts a string list into a dataframe
def listify(d_frame): 
    clean_var = []

    for sample in d_frame: 
        str1 = sample.replace(']','').replace('[','')
        l = str1.replace('"','').split(",")
        clean_var.append(l)

    df = pd.DataFrame(clean_var)
    return df.astype(float)


#load data
if using_tfidf:
    training_data = pd.read_csv('./split/1/training_data.csv', header=0)
    test_data = pd.read_csv('./split/1/test_data.csv', header=0)
    X_train = training_data.drop(['output1','output2'], axis=1)
    X_test = test_data.drop(['output1','output2'], axis=1)
    y_train = training_data['output2'] if predict_rating else training_data['output1']
    y_test = test_data['output2'] if predict_rating else test_data['output1']
else: #doc2vec
    training_data = pd.read_csv('./split_doc2vec/1/training_data.csv', header=0)
    test_data = pd.read_csv('./split_doc2vec/1/test_data.csv', header=0)
    X_train = listify(training_data['Vector'])
    X_test = listify(test_data['Vector'])
    y_train = training_data['overall'] if predict_rating else training_data['Category']
    y_test = test_data['overall'] if predict_rating else test_data['Category']
        
    
#one hot encoding of the ratings
encode_label = LabelBinarizer()
encode_label.fit(y_train)


#calc class weights
class_weight = None
if weight_class:
    y_int = np.argmax(encode_label.transform(y_train), axis=1)
    class_weight = compute_class_weight('balanced', np.unique(y_int), y_int)
    class_weight = dict(enumerate(class_weight))


#grid search params to test
num_hidden_layers = [1,2,3]
num_nodes = [2,4,8,16,32]
epochs = [50]
batch_size = [32]

callbacks = None 
metrics = ['accuracy']

if neural_net:
    grid_search(X_train, encode_label.transform(y_train), X_test, encode_label.transform(y_test),\
            num_hidden_layers, num_nodes, epochs, batch_size, activation='sigmoid', output_activation='sigmoid',\
                optimizer='adam', loss='mse', metrics=metrics, callbacks=callbacks, class_weight=class_weight,\
                    save_model=save_model, predict_rating=predict_rating, using_tfidf=using_tfidf)

else:
    y_train = np.argmax(encode_label.transform(y_train), axis=1)
    y_test = np.argmax(encode_label.transform(y_test), axis=1)
    log_regression(X_train, y_train, X_test, y_test, class_weight=class_weight, save_model=save_model, predict_rating=predict_rating, using_tfidf=using_tfidf)
