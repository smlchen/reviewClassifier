import numpy as np
import sklearn.metrics
from sklearn.preprocessing import label_binarize
import pandas as pd
from keras.models import load_model
from keras.models import Sequential


#tf-idf or doc2vec
using_tfidf = True

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

    
    return X_train, y_train, X_test, y_test, results
 

def get_accuracy(model_file):
    
    model = load_model(model_file)
    
    accuracy = []
    # five fold cross validation
    for fold in range(1, 6):
        X_train, y_train, X_test, y_test, results = getData(str(fold))
        
        #reload model using same config as best_model for doing cross validation
        model = Sequential.from_config(model.get_config()) 
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=2)
        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        accuracy.append(acc)
        
    print('accuracy per fold =', accuracy)
    print('mean =', np.mean(accuracy))
    print('std =', np.std(accuracy)) 
        
if __name__ == '__main__':
    model = r'./best_tfidf_model.h5'
    get_accuracy(model)
    
    





