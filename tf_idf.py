import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from pprint import pprint

amazon = pd.read_csv('./cleanAmazon_5_features.csv', header=0)      #reading in data
amazon = amazon.sample(frac=1, random_state=100).reset_index(drop=True)               #shuffle
# print(amazon)

# splittig into training and test data using 5-fold cross validation
kf = KFold(n_splits=5)
i = 1
for train_index, test_index in kf.split(amazon):
    # print("TRAIN:", train_index)
    # print("TEST:", test_index)
    train, test = amazon.iloc[train_index], amazon.iloc[test_index]
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    train.to_csv(r'./split/' + str(i) + r'/training_data.csv')
    test.to_csv(r'./split/' + str(i) + r'/test_data.csv')
    i += 1

#function for computing a tf-idf matrix given training/test data
def computeTFIDF(df):
    collection = df['cleanText'].astype(str).tolist()
    tfidf = TfidfVectorizer(token_pattern=r'(?u)\b[A-Za-z]+\b')     #regrex ignores any that contains numbers
    features = tfidf.fit_transform(collection)
    matrix = pd.DataFrame(features.todense(), columns=tfidf.get_feature_names())    #panda dataFrame representation of the matrix
    matrix.to_csv(r'./tfidf.csv')       #takes about 30 seconds to write 3000ish samples
    return matrix

#Here's an example
example = pd.read_csv('./split/1/training_data.csv', header=0)
computeTFIDF(example)
