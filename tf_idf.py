import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from pprint import pprint

#function for computing a tf-idf matrix given training/test data
def computeTFIDF(df):
    collection = df['cleanText'].astype(str).tolist()
    tfidf = TfidfVectorizer(token_pattern=r'(?u)\b[A-Za-z]+\b', max_features=1500, min_df=20)     #regrex ignores any that contains numbers
    features = tfidf.fit_transform(collection)
    matrix = pd.DataFrame(features.todense(), columns=tfidf.get_feature_names())    #panda dataFrame representation of the matrix
    return matrix

amazon = pd.read_csv('./cleanAmazon_reviewtextOnly.csv', header=0)
tfidf = computeTFIDF(amazon)
tfidf['output1'] = amazon['Category']
tfidf['output2'] = amazon['overall']
tfidf.to_csv(r'./tfidf.csv', index=False)
tfidf = tfidf.sample(frac=1, random_state=100).reset_index(drop=True)               #shuffle

# splittig into training and test data using 5-fold cross validation
kf = KFold(n_splits=5)
i = 1
for train_index, test_index in kf.split(tfidf):
    # print("TRAIN:", train_index)
    # print("TEST:", test_index)
    train, test = tfidf.iloc[train_index], tfidf.iloc[test_index]
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    print(train.shape)
    print(test.shape)
    train.to_csv(r'./split/' + str(i) + r'/training_data.csv', index=False)
    test.to_csv(r'./split/' + str(i) + r'/test_data.csv', index=False)
    i += 1
