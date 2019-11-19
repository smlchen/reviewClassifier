#Import all the dependencies
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import pandas as pd
import json
from sklearn.model_selection import KFold

amazon = pd.read_csv('./cleanAmazon_reviewtextOnly.csv', header=0)
doc2vec = amazon.dropna()            #getting rid of null values in cleantextd
doc2vec = doc2vec.reset_index(drop=True)
data = doc2vec['cleanText'].tolist()
tagged_data = [TaggedDocument(words=word_tokenize(_d),
        tags=[str(i)]) for i, _d in enumerate(data)]
max_epochs = 100
vec_size = 50
alpha = 0.025

model = Doc2Vec(size=vec_size,
                alpha=alpha,
                min_alpha=0.00025,
                min_count=1,
                dm =1)

model.build_vocab(tagged_data)
for epoch in range(max_epochs):
    print('iteration {0}'.format(epoch))
    model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=model.iter)
    # decrease the learning rate
    model.alpha -= 0.0002
    # fix the learning rate, no decay
    model.min_alpha = model.alpha

# to find most similar doc using tags
similar_doc = model.docvecs.most_similar('2')
# print(similar_doc)

#get all vectors
vectors = model.docvecs.doctag_syn0.tolist()
doc2vec['Vector'] = vectors
doc2vec.to_csv(r'doc2vec.csv', index=False)

#randomize and split and data into training and test
doc2vec = doc2vec.sample(frac=1, random_state=100).reset_index(drop=True)               #shuffle
kf = KFold(n_splits=5)
i = 1
for train_index, test_index in kf.split(doc2vec):
    # print("TRAIN:", train_index)
    # print("TEST:", test_index)
    train, test = doc2vec.iloc[train_index], doc2vec.iloc[test_index]
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    # print(train.shape)
    # print(test.shape)
    train.to_csv(r'./split_doc2vec/' + str(i) + r'/training_data.csv', index=False)
    test.to_csv(r'./split_doc2vec/' + str(i) + r'/test_data.csv', index=False)
    i += 1
