#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Import packages.
import numpy as np
import pandas as pd
import nltk
import nltk.corpus
import gzip
import json
import re
from nltk.corpus import wordnet


# #### Build function to clean text with test data.

# In[4]:


def parse(path):
  g = gzip.open(path, 'r')
  for l in g:
    yield json.loads(l)
    
    
def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

df = getDF("../../../Downloads/AMAZON_FASHION_5.json.gz")


# In[5]:


df.head()


# In[6]:


np.shape(df)


# In[7]:


# Drop duplicate reviews
df_nodup = df.drop_duplicates(subset = ['reviewText'])


# In[8]:


def tokenize_text(doc):
    """
    Input: A string of words.
    Output: List of tokenized words that are all lowercase.
    """

    # Tokenize and make lowercase.
    words = nltk.word_tokenize(doc)
    words = [w.lower() for w in words]
    
    return words


def wordnet_pos(tag):
    """
    Map a Brown POS tag to a WordNet POS tag. This is for lemmatization.
    """
    
    table = {"N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV, "J": wordnet.ADJ}
    
    # Default to a noun.
    return table.get(tag[0], wordnet.NOUN)


def lemmatize_text(words):
    """
    Input: A list of tokenized words.
    Output: A list of tokenized words that are lemmatized.
    """
    
    lemmatizer = nltk.WordNetLemmatizer()
    word_tags = nltk.pos_tag(words)
    words = [lemmatizer.lemmatize(w, wordnet_pos(t)) for (w, t) in word_tags]
    
    return words


def remove_stopwords(words):
    """
    Input: A list of tokenized words.
    Output: A list of tokenized words that have stopwords removed.
    """
    
    stopwords = nltk.corpus.stopwords.words("english")
    words = [w for w in words if w not in stopwords]
    
    return words

def clean_text(doc): 
    """
    Input: A string of words.
    Output: A string of words that has been lemmatized, has the stopwords removed, and has the puncuation removed.
    """
    
    words = re.sub("< ?/?[a-z]+ ?>|\n", "", doc)
    words = tokenize_text(words)
    words = lemmatize_text(words)
    words = remove_stopwords(words)
    doc = [w for w in words if w.isalnum()]
    doc = ' '.join(doc)
    
    return doc

def clean_df(df):
    """
    Input: A dataframe with a column of reviews called 'reviewText'.
    Output: The same dataframe as the input, but with an extra column called 'text' which has the 
            cleaned 'reviewText'.
    """
    
    text = df['reviewText']
    df_clean = df.copy()
    df_clean['text'] = [clean_text(str(i)) for i in text]

    return df_clean


# In[9]:


print(df['reviewText'][10])
print(clean_text(df['reviewText'][10]))


# In[10]:


print(df['reviewText'][300])
print(clean_text(df['reviewText'][300]))


# In[11]:


# Test the function
clean_df(df_nodup)[['reviewText', 'text']].head()


# In[12]:


# from sklearn.feature_extraction.text import TfidfVectorizer
# vectorizer = TfidfVectorizer(strip_accents='unicode', analyzer='word', ngram_range=(1,3), norm='l2')

# tmp = df['text']

# vectorizer.fit(tmp)

# tmp_tfidf = vectorizer.transform(tmp)


# #### Clean the amazon data.

# In[13]:


# Clean the amazon text.
amazon = pd.read_csv('~/Downloads/GroupProject/AMAZON.csv')


# In[15]:


amazon.head()


# In[21]:


clean_data = clean_df(amazon)


# In[23]:


clean_data.to_csv(r'~/Documents/ECS171/reviewClassifier/clean_data.csv')


# In[24]:


clean_data


# In[ ]:




