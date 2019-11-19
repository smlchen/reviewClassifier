# Import packages.
import numpy as np
import pandas as pd
import nltk
import nltk.corpus
import gzip
import json
import re
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

# Reduces adjectives and nouns to their stems.
ps = PorterStemmer()

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
    Output: A list of tokenized words that have stopwords and non-English words removed.
    """
    
    stopwords = nltk.corpus.stopwords.words("english")
    words = [w for w in words if w not in stopwords]
    
    # Remove typos and non-English words.
    englishwords = set(nltk.corpus.words.words())
    words = [w for w in words if w in englishwords]
    
    return words

def clean_text(doc): 
    """
    Input: A string of words.
    Output: A string of words that has been lemmatized, has the stopwords removed, and has the puncuation removed.
    """
    
    words = re.sub("< ?/?[a-z]+ ?>|\n", "", doc)
    words = tokenize_text(words)
    words = lemmatize_text(words)
    words = [ps.stem(i) for i in words] # reduce adjectives and nouns to their stems
    words = remove_stopwords(words)
    doc = [w for w in words if w.isalpha()]
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

amazon = pd.read_csv('~/Desktop/rawAmazon.csv')
clean_data = clean_df(amazon)
clean_data.to_csv(r'~/Desktop/cleanAmazon.csv')




