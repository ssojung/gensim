import logging
logging.root.handlers = []  # Jupyter messes up logging so needs a reset
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from smart_open import smart_open
import pandas as pd
import numpy as np
from numpy import random
import gensim
import nltk
from sklearn.cross_validation import train_test_split
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

"""
load data set
"""

def tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            if len(word) < 2:
                continue
            tokens.append(word.lower())
    return tokens



wiki_path = '/home/sojung/workspace/c2v/data/news_level1.csv'
#original_path = 'data/tagged_plots_movielens.csv'
df = pd.read_csv(wiki_path)
df = df.dropna()
df['plot'].apply(lambda x: len(x.split(' '))).sum()

train_data, test_data = train_test_split(df, test_size=0.1, random_state=42)

train_tagged = train_data.apply(
    lambda r: TaggedDocument(words=tokenize_text(r['plot']), tags=[r.tag]), axis=1)
test_tagged = test_data.apply(
    lambda r: TaggedDocument(words=tokenize_text(r['plot']), tags=[r.tag]), axis=1)

trainsent = train_tagged.values

doc2vec_model = Doc2Vec(trainsent, workers=4, dm=1, iter=20, size=400)