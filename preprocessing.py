import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDRegressor
import pandas as pd
import gensim
import scipy
from sklearn.metrics import make_scorer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import sys
import gc
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt


def build_matrix(word_index, embedding, size):
    # embedding_index = load_embeddings(path)

    embedding_matrix = np.zeros((len(word_index) + 1, size))
    for word, i in word_index.items():
        try:
            embedding_matrix[i] = embedding[word]
        except KeyError:
            pass
    return embedding_matrix

MAX_LEN = 40


# train total
train = pd.read_csv('train.csv', sep=',')
train = train[(train['num_votes'] < 50)]
train['description'].fillna('this is a test', inplace=True)
# train = train[train['created_time'] > '2013-01-01 00:00:00']
# train = train.dropna(subset=['source'])

#train = train.sample(frac=.2) #?????? just to speed up things ??

train.reset_index(inplace=True)
print(train.info())

# train reduced
# train = train[(train['num_votes'] > 1) & (train['num_votes'] < 50)]

# train baseline
#train_baseline = train[~train['tag_type'].isna()]
# train = train_baseline
# print(train.info())

train.loc[:, 'description'] = train['description'].apply(lambda x: ' '.join(gensim.utils.simple_preprocess(x)))
train.loc[:, 'summary'] = train['summary'].apply(lambda x: ' '.join(gensim.utils.simple_preprocess(x)))
train.loc[:, 'latitude'] = train['latitude'].apply(lambda x: np.round(x, 3))
train.loc[:, 'longitude'] = train['longitude'].apply(lambda x: np.round(x, 3))

train['hour'] = [str(pd.to_datetime(x).hour) for x in train['created_time']]
train['dayofweek'] = [str(pd.to_datetime(x).weekday()) for x in train['created_time']]
train['year'] = [str(pd.to_datetime(x).year) for x in train['created_time']]

import tensorflow as tf

tokenizer=tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(list(train['description']) + list(train['summary']))

# test = train.sample(frac=.2,random_state=999)
# train =  train.iloc[~train.index.isin(test.index),:]

# y_test = test['num_votes'].apply(lambda x: np.log1p(x+1))
# y_train = train['num_votes'].apply(lambda x: np.log1p(x+1))
# y_test = test['num_votes']
y_train = train['num_votes']

description_train = train['description']
description_train = tokenizer.texts_to_sequences(description_train)
description_train = tf.keras.preprocessing.sequence.pad_sequences(description_train, maxlen=MAX_LEN)

summary_train = train['summary']
summary_train = tokenizer.texts_to_sequences(summary_train)
summary_train = tf.keras.preprocessing.sequence.pad_sequences(summary_train, maxlen=MAX_LEN)

from sklearn.preprocessing import OneHotEncoder

dummies = OneHotEncoder(handle_unknown='ignore')

# source_train = dummies.fit_transform(np.array(train[['source']]).reshape(-1,1)).todense()
# source = dummies.fit_transform(train[['source','num_votes']]) #this should be wrong but is issuing a smaller error
# source_test = dummies.transform(np.array(test[['source']]).reshape(-1,1)).todense()

# Decimal places      Object that can be unambiguously recognized at this scale
# 0	                  country or large region
# 1	            	  large city or district
# 2	             	  town or village
# 3               	  neighborhood, street
# 4                   individual street, land parcel
# 5                   individual trees, door entrance
# 6                   individual humans
# 7                   practical limit of commercial surveying
# 8                   specialized surveying (e.g. tectonic plate mapping)

latitude_train = train['latitude']
longitude_train = train['longitude']
# latitude_test = test['latitude']
# longitude_test = test['longitude']

onehot_train = np.stack((train['hour'], train['dayofweek'], train['year'], latitude_train, longitude_train), axis=-1)
onehot_train = dummies.fit_transform(onehot_train)
# onehot_train = scipy.sparse.csr_matrix(onehot_train)



# onehot_test = np.stack((test['hour'],test['dayofweek'],test['year'],latitude_test,longitude_test),axis=-1)
# onehot_test = dummies.transform(onehot_test)
# onehot_test = scipy.sparse.csr_matrix(onehot_test)

from gensim.models import KeyedVectors
embedding_matrix=[]
try:
    embedding_matrix = np.load("embedding_matrix.npy")
except IOError:
    embedding = KeyedVectors.load_word2vec_format("./glove.twitter.27B/word2vec200d.txt", binary=False)
    # embedding = KeyedVectors.load_word2vec_format("../input/glove2word2vec/glove_w2v.txt", binary=False)
    # embedding = KeyedVectors.load_word2vec_format("../input/glove840B300dtxt/glove.840B.300d.txt",binary=False)
    embedding_matrix = build_matrix(tokenizer.word_index, embedding, 200)
    del embedding
    gc.collect()
    np.save("embedding_matrix",embedding_matrix)




# embedding_matrix = np.concatenate(
#     [build_matrix(tokenizer.word_index, wordvect,200) for wordvect in EMBEDDINGS], axis=-1)

# This creates a matrix with all the train data. But is useless
# description_train = np.array(description_train).reshape(len(description_train),MAX_LEN)
# summary_train = np.array(summary_train).reshape(len(summary_train),50)
# final_train = scipy.sparse.hstack([scipy.sparse.csr_matrix(description_train),scipy.sparse.csr_matrix(summary_train),onehot_train],format='csr')

