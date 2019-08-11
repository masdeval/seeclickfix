import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import dill
import gensim
import pickle
import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, Dropout, add, concatenate
from keras.layers import Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D, CuDNNLSTM, TimeDistributed, LSTM
from keras.preprocessing import text, sequence
from keras.callbacks import LearningRateScheduler
from sklearn.metrics import mean_squared_error
import sklearn
import gensim
import keras
from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors
import gc
import pickle
import gensim.downloader as api
import random
from sklearn.externals import joblib
from collections import defaultdict
import dill
import copy
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from nltk.tokenize import TweetTokenizer
from sklearn.model_selection import cross_validate
from math import sqrt


def rmsle(h, y):
    """
   Compute the Root Mean Squared Log Error for hypthesis h and targets y

   Args:
       h - numpy array containing predictions with shape (n_samples, n_targets)
       y - numpy array containing targets with shape (n_samples, n_targets)
   """
    return np.sqrt(np.square(np.log(h + 1) - np.log(y + 1)).mean())


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

train = pd.read_csv('train.csv', sep=',')
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in


EMBEDDING_FILES = [
    # '../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec',
    '../input/glove840b300dtxt/glove.840B.300d.txt'
]

NUM_MODELS = 1
BATCH_SIZE = 128
LSTM_UNITS = 200
DENSE_HIDDEN_UNITS = 2 * LSTM_UNITS
EPOCHS = 10
MAX_LEN = 100


def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')


def load_embeddings(path):
    with open(path) as f:
        return dict(get_coefs(*line.strip().split(' ')) for line in f)


def build_matrix(word_index, embedding, size):
    # embedding_index = load_embeddings(path)

    embedding_matrix = np.zeros((len(word_index) + 1, size))
    for word, i in word_index.items():
        try:
            embedding_matrix[i] = embedding[word]
        except KeyError:
            pass
    return embedding_matrix


def build_model(embedding_matrix, source_shape, location_shape):
    words = Input(shape=(MAX_LEN,))
    x = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(words)
    x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True), merge_mode='ave')(x)
    x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True), merge_mode='ave')(x)
    # x = SpatialDropout1D(rate=0.3)(x)

    hidden = GlobalAveragePooling1D()(x)  # this layer average each output from the Bidirectional layer

    summary = Input(shape=(20,))
    x_aux = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(summary)
    x_aux = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True), merge_mode='ave')(x_aux)
    x_aux = GlobalAveragePooling1D()(x_aux)

    source = Input(shape=(source_shape,))
    location = Input(shape=(location_shape,))
    hidden = concatenate([hidden, x_aux, source, location])
    # hidden = concatenate([
    #    GlobalMaxPooling1D()(x),
    #    GlobalAveragePooling1D()(x),
    # ])
    hidden = Dense(300, activation='relu')(hidden)
    hidden = Dropout(0.3)(hidden)
    # hidden = add([hidden,Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])
    # hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])
    hidden = Dense(400, activation='relu')(hidden)
    hidden = Dropout(0.3)(hidden)
    hidden = Dense(100, activation='relu')(hidden)
    result = Dense(1, activation='linear')(hidden)
    # aux_result = Dense(num_aux_targets, activation='sigmoid')(hidden)

    # model = Model(inputs=words, outputs=[result, aux_result])
    model = Model(inputs=[words, summary, source, location], outputs=[result])
    model.compile(loss='mse', optimizer='adam')

    return model


def preprocess(data):
    '''
    Credit goes to https://www.kaggle.com/gpreda/jigsaw-fast-compact-solution
    '''
    punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~`" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'

    def clean_special_chars(text, punct):
        for p in punct:
            text = text.replace(p, ' ')
        return text

    data = data.astype(str).apply(lambda x: clean_special_chars(x, punct))
    return data


train = pd.read_csv('../input/see-click-predict-fix/train.csv', sep=',')
train['description'].fillna(' ', inplace=True)
# train = train[train['created_time'] > '2013-01-01 00:00:00']
train = train.dropna(subset=['source'])
print(train.info())

train.loc[:, 'description'] = train['description'].apply(lambda x: ' '.join(gensim.utils.simple_preprocess(x)))
train.loc[:, 'summary'] = train['summary'].apply(lambda x: ' '.join(gensim.utils.simple_preprocess(x)))
train.loc[:, 'latitude'] = train['latitude'].apply(lambda x: np.round(x, 3))
train.loc[:, 'longitude'] = train['longitude'].apply(lambda x: np.round(x, 3))

tokenizer = text.Tokenizer()
tokenizer.fit_on_texts(list(train['description']) + list(train['summary']))

test = train.sample(frac=.2, random_state=999)
train = train.iloc[~train.index.isin(test.index), :]

description_train = train['description']
description_test = test['description']
# x_train = preprocess(train['description'])
# x_test = preprocess(test['description'])
description_train = tokenizer.texts_to_sequences(description_train)
description_test = tokenizer.texts_to_sequences(description_test)
description_train = sequence.pad_sequences(description_train, maxlen=MAX_LEN)
description_test = sequence.pad_sequences(description_test, maxlen=MAX_LEN)

summary_train = train['summary']
summary_test = test['summary']
# input_aux_train = preprocess(input_aux_train)
# input_aux_test = preprocess(input_aux_test)
summary_train = tokenizer.texts_to_sequences(summary_train)
summary_test = tokenizer.texts_to_sequences(summary_test)
summary_train = sequence.pad_sequences(summary_train, 20)
summary_test = sequence.pad_sequences(summary_test, 20)

from sklearn.preprocessing import OneHotEncoder

dummies = OneHotEncoder(handle_unknown='ignore')
source_train = dummies.fit_transform(np.array(train[['source']]).reshape(-1, 1)).todense()
# source = dummies.fit_transform(train[['source','num_votes']]) #this should be wrong but is issuing a smaller error
source_test = dummies.transform(np.array(test[['source']]).reshape(-1, 1)).todense()

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

latitude = train['latitude']
longitude = train['longitude']
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
location_train = scaler.fit_transform(np.stack((latitude, longitude), axis=-1))
# location_train = scipy.sparse.csr_matrix(location_train)
latitude = test['latitude']
longitude = test['longitude']
location_test = scaler.transform(np.stack((latitude, longitude), axis=-1))
# location_test = scipy.sparse.csr_matrix(location_test)


embedding = KeyedVectors.load_word2vec_format("../input/glove2word2vec/glove_w2v.txt", binary=False)
# embedding = KeyedVectors.load_word2vec_format("../input/glove840B300dtxt/glove.840B.300d.txt",binary=False)
EMBEDDINGS = [embedding]
embedding_matrix = build_matrix(tokenizer.word_index, embedding, 200)
del embedding

gc.collect()


# embedding_matrix = np.concatenate(
#     [build_matrix(tokenizer.word_index, wordvect,200) for wordvect in EMBEDDINGS], axis=-1)

def main():
    checkpoint_predictions = []
    weights = []

    for model_idx in range(NUM_MODELS):
        model = build_model(embedding_matrix, source_shape=source_train.shape[1],
                            location_shape=location_train.shape[1])
        for global_epoch in range(EPOCHS):
            model.fit(
                [description_train, summary_train, source_train, location_train],
                # [Y_train, y_aux_train],
                train['num_votes'],
                batch_size=BATCH_SIZE,
                epochs=1,
                verbose=2,
                callbacks=[
                    LearningRateScheduler(lambda epoch: 1e-3 * (0.6 ** global_epoch))
                ]
            )
            checkpoint_predictions.append(
                model.predict([description_test, summary_test, source_test, location_test], batch_size=2048)[
                    0].flatten())
            weights.append(2 ** global_epoch)

    model.save('lstm_model_3.h5')
    # pickle.dump(model, open('lstm_model','wb'))

    pred = model.predict([description_test, summary_test, source_test, location_test]).flatten()
    pickle.dump(pred, open('prediction_3.save', 'wb'))


main()
model = keras.models.load_model('lstm_model_3.h5')
pred = pickle.load(open('prediction_3.save', 'rb'))
auc = mean_squared_error(test['num_votes'], pred)
print('Overall Test RMSE %.3f' % sqrt(auc))
print('RMSLE %.3f' % rmsle(pred, test['num_votes'])) #RMSLE 0.145

