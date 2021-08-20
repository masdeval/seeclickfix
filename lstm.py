
from  preprocessing import *
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
from statsmodels.graphics.gofplots import ProbPlot
import tensorflow as tf
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Embedding, Bidirectional, SpatialDropout1D, CuDNNLSTM, concatenate, \
    GlobalAveragePooling1D, GlobalMaxPooling1D, Dense, Dropout, LSTM

from SeeClickFix.preprocessing import onehot_train, embedding_matrix, description_train, summary_train, y_train, train, MAX_LEN

BATCH_SIZE = 64
LSTM_UNITS = 200
DENSE_HIDDEN_UNITS =  LSTM_UNITS
EPOCHS = 5


def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')


def load_embeddings(path):
    with open(path) as f:
        return dict(get_coefs(*line.strip().split(' ')) for line in f)



def build_model_1(embedding_matrix, one_hot_shape):
    words = Input(shape=(MAX_LEN,))
    x = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(words)
    x = Bidirectional(LSTM(LSTM_UNITS, return_sequences=True), merge_mode='concat')(x)
    x = SpatialDropout1D(rate=0.3)(x)
    #x = Bidirectional(LSTM(LSTM_UNITS, return_sequences=True), merge_mode='ave')(x)
    #x = SpatialDropout1D(rate=0.3)(x)

    #x = GlobalAveragePooling1D()(x) # this layer average each output from the Bidirectional layer

    x = concatenate([
        GlobalMaxPooling1D()(x),
        GlobalAveragePooling1D()(x),
    ])

    summary = Input(shape=(MAX_LEN,))
    x_aux = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(summary)
    x_aux = Bidirectional(LSTM(LSTM_UNITS, return_sequences=True), merge_mode='concat')(x_aux)
    x_aux = SpatialDropout1D(rate=0.3)(x_aux)
    #x_aux = Bidirectional(LSTM(LSTM_UNITS, return_sequences=True), merge_mode='ave')(x_aux)
    #x_aux = SpatialDropout1D(rate=0.3)(x_aux)

    # x_aux = GlobalAveragePooling1D()(x_aux)
    x_aux = concatenate([
        GlobalMaxPooling1D()(x_aux),
        GlobalAveragePooling1D()(x_aux),
    ])

    one_hot = Input(shape=(one_hot_shape,))
    hidden = concatenate([x, x_aux, one_hot])

    hidden = Dense(400, activation='relu')(hidden)
    hidden = Dropout(0.4)(hidden)
    hidden = Dense(400, activation='relu')(hidden)
    hidden = Dropout(0.4)(hidden)
    hidden = Dense(300, activation='relu')(hidden)
    hidden = Dropout(0.4)(hidden)
    hidden = Dense(300, activation='relu')(hidden)
    hidden = Dropout(0.4)(hidden)
    hidden = Dense(100, activation='relu')(hidden)
    result = Dense(1, activation='linear')(hidden)

    model = Model(inputs=[words, summary, one_hot], outputs=[result])
    # adam = keras.optimizers.Adam(lr=0.0001, clipnorm=1.0, clipvalue=0.5)
    model.compile(loss='mse', optimizer='adam')

    return model

def build_model_2(embedding_matrix, one_hot_shape):
    words = Input(shape=(MAX_LEN,))
    x = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(words)
    x = LSTM(LSTM_UNITS, return_sequences=False, dropout=0,recurrent_dropout=0)(x)
    #x = LSTM(LSTM_UNITS, return_sequences=False)(x)

    summary = Input(shape=(MAX_LEN,))
    x_aux = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(summary)
    x_aux = LSTM(LSTM_UNITS, return_sequences=False)(x_aux)

    one_hot = Input(shape=(one_hot_shape,),sparse=True)
    hidden = concatenate([x, x_aux, one_hot])
    hidden = concatenate([x, x_aux])

    hidden = Dense(400, activation='relu')(hidden)
    hidden = Dropout(0.4)(hidden)
    hidden = Dense(300, activation='relu')(hidden)
    hidden = Dropout(0.4)(hidden)
    hidden = Dense(100, activation='relu')(hidden)
    result = Dense(1, activation='linear')(hidden)

    model = Model(inputs=[words, summary, one_hot], outputs=[result])
    # adam = keras.optimizers.Adam(lr=0.0001, clipnorm=1.0, clipvalue=0.5)
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




def main(train, models):

    test_idx = (train.sample(frac=.2)).index
    train_idx = train[~train.index.isin(test_idx)].index

    for model_idx,model in enumerate(models):

        checkpoint_predictions = []

        for global_epoch in range(EPOCHS):

            x_train = ([np.asmatrix(description_train[train_idx]), np.asmatrix(summary_train[train_idx]), onehot_train.toarray()[train_idx]])

            model.fit(
                x_train,
                # [Y_train, y_aux_train],
                y_train[train_idx],
                batch_size=BATCH_SIZE,
                epochs=1,
                verbose=2,
                # callbacks=[
                #   LearningRateScheduler(lambda epoch: 1e-3 * (0.6 ** global_epoch))
                # ]
            )

            pred = model.predict(
                [description_train[test_idx], summary_train[test_idx], onehot_train.toarray()[test_idx]],batch_size=2048).flatten()

            checkpoint_predictions.append(np.mean(np.square(pred - y_train[test_idx])))

            print('MSE  for model {} : {:.3}'.format(model_idx,np.mean(np.square(pred - y_train[test_idx]))))

        print('Overall MSE for model {}: {:.3}'.format(model_idx, np.mean(checkpoint_predictions)))


# This creates a matrix with all the train data. But is useless
# description_train = np.array(description_train).reshape(len(description_train),MAX_LEN)
# summary_train = np.array(summary_train).reshape(len(summary_train),50)
# final_train = scipy.sparse.hstack([scipy.sparse.csr_matrix(description_train),scipy.sparse.csr_matrix(summary_train),onehot_train],format='csr')

models=list()
models.append(build_model_1(embedding_matrix,onehot_train.get_shape()[1]))
models.append(build_model_2(embedding_matrix,onehot_train.get_shape()[1]))

main(train,models)

