# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in


EMBEDDING_FILES = [
    # '../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec',
    '../input/glove840b300dtxt/glove.840B.300d.txt'
]

NUM_MODELS = 1
BATCH_SIZE = 64
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


def build_model(embedding_matrix, one_hot_shape):
    words = Input(shape=(MAX_LEN,))
    x = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(words)
    x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True), merge_mode='concat')(x)
    x = SpatialDropout1D(rate=0.3)(x)
    x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True), merge_mode='ave')(x)
    x = SpatialDropout1D(rate=0.3)(x)

    # x = GlobalAveragePooling1D()(x) # this layer average each output from the Bidirectional layer

    x = concatenate([
        GlobalMaxPooling1D()(x),
        GlobalAveragePooling1D()(x),
    ])

    summary = Input(shape=(50,))
    x_aux = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(summary)
    x_aux = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True), merge_mode='concat')(x_aux)
    x_aux = SpatialDropout1D(rate=0.3)(x_aux)
    x_aux = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True), merge_mode='ave')(x_aux)
    x_aux = SpatialDropout1D(rate=0.3)(x_aux)

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


# train total
train = pd.read_csv('../input/see-click-predict-fix/train.csv', sep=',')
train = train[(train['num_votes'] < 50)]
train['description'].fillna('this is a test', inplace=True)
# train = train[train['created_time'] > '2013-01-01 00:00:00']
# train = train.dropna(subset=['source'])
train.reset_index(inplace=True)
print(train.info())

# train reduced
# train = train[(train['num_votes'] > 1) & (train['num_votes'] < 50)]

# train baseline
train_baseline = train[~train['tag_type'].isna()]
# train = train_baseline
# print(train.info())

train.loc[:, 'description'] = train['description'].apply(lambda x: ' '.join(gensim.utils.simple_preprocess(x)))
train.loc[:, 'summary'] = train['summary'].apply(lambda x: ' '.join(gensim.utils.simple_preprocess(x)))
train.loc[:, 'latitude'] = train['latitude'].apply(lambda x: np.round(x, 3))
train.loc[:, 'longitude'] = train['longitude'].apply(lambda x: np.round(x, 3))

train['hour'] = [str(pd.to_datetime(x).hour) for x in train['created_time']]
train['dayofweek'] = [str(pd.to_datetime(x).weekday()) for x in train['created_time']]
train['year'] = [str(pd.to_datetime(x).year) for x in train['created_time']]

tokenizer = text.Tokenizer()
tokenizer.fit_on_texts(list(train['description']) + list(train['summary']))

# test = train.sample(frac=.2,random_state=999)
# train =  train.iloc[~train.index.isin(test.index),:]

# y_test = test['num_votes'].apply(lambda x: np.log1p(x+1))
# y_train = train['num_votes'].apply(lambda x: np.log1p(x+1))
# y_test = test['num_votes']
y_train = train['num_votes']

description_train = train['description']
# description_test = test['description']
description_train = tokenizer.texts_to_sequences(description_train)
# description_test = tokenizer.texts_to_sequences(description_test)
description_train = sequence.pad_sequences(description_train, maxlen=MAX_LEN)
# description_test = sequence.pad_sequences(description_test, maxlen=MAX_LEN)

summary_train = train['summary']
# summary_test = test['summary']
summary_train = tokenizer.texts_to_sequences(summary_train)
# summary_test = tokenizer.texts_to_sequences(summary_test)
summary_train = sequence.pad_sequences(summary_train, 50)
# summary_test = sequence.pad_sequences(summary_test,50)

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


embedding = KeyedVectors.load_word2vec_format("../input/glove2word2vec/glove_w2v.txt", binary=False)
# embedding = KeyedVectors.load_word2vec_format("../input/glove840B300dtxt/glove.840B.300d.txt",binary=False)
EMBEDDINGS = [embedding]
embedding_matrix = build_matrix(tokenizer.word_index, embedding, 200)
del embedding
gc.collect()


# embedding_matrix = np.concatenate(
#     [build_matrix(tokenizer.word_index, wordvect,200) for wordvect in EMBEDDINGS], axis=-1)


def main(x_train, y_train, one_hot_shape):
    checkpoint_predictions = []
    weights = []

    model = build_model(embedding_matrix, one_hot_shape)

    for model_idx in range(NUM_MODELS):
        # model = build_model(embedding_matrix,one_hot_shape)
        for global_epoch in range(EPOCHS):
            model.fit(
                x_train,
                # [Y_train, y_aux_train],
                y_train,
                batch_size=BATCH_SIZE,
                epochs=1,
                verbose=2,
                # callbacks=[
                #   LearningRateScheduler(lambda epoch: 1e-3 * (0.6 ** global_epoch))
                # ]
            )
            # checkpoint_predictions.append(model.predict(x_test, batch_size=2048)[0].flatten())
            # weights.append(2 ** global_epoch)

    model.save('lstm_model_3.h5')
    return model


# This creates a matrix with all the train data. But is useless
# description_train = np.array(description_train).reshape(len(description_train),MAX_LEN)
# summary_train = np.array(summary_train).reshape(len(summary_train),50)
# final_train = scipy.sparse.hstack([scipy.sparse.csr_matrix(description_train),scipy.sparse.csr_matrix(summary_train),onehot_train],format='csr')

kfold = KFold(n_splits=2, shuffle=True, random_state=999)
cvscores = []
for train_idx, test_idx in kfold.split(onehot_train):
    ## this is to guarantee full batches of BATCH_SIZE examples
    # length_train = len(train_idx)%BATCH_SIZE
    # length_train = len(train_idx) - length_train
    # train_idx = train_idx[0:length_train]
    # test_idx = test_idx[0:length_train]

    test_idx = (train.sample(frac=.2)).index
    train_idx = train[~train.index.isin(test_idx)].index

    model = main([description_train[train_idx], summary_train[train_idx], onehot_train[train_idx]], y_train[train_idx],
                 onehot_train.get_shape()[1])
    pred = model.predict([description_train[test_idx], summary_train[test_idx], onehot_train[test_idx]]).flatten()
    cvscores.append(np.mean(np.square(pred - y_train[test_idx])))
    print('MSE %.3f' % np.mean(np.square(pred - y_train[test_idx])))

print('Overall MSE: %.3f' % np.mean(cvscores))


