import numpy as np
import sklearn
from sklearn.model_selection import cross_validate
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDRegressor
import pandas as pd
import gensim
import scipy
from sklearn.metrics import make_scorer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

def rmsle(y, y0):
    #return np.sqrt(np.mean(np.square(np.log1p(y) - np.log1p(y0))))
    return np.sqrt(np.square(np.log(y + 1) - np.log(y0 + 1)).mean())

rmsle_score = make_scorer(rmsle, greater_is_better=False)


train = pd.read_csv('train.csv', sep=',')
train['description'].fillna(' ',inplace=True)
#train = train[train['created_time'] > '2013-01-01 00:00:00']
train = train.dropna(subset=['source'])
print(train.info())


description = train['description']
summary = train['summary']
description = description.apply(lambda x:  ' '.join(gensim.utils.simple_preprocess(x)))
summary = summary.apply(lambda x:  ' '.join(gensim.utils.simple_preprocess(x)))
bow = CountVectorizer(max_features=5000, binary=True, max_df=0.5)
#bow = bow.fit(description+summary)
description = bow.fit_transform(description)
bow.vocabulary_ = None
summary = bow.fit_transform(summary)

from sklearn.preprocessing import OneHotEncoder
dummies = OneHotEncoder(handle_unknown='ignore')
source = dummies.fit_transform(np.array(train[['source']]).reshape(-1,1))
#source = dummies.fit_transform(train[['source','num_votes']]) #this should be wrong but is issuing a smaller error


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

location = train[['latitude','longitude']]
latitude = location['latitude'].apply(lambda x: np.round(x,3))
longitude = location['longitude'].apply(lambda x: np.round(x,3))
scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
location = scaler.fit_transform(np.stack((latitude,longitude),axis=-1))
location = scipy.sparse.csr_matrix(location)


#SGD L1 learning_rate='invscaling'
#text_fields = scipy.sparse.hstack([description]) # Overall MSE: -0.207462
#text_fields = scipy.sparse.hstack([description,summary]) # Overall MSE: -0.201481
#text_fields = scipy.sparse.hstack([description,summary,source]) # Overall MSE: -0.177597
#text_fields = scipy.sparse.hstack([description,summary,source,location]) #Overall MSE: -0.180237

#SGD L2 learning_rate='optimal':
#text_fields = scipy.sparse.hstack([description]) # Overall MSE: -8.615321
#text_fields = scipy.sparse.hstack([description,summary]) # Overall MSE: -16.659406
#text_fields = scipy.sparse.hstack([description,summary,source]) # Overall MSE: -15.788960
#text_fields = scipy.sparse.hstack([description,summary,source,location]) #Overall MSE: -17.936516


#SGD L2 learning_rate='invscaling':
#text_fields = scipy.sparse.hstack([description]) # Overall MSE: -0.197950
#text_fields = scipy.sparse.hstack([description,summary]) # Overall MSE:  -0.196244
#text_fields = scipy.sparse.hstack([description,summary,source]) # Overall MSE: -0.176347
#text_fields = scipy.sparse.hstack([description,summary,source,location]) #Overall MSE: -0.176827


#RidgeCV
#text_fields = scipy.sparse.hstack([description,summary,source,location]) #Overall MSE: -0.179918

#SVM (kernel=linear, C=0.4)
#text_fields = scipy.sparse.hstack([description,summary,source,location]) #-0.179531


#text_fields = scipy.sparse.hstack([description,summary,source,location])

#y_train = train['num_votes'].apply(lambda x: np.log1p(x+1))
y_train = train['num_votes']

regressor = SGDRegressor(loss='squared_loss', penalty='l2', alpha=0.001, l1_ratio=0.15, fit_intercept=True,
                        max_iter=1000, tol=0.001, shuffle=True, verbose=0, epsilon=0.1, random_state=999,
                        learning_rate='invscaling', eta0=0.01, power_t=0.25, early_stopping=False,
                        validation_fraction=0.1, n_iter_no_change=5, warm_start=False, average=False)

# regressor = sklearn.linear_model.ElasticNetCV(l1_ratio=0.5, eps=0.001, n_alphas=100, alphas=None, fit_intercept=True,
#                                 normalize=False, precompute='auto', max_iter=1000, tol=0.0001, cv=5, copy_X=True,
#                                 verbose=0, n_jobs=1, positive=False, random_state=666, selection='cyclic')

#regressor = sklearn.linear_model.RidgeCV(alphas=(0.1, 1.0, 10.0), fit_intercept=True, normalize=False, scoring=None, cv=5, gcv_mode=None, store_cv_values=False)

# regressor = sklearn.linear_model.LassoCV(eps=0.001, n_alphas=100, alphas=None, fit_intercept=True, normalize=False,
#                                          precompute='auto', max_iter=1000, tol=0.0001, copy_X=True, cv='warn',
#                                          verbose=False, n_jobs=None, positive=False, random_state=None,
#                                          selection='cyclic')

# regressor = GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=100, subsample=1.0,
#                 criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
#                 max_depth=3, min_impurity_decrease=0.0, min_impurity_split=None, init=None, random_state=999,
#                 max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto',
#                 validation_fraction=0.1, n_iter_no_change=None, tol=0.0001)


# regressor = sklearn.svm.SVR(kernel='poly', degree=2, gamma='scale', coef0=0.0, tol=0.001, C=0.5,
#              epsilon=0.1, shrinking=True, cache_size=200, verbose=False, max_iter=-1)

# regressor = sklearn.svm.LinearSVR(epsilon=0.0, tol=0.0001, C=0.4, loss='epsilon_insensitive',
#                                  fit_intercept=True, intercept_scaling=1.0, dual=True, verbose=0,
#                                  random_state=999, max_iter=1000)

result = cross_validate(regressor,X=text_fields,y=y_train,cv=5,scoring=rmsle_score, return_train_score=False)

#result = cross_validate(regressor,X=text_fields,y=y_train,cv=5,scoring='neg_mean_squared_error', return_train_score=False)

from prettytable import PrettyTable
print("\n BOW features")
x = PrettyTable()
x.field_names = [" ","Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5"]
x.add_row(["MSE: "] + [str(v) for v in result['test_score']])
print(x)
print("Overall MSE: %f" % np.mean(result['test_score']))
