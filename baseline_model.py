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
import sys
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import ProbPlot

plt.style.use('seaborn') # pretty matplotlib plots
plt.rc('font', size=14)
plt.rc('figure', titlesize=18)
plt.rc('axes', labelsize=15)
plt.rc('axes', titlesize=18)

def graph(formula, x_range, label=None):
    """
    Helper function for plotting cook's distance lines
    """
    x = x_range
    y = formula(x)
    plt.plot(x, y, label=label, lw=1, ls='--', color='red')


def diagnostic_plots(X, y, model_fit=None):
  """
  Function to reproduce the 4 base plots of an OLS model in R.

  ---
  Inputs:

  X: A numpy array or pandas dataframe of the features to use in building the linear regression model

  y: A numpy array or pandas series/dataframe of the target variable of the linear regression model

  model_fit [optional]: a statsmodel.api.OLS model after regressing y on X. If not provided, will be
                        generated from X, y
  """

  if not model_fit:
      model_fit = sm.OLS(y, sm.add_constant(X)).fit()

  print(model_fit.summary())
  # create dataframe from X, y for easier plot handling
  #dataframe = pd.concat([X, y], axis=1)

  # model values
  model_fitted_y = model_fit.fittedvalues
  # model residuals
  model_residuals = model_fit.resid
  # normalized residuals
  model_norm_residuals = model_fit.get_influence().resid_studentized_internal
  # absolute squared normalized residuals
  model_norm_residuals_abs_sqrt = np.sqrt(np.abs(model_norm_residuals))
  # absolute residuals
  model_abs_resid = np.abs(model_residuals)
  # leverage, from statsmodels internals
  model_leverage = model_fit.get_influence().hat_matrix_diag
  # cook's distance, from statsmodels internals
  model_cooks = model_fit.get_influence().cooks_distance[0]

  plot_lm_1 = plt.figure()
  plot_lm_1.axes[0] = sns.residplot(model_fitted_y, y, data=None,
                            lowess=True,
                            scatter_kws={'alpha': 0.5},
                            line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})

  plot_lm_1.axes[0].set_title('Residuals vs Fitted')
  plot_lm_1.axes[0].set_xlabel('Fitted values')
  plot_lm_1.axes[0].set_ylabel('Residuals');

  # annotations
  abs_resid = model_abs_resid.sort_values(ascending=False)
  abs_resid_top_3 = abs_resid[:3]
  for i in abs_resid_top_3.index:
      plot_lm_1.axes[0].annotate(i,
                                 xy=(model_fitted_y[i],
                                     model_residuals[i]));

  QQ = ProbPlot(model_norm_residuals)
  plot_lm_2 = QQ.qqplot(line='45', alpha=0.5, color='#4C72B0', lw=1)
  plot_lm_2.axes[0].set_title('Normal Q-Q')
  plot_lm_2.axes[0].set_xlabel('Theoretical Quantiles')
  plot_lm_2.axes[0].set_ylabel('Standardized Residuals');
  # annotations
  abs_norm_resid = np.flip(np.argsort(np.abs(model_norm_residuals)), 0)
  abs_norm_resid_top_3 = abs_norm_resid[:3]
  for r, i in enumerate(abs_norm_resid_top_3):
      plot_lm_2.axes[0].annotate(i,
                                 xy=(np.flip(QQ.theoretical_quantiles, 0)[r],
                                     model_norm_residuals[i]));

  plot_lm_3 = plt.figure()
  plt.scatter(model_fitted_y, model_norm_residuals_abs_sqrt, alpha=0.5);
  sns.regplot(model_fitted_y, model_norm_residuals_abs_sqrt,
              scatter=False,
              ci=False,
              lowess=True,
              line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8});
  plot_lm_3.axes[0].set_title('Scale-Location')
  plot_lm_3.axes[0].set_xlabel('Fitted values')
  plot_lm_3.axes[0].set_ylabel('$\sqrt{|Standardized Residuals|}$');

  # annotations
  abs_sq_norm_resid = np.flip(np.argsort(model_norm_residuals_abs_sqrt), 0)
  abs_sq_norm_resid_top_3 = abs_sq_norm_resid[:3]
  for i in abs_norm_resid_top_3:
      try:
       plot_lm_3.axes[0].annotate(i,
                                 xy=(model_fitted_y[i],
                                     model_norm_residuals_abs_sqrt[i]));
      except:
          pass

  plot_lm_4 = plt.figure();
  plt.scatter(model_leverage, model_norm_residuals, alpha=0.5);
  sns.regplot(model_leverage, model_norm_residuals,
              scatter=False,
              ci=False,
              lowess=True,
              line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8});
  plot_lm_4.axes[0].set_xlim(0, max(model_leverage)+0.01)
  plot_lm_4.axes[0].set_ylim(-3, 5)
  plot_lm_4.axes[0].set_title('Residuals vs Leverage')
  plot_lm_4.axes[0].set_xlabel('Leverage')
  plot_lm_4.axes[0].set_ylabel('Standardized Residuals');

  # annotations
  leverage_top_3 = np.flip(np.argsort(model_cooks), 0)[:3]
  for i in leverage_top_3:
      plot_lm_4.axes[0].annotate(i,
                                 xy=(model_leverage[i],
                                     model_norm_residuals[i]));

  p = len(model_fit.params) # number of model parameters
  graph(lambda x: np.sqrt((0.5 * p * (1 - x)) / x),
        np.linspace(0.001, max(model_leverage), 50),
        'Cook\'s distance') # 0.5 line
  graph(lambda x: np.sqrt((1 * p * (1 - x)) / x),
        np.linspace(0.001, max(model_leverage), 50)) # 1 line
  plot_lm_4.legend(loc='upper right');







def rmsle(y, y0):
    y = np.array(y)
    y0 = np.array(y0)
    #return np.sqrt(np.mean(np.square(np.log1p(y) - np.log1p(y0))))
    return np.sqrt(np.square(np.log1p(y) - np.log1p(y0)).mean())

# This version is to be used with the log transformed target
def rmsle_v2(y, y0):
    #return np.sqrt(np.mean(np.square(np.log1p(y) - np.log1p(y0))))
    return np.sqrt(np.square(y  - y0 ).mean())


rmsle_score = make_scorer(rmsle_v2, greater_is_better=False)


train = pd.read_csv('train.csv', sep=',')
train['description'].fillna(' ',inplace=True)
#train = train[train['created_time'] > '2013-01-01 00:00:00']
train = train.dropna(subset=['source'])
print(train.info())

#train['location'] = [[x1,x2] for x1,x2 in zip(train['latitude'].apply(lambda x:round(x,3)),train['longitude'].apply(lambda x:round(x,3)))]

####################################  Baseline ########################################
# Strategy: for the 24% of the data that has issue_type populated, predict the average num_votes by neighborhood

train_baseline = train[~train['tag_type'].isna()]
train_baseline['lat_neighbor'] = train_baseline['latitude'].apply(lambda x: round(x,3))
train_baseline['lon_neighbor'] = train_baseline['longitude'].apply(lambda x: round(x,3))
votes_avg_issue_type = train_baseline[['tag_type','lat_neighbor','lon_neighbor','num_votes']].groupby(['tag_type','lat_neighbor','lon_neighbor']).mean()
#print(votes_avg_issue_type)

pred = list()

# for index,sample in train_baseline.iterrows():
#      pred.append(votes_avg_issue_type.loc[(sample['tag_type'],sample['lat_neighbor'],sample['lon_neighbor']),['num_votes']][0])

# print("\n Baseline RMSLE: %f " % rmsle(pred, train_baseline['num_votes']))
# print("\n")

########################################################################################

#train = train_baseline

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
location = np.stack((latitude,longitude),axis=-1)

#Location not rounded
#location = train[['latitude','longitude']]

#Location not scaled
#location = np.stack((latitude,longitude),axis=-1)

location = scipy.sparse.csr_matrix(location)


#SGD L1 learning_rate='invscaling' Lasso
#text_fields = scipy.sparse.hstack([description]) # Overall MSE: -0.207462
#text_fields = scipy.sparse.hstack([description,summary]) # Overall MSE: -0.201481
#text_fields = scipy.sparse.hstack([description,summary,source]) # Overall MSE: -0.177597
#text_fields = scipy.sparse.hstack([description,summary,source,location]) #Overall MSE: -0.180237


#SGD L2 learning_rate='optimal': Ridge
#text_fields = scipy.sparse.hstack([description]) # Overall MSE: -8.615321
#text_fields = scipy.sparse.hstack([description,summary]) # Overall MSE: -16.659406
#text_fields = scipy.sparse.hstack([description,summary,source]) # Overall MSE: -15.788960
#text_fields = scipy.sparse.hstack([description,summary,source,location]) #Overall MSE: -17.936516

#SGD L2 learning_rate='invscaling': Ridge
#text_fields = scipy.sparse.hstack([description]) # Overall MSE: -0.197950
#text_fields = scipy.sparse.hstack([description,summary]) # Overall MSE:  -0.196244
#text_fields = scipy.sparse.hstack([description,summary,source]) # Overall MSE: -0.176347
#text_fields = scipy.sparse.hstack([description,summary,source,location]) #Overall MSE: -0.176827


#SVM (kernel=linear, C=0.001)
#text_fields = scipy.sparse.hstack([description,summary,source,location],format='csr')
#-0.179531 - C=0.4
#-0.166674 - C=0.001
#text_fields = scipy.sparse.hstack([description,summary])
#-0.188104 - C=0.4
#-0.183882 - C=0.2
#-0.181601 - C=0.1
#-0.174363 - C=0.01
#-0.174183 - C=0.001


#text_fields = scipy.sparse.hstack([description,summary])
#text_fields = scipy.sparse.hstack([description,summary,source])
text_fields = scipy.sparse.hstack([description,summary,source,location])
y_train = train['num_votes'].apply(lambda x: np.log1p(x))
#y_train = train['num_votes']



# regressor = SGDRegressor(loss='squared_loss', penalty='l2', alpha=0.001, l1_ratio=0.15, fit_intercept=True,
#                          max_iter=1000, tol=0.001, shuffle=True, verbose=0, epsilon=0.1, random_state=999,
#                          learning_rate='invscaling', eta0=0.01, power_t=0.25, early_stopping=False,
#                          validation_fraction=0.1, n_iter_no_change=5, warm_start=False, average=False)

# regressor = sklearn.linear_model.ElasticNetCV(l1_ratio=0.5, eps=0.001, n_alphas=100, alphas=None, fit_intercept=True,
#                                 normalize=False, precompute='auto', max_iter=1000, tol=0.0001, cv=5, copy_X=True,
#                                 verbose=0, n_jobs=1, positive=False, random_state=666, selection='cyclic')

# regressor = sklearn.linear_model.RidgeCV(alphas=(0.001, 0.01, 0.1, 1.0, 10.0), fit_intercept=True, normalize=False, scoring=None, cv=5, gcv_mode=None, store_cv_values=False)
# regressor.fit(text_fields,y_train)
# alpha = regressor.alpha_
# regressor = sklearn.linear_model.Ridge(alpha=alpha, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=0.001, solver='auto', random_state=999)

# regressor = sklearn.linear_model.LassoCV(eps=0.001, n_alphas=10, alphas=None, fit_intercept=True, normalize=False,
#                                          precompute='auto', max_iter=1000, tol=0.0001, copy_X=True, cv='warn',
#                                          verbose=False, n_jobs=None, positive=False, random_state=999,
#                                          selection='cyclic')

# regressor = GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=100, subsample=1.0,
 #                 criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
#                 max_depth=3, min_impurity_decrease=0.0, min_impurity_split=None, init=None, random_state=999,
#                 max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto',
#                 validation_fraction=0.1, n_iter_no_change=None, tol=0.0001)


regressor = sklearn.svm.LinearSVR(epsilon=0.0, tol=0.0001, C=0.01, loss='epsilon_insensitive',
                                 fit_intercept=True, intercept_scaling=1.0, dual=True, verbose=0,
                                random_state=999, max_iter=1000)

result = cross_validate(regressor,X=text_fields,y=y_train,cv=5,scoring={'rmsle':rmsle_score,'r2':'r2','explained_variance':'explained_variance'}, return_estimator=True)


print("Overall RMSLE: %f" % np.mean(result['test_rmsle']))
print("R2: %f" % np.mean(result['test_r2']))
print(result['test_r2'])

# from prettytable import PrettyTable
# print("\n BOW features")
# x = PrettyTable()
# #x.field_names = [" ","Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5"]
# x.field_names = [" ","Fold 1", "Fold 2"]
# x.add_row(["RMSLE: "] + [str(v) for v in result['test_rmsle']])
# print(x)


############################### StatsModels ###################################
######### Generating Diagnostic Plots for the data #################

y_train = train['num_votes'].apply(lambda x: np.log(x))

#y_train = train['num_votes']

print(type(text_fields))
text_fields = scipy.sparse.hstack([summary,source,location],format='csr')
rows = int(0.2 * (text_fields.get_shape()[0]))
text_fields = text_fields[0:rows,:]
text_fields = text_fields.toarray()
#diagnostic_plots(text_fields,y_train[0:rows])
#plt.show()


