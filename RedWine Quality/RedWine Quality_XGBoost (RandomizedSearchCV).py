# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 20:54:49 2018

@author: yhj
"""

import numpy as np
import pandas as pd
from time import time
import scipy.stats as st

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# data loding
data = pd.read_csv('C:/data/winequality-red.csv')


# Dividing Quality as bad(0), good(1), Excellent(2)
# good quality is 3~6 / Excellent quality is 7~8
data['quality'] = pd.cut(data['quality'], bins = [1,6.5,10], labels = [0,1])
sns.countplot(x='quality', data=data)


X = data.iloc[:,:-1]
y = data['quality']

# Standatdising Data
# Scale the data to be between -1 and 1
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

# PCA - Principal Component Analysis
from sklearn.decomposition import PCA
pca = PCA()
pca.fit_transform(X)

pca_comp = PCA(n_components = 8)
X = pca_comp.fit_transform(X)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=25)


def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")



# XGBoost
import xgboost as xgb

param = {
        'objective' = 'binary:logistic'
        'n_estimators' : 100,
        'max_depth' : 5,
        'learning_rate' : 0.1,
        'colsample_bytree' : 0.8,
        'subsample' : 0.8,
        'gamma' : 0,
        'min_child_weight' : 1
}

xgtrain = xgb.DMatrix(X_train, label=y_train)
xgtest = xgb.DMatrix(X_test, label=y_test)

num_rounds = 100
watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]
xgb_model = xgb.train(param, xgtrain, num_rounds, watchlist, early_stopping_rounds=20)

y_pred = xgb_model.predict(xgtest)

xgb_model.attributes()
xgb_model.get_score()
xgb.plot_importance(xgb_model)

predictions = [round(value) for value in y_pred]
acc =  metrics.accuracy_score(y_test, predictions)*100
print('\nAccuracy: %.2f %%\n' % acc)



from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import cross_val_score

param_dist = {  
        'n_estimators' : st.randint(20, 100),
        'max_depth' : st.randint(5, 20),
        'learning_rate' : st.uniform(0.05, 0.2),
        'colsample_bytree' : st.beta(10, 1),
        'subsample' : st.beta(10, 1),
        'gamma' : st.uniform(0, 10),
        'min_child_weight' : st.expon(0, 10)
}

xgbc = XGBClassifier(objective='binary:logistic', nthreads=-1)
xgbc.fit(X_train, y_train)
y_pred = xgbc.predict(X_test)


n_iter_search = 20
RSCV_xgbc = RandomizedSearchCV(xgbc, param_dist, scoring='accuracy', n_iter=n_iter_search, cv=10)

RSCV_xgbc.fit(X_train, y_train)  
RSCV_xgbc.best_params_
RSCV_xgbc.best_score_

y_pred = RSCV_xgbc.predict(X_test)

cross_val_score(xgbc, X_train, y_train)  

conf_matrix = metrics.confusion_matrix(y_test,y_pred)
predictions = [round(value) for value in y_pred]
acc =  metrics.accuracy_score(y_test, predictions)*100
print(conf_matrix)
print('\nAccuracy: %.2f %%\n' % acc)
print(classification_report(y_test, y_pred))



