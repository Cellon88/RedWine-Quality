# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 20:54:49 2018

@author: yhj
"""

# RedWine Quality Analysis
# https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009
#


import numpy as np
import pandas as pd
from time import time
import scipy.stats as st

import seaborn as sns
import matplotlib.pyplot as plt

# data loding
data = pd.read_csv('C:/data/winequality-red.csv')

# data check
data.head(10)
data.info()
data.describe()

# Correlation Analysis
data.corr()
data.corr()['quality'].sort_values(ascending=False)
# Seaborn Heatmap, cmap param = Blues, Greys, OrRd, RdBu_r, Reds
plt.figure(figsize=(10,10))
sns.heatmap(data.corr(), cmap='RdBu_r', annot=True, linewidths=0.5)


# volatile acidity, citric acid, sulphates, alcohol
# fixed acidity, chlorides, total sulfur dioxide, density
# Use Boxplot
fig, axs = plt.subplots(2, 4, figsize = (16,8)) 
ax1 = plt.subplot2grid((5,15), (0,0), rowspan=2, colspan=3) 
ax2 = plt.subplot2grid((5,15), (0,4), rowspan=2, colspan=3)
ax3 = plt.subplot2grid((5,15), (0,8), rowspan=2, colspan=3)
ax4 = plt.subplot2grid((5,15), (0,12), rowspan=2, colspan=3)
ax5 = plt.subplot2grid((5,15), (3,0), rowspan=2, colspan=3) 
ax6 = plt.subplot2grid((5,15), (3,4), rowspan=2, colspan=3)
ax7 = plt.subplot2grid((5,15), (3,8), rowspan=2, colspan=3)
ax8 = plt.subplot2grid((5,15), (3,12), rowspan=3, colspan=3)

sns.boxplot(x='quality',y='volatile acidity', data = data, ax=ax1)
sns.boxplot(x='quality',y='citric acid', data = data, ax=ax2)
sns.boxplot(x='quality',y='sulphates', data = data, ax=ax3)
sns.boxplot(x='quality',y='alcohol', data = data, ax=ax4)
sns.boxplot(x='quality',y='fixed acidity', data = data, ax=ax5)
sns.boxplot(x='quality',y='chlorides', data = data, ax=ax6)
sns.boxplot(x='quality',y='total sulfur dioxide', data = data, ax=ax7)
sns.boxplot(x='quality',y='density', data = data, ax=ax8)


sns.countplot(x='quality', data=data)
((sum(data['quality'] == 5) + sum(data['quality'] == 6)) / len(data['quality']))*100
# 전체 Quality중 5와 6의 값이 전체의 82%를 차지한다. 사실상 (5이하)와 (6이상)구분의 이진 분류로 봐도 무방하다.

# Dividing Quality as bad(0), good(1), Excellent(2)
# Bad quality is 3~4 / Good quality is 5~6 / Excellent quality is 7~8
data['quality'] = pd.cut(data['quality'], bins = [1,4.5,6.5,10], labels = [0,1,2])
sns.countplot(x='quality', data=data)


X = data.iloc[:,:-1]
y = data['quality']


# # Sscale standization
# Scale the data to be between -1 and 1
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)


# PCA - Principal Component Analysis
from sklearn.decomposition import PCA
pca = PCA()
pca.fit_transform(X)

# PCA Components - Barplot
plt.figure(figsize=(6,6))
sns.barplot(x=list(range(len(pca.explained_variance_))), y=pca.explained_variance_, palette="Blues_d")
plt.ylabel('Explained variance Value')
plt.xlabel('Principal components')
plt.grid(True)
plt.tight_layout()

# PCA Components Ratio - plot
plt.figure(figsize=(6,6))
plt.plot(np.cumsum(pca.explained_variance_ratio_), 'ro-')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.grid(True)
plt.tight_layout()

# Components 0 ~ 8 is explain the data 95% more
pca_comp = PCA(n_components = 8)
X = pca_comp.fit_transform(X)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=25)




# 1. Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

lr = LogisticRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)

conf_matrix = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)*100
print(conf_matrix)
print('\nAccuracy : %0.2f %%\n' % acc)
print(classification_report(y_test, y_pred))




# 2. Decision Tree
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(max_depth = 5)
dt.fit(X_train,y_train)
y_pred = dt.predict(X_test)

conf_matrix = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)*100
print(conf_matrix)
print('\nAccuracy : %0.2f %%\n' % acc)
print(classification_report(y_test, y_pred))



# GridSearchCV, RandomizedSearchCV Report Function 
# -> by. scikit-learn.org "Comparing randomized search and grid search for hyperparameter estimation"
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




# 3. SVC / GridSearchCV
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

# SVM - Classifier
svc = SVC()

# Search Bast param
param_dist = {
        'C' : [0.9, 1, 1.1, 1.2, 2, 5, 10], 'gamma'  : [0.9, 1, 1.1, 1.2, 2, 5, 10], 'kernel' : ['rbf'],
        'C' : [0.9, 1, 1.1, 1.2, 2, 5, 10], 'degree' : [2,3,4,5,6],                  'kernel' : ['poly']
}

# CAUTION! GridSearchCV is takes a lot of resources and time.
# if you want faster result, using param 'n_jobs = -1'
# GSCV_svc = GridSearchCV(svc, param_dist, scoring='accuracy', cv=10, n_jobs=-1)
GSCV_svc = GridSearchCV(svc, param_dist, scoring='accuracy', cv=10)

start = time()
GSCV_svc.fit(X_train, y_train)

print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(GSCV_svc.cv_results_['params'])))
report(GSCV_svc.cv_results_)

y_pred = GSCV_svc.predict(X_test)

conf_matrix = metrics.confusion_matrix(y_test,y_pred)
acc = metrics.accuracy_score(y_test, y_pred)*100
print(conf_matrix)
print('\nAccuracy : %0.2f %%\n' % acc)
print(classification_report(y_test, y_pred))




# 4. Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
rf = RandomForestClassifier()

param_dist = {
        'n_estimators': st.randint(10, 100),
        'max_depth' : [3, None],
        'max_features' : st.randint(1, 8),
        'min_samples_split' : st.randint(2, 8),
        'min_samples_leaf' : st.randint(1, 8),
        'bootstrap' : [True, False],
        'criterion' : ["gini", "entropy"]
}

n_iter_search = 20
RSCV_rf = RandomizedSearchCV(rf, param_dist, scoring='accuracy', n_iter=n_iter_search, cv=10)

start = time()
RSCV_rf.fit(X_train, y_train)

print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
report(RSCV_rf.cv_results_)

y_pred = RSCV_rf.predict(X_test)

conf_matrix = metrics.confusion_matrix(y_test,y_pred)
acc = metrics.accuracy_score(y_test, y_pred)*100
print(conf_matrix)
print('\nAccuracy : %0.2f %%\n' % acc)
print(classification_report(y_test, y_pred))


importances = RSCV_rf.best_estimator_.feature_importances_
std = np.std([tree.feature_importances_ for tree in RSCV_rf.best_estimator_], axis=0)
indices = np.argsort(importances)[::-1]

print("Feature ranking:")
for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices], color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.grid(True)




# 5. GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier()

param_dist = {
        'n_estimators': st.randint(10, 100),
        'max_depth': [3, None],
        'min_samples_leaf': st.randint(1, 5)
}

n_iter_search = 20
RSCV_gb = RandomizedSearchCV(gb, param_dist, scoring='accuracy', n_iter=n_iter_search, cv=10)

start = time()
RSCV_gb.fit(X_train, y_train)

print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
report(RSCV_gb.cv_results_)

y_pred = RSCV_gb.predict(X_test)

conf_matrix = metrics.confusion_matrix(y_test, y_pred)
acc = metrics.accuracy_score(y_test, y_pred)*100
print(conf_matrix)
print('\nAccuracy : %0.2f %%\n' % acc)
print(classification_report(y_test, y_pred))


importances = RSCV_gb.best_estimator_.feature_importances_
indices = np.argsort(importances)[::-1]

print("Feature ranking:")
for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices], color="r",align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.grid(True)


# XGBoost
import xgboost as xgb

param = {
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




xgbc = xgb.XGBClassifier()

param_dist = {  
        'n_estimators' : st.randint(20, 100),
        'max_depth' : st.randint(5, 20),
        'learning_rate' : st.uniform(0.05, 0.2),
        'colsample_bytree' : st.beta(10, 1),
        'subsample' : st.beta(10, 1),
        'gamma' : st.uniform(0, 10),
        'min_child_weight' : st.expon(0, 10)
}

xgbc = xgb.XGBClassifier(nthreads=-1)

n_iter_search = 20
RSCV_xgbc = RandomizedSearchCV(xgbc, param_dist, scoring='accuracy', n_iter=n_iter_search, cv=10)

RSCV_xgbc.fit(X_train, y_train)  
RSCV_xgbc.best_params_
RSCV_xgbc.best_score_

y_pred = RSCV_xgbc.predict(X_test)

conf_matrix = metrics.confusion_matrix(y_test,y_pred)
predictions = [round(value) for value in y_pred]
acc =  metrics.accuracy_score(y_test, predictions)*100
print(conf_matrix)
print('\nAccuracy: %.2f %%\n' % acc)
print(classification_report(y_test, y_pred))

















from matplotlib.colors import ListedColormap

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)



plot_decision_regions(X_test, y_test, model)






















































