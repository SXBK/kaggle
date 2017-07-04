from __future__ import division
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsRegressor
import load_data
import gc
import lightgbm as lgb
import xgboost as xgb

np.random.seed(0)  # seed to shuffle the train set

n_folds = 4
verbose = True
shuffle = False

#train, test, ftrain, ftest = load_data.load(320)
train, test = load_data.load()
y = train.y.values
y_mean = np.mean(y)
X = train.drop('y', axis = 1).values
id_test = test['ID'].values
X_submission = test.values

skf = StratifiedKFold(n_splits=n_folds)

#lgb clif
gbm = lgb.LGBMRegressor(objective='regression')

#xgb clif
xgbmodel = xgb.XGBRegressor()

clfs = [xgbmodel,
        gbm,
        #RandomForestRegressor(n_estimators=30, n_jobs=-1, criterion='mae'),
        #RandomForestRegressor(n_estimators=30, n_jobs=-1, criterion='mse'),
        #KNeighborsRegressor(n_neighbors=1500, n_jobs=-1, weights='distance'),
        #ExtraTreesRegressor(n_estimators=30, n_jobs=-1, criterion='mae'),
        #ExtraTreesRegressor(n_estimators=30, n_jobs=-1, criterion='mse'),
        #GradientBoostingRegressor(learning_rate=0.01, subsample=0.5, max_depth=10, n_estimators=30)
		]

dataset_blend_train = np.zeros((X.shape[0], len(clfs)))
dataset_blend_test = np.zeros((X_submission.shape[0], len(clfs)))


for j, clf in enumerate(clfs):
    print(j, clf)
    dataset_blend_test_j = np.zeros((X_submission.shape[0], n_folds))
    for i, (train, test) in enumerate(skf.split(X, y)):
        print("Fold", i)
        X_train = X[train]
        y_train = y[train]
        X_test = X[test]
        y_test = y[test]
        print(X_train.shape)
        clf.fit(X_train, y_train)
        y_submission = clf.predict(X_test)
        dataset_blend_train[test, j] = y_submission
        dataset_blend_test_j[:, i] = clf.predict(X_submission)
    dataset_blend_test[:, j] = dataset_blend_test_j.mean(1)


from sklearn.linear_model import LinearRegression
clf = LinearRegression()
clf.fit(dataset_blend_train, y)

y_submission = clf.predict(dataset_blend_test)
y_pred = clf.predict(dataset_blend_train)

print(y_pred)
print('r2:',r2_score(y, y_pred))
