'''benz with feature_selection
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import feature_selection
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
from sklearn.feature_selection import chi2

import lightgbm as lgb

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

for c in train.columns:
    if train[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(train[c].values) + list(test[c].values))
        train[c] = lbl.transform(list(train[c].values))
        test[c] = lbl.transform(list(test[c].values))


trainX = train.drop(["ID", "y"], axis=1).values
trainY = train['y'].values

model = feature_selection.feature_selection(
    trainX, trainY, chi2, method="SelectKBest", k=150)

# trainX_new = model.transform(trainX)
params = {}
params['max_bin'] = 10
params['learning_rate'] = 0.003 # shrinkage_rate
params['boosting_type'] = 'gbdt'
params['objective'] = 'regression'
params['metric'] = 'l1'          # or 'mae'
params['sub_feature'] = 0.95      # feature_fraction
params['bagging_fraction'] = 0.85 # sub_row
params['bagging_freq'] = 20
params['num_leaves'] = 512        # num_leaf
params['min_data'] = 500         # min_data_in_leaf
params['min_hessian'] = 0.05     # min_sum_hessian_in_leaf

split = 3000

trainX, trainY, validX, validY =\
 trainX[:split], trainY[:split], trainX[split:], trainY[split:]

d_train = lgb.Dataset(trainX, label=trainY)
d_valid = lgb.Dataset(validX, label=validY)
clf = lgb.train(params, d_train, 2000, [d_valid])

ypred = clf.predict(validX)
print("R2 score on valid is {}".format(r2_score(validY, ypred)))
