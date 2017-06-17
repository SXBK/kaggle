# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "data/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "data"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
import lightgbm as lgb
import gc
import sklearn
import commands
import matplotlib.pyplot as plt

def log(sta, info):
	#commands.getstatusoutput('cat "' + sta+ ': ' + info + '">> out/mse.log')
	return 1

print('Loading data ...')

train = pd.read_csv('data/train_2016.csv')
prop = pd.read_csv('data/properties_2016.csv')

for c, dtype in zip(prop.columns, prop.dtypes):	
    if dtype == np.float64:		
        prop[c] = prop[c].astype(np.float32)

df_train = train.merge(prop, how='left', on='parcelid')

#drop miss to much
df_missing = df_train.isnull().sum(axis=1).reset_index()
df_missing.columns=['index','missingnum']
df_train = df_train.merge(df_missing, left_index=True, right_on='index')
df_train=df_train.ix[df_train['missingnum']<56]

x_train = df_train.drop(['parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc', 'propertycountylandusecode'], axis=1)
y_train = df_train['logerror'].values
print(x_train.shape, y_train.shape)


train_columns = x_train.columns

for c in x_train.dtypes[x_train.dtypes == object].index.values:
    x_train[c] = (x_train[c] == True)

del df_train; gc.collect()

split = 90000
x_train, y_train, x_valid, y_valid = x_train[:split], y_train[:split], x_train[split:], y_train[split:]
x_train = x_train.values.astype(np.float32, copy=False)
x_valid = x_valid.values.astype(np.float32, copy=False)

d_train = lgb.Dataset(x_train, label=y_train)
d_valid = lgb.Dataset(x_valid, label=y_valid)

params = {}
params['max_bin'] = 10
params['learning_rate'] = 0.0021 # shrinkage_rate
params['boosting_type'] = 'gbdt'
params['objective'] = 'regression'
params['metric'] = 'l2'          # or 'mae'
params['sub_feature'] = 0.5      # feature_fraction 
params['bagging_fraction'] = 0.85 # sub_row
params['bagging_freq'] = 40
params['num_leaves'] = 512        # num_leaf
params['min_data'] = 500         # min_data_in_leaf
params['min_hessian'] = 0.05     # min_sum_hessian_in_leaf

watchlist = [d_valid]
labellist = []
for i in [400, 600, 800, 1000]:
	train_time = i
	clf = lgb.train(params, d_train, train_time, watchlist)
	fig, ax = plt.subplots(figsize=(12,16))
	lgb.plot_importance(clf, ax=ax, title=str(train_time))
	labels = [int(j.get_text()[7:]) for j in ax.get_yticklabels()]
	labellist.append(labels)
#plt.show()


p_valid = clf.predict(x_valid)

for i in range(10):
	print('valid:', y_valid[i],'predict:', p_valid[i])
	log('origin lgbm','valid:'+ str(y_valid[i])+' predict:'+str(p_valid[i]))
print(sklearn.metrics.mean_squared_error(y_valid, p_valid))
log('origin lgbm',str(sklearn.metrics.mean_squared_error(y_valid, p_valid)))

bset=set()
sset=set()
for k, labels in zip([400, 600, 800, 1000], labellist):
	print('Iter is:', k)
	print('Best 15:', labels[:15])
	print('Sbst 15:', labels[-15:])
	if len(bset)==0 and len(sset)==0:    #first time
		bset=set(labels[:15])
		sset=set(labels[-15:])
	else:
		bset=bset&set(labels[:15])
		sset=sset&set(labels[-15:])

print(list(bset))
print(train_columns.values[list(bset)])
print(list(sset))
print(train_columns.values[list(sset)])


del d_train, d_valid; gc.collect()
del x_train, x_valid; gc.collect()

'''
print("Prepare for the prediction ...")
sample = pd.read_csv('data/sample_submission.csv')
sample['parcelid'] = sample['ParcelId']
df_test = sample.merge(prop, on='parcelid', how='left')
del sample, prop; gc.collect()
x_test = df_test[train_columns]
del df_test; gc.collect()
for c in x_test.dtypes[x_test.dtypes == object].index.values:
    x_test[c] = (x_test[c] == True)
x_test = x_test.values.astype(np.float32, copy=False)

print("Start prediction ...")
# num_threads > 1 will predict very slow in kernal
clf.reset_parameter({"num_threads":1})
p_test = clf.predict(x_test)


del x_test; gc.collect()

print("Start write result ...")
sub = pd.read_csv('data/sample_submission.csv')
for c in sub.columns[sub.columns != 'ParcelId']:
    sub[c] = p_test

sub.to_csv('lgb_starter_2.csv', index=False, float_format='%.4f')
'''
