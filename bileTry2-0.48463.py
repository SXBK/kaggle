import pandas as pd
import numpy as np
from pandas import Series, DataFrame
from sklearn.ensemble import RandomForestRegressor
import re

train = pd.read_csv('data/train.csv')

hour=[]
for d in train['datetime']:
	hour.append(int(re.findall(".*? (.*?):", d)[0]))
train['hour']=hour


test=pd.read_csv('data/test.csv')

hour=[]
for d in test['datetime']:
	hour.append(int(re.findall(".*? (.*?):", d)[0]))
test['hour']=hour

categoryVariableList = ["hour","season","weather","holiday","workingday"]
for var in categoryVariableList:
	train[var]=train[var].astype("category")
	test[var]=test[var].astype("category")


x_train = train[['hour','season','holiday','workingday','weather','temp','humidity','windspeed']]
y_train = train['count']

x_test = test[['hour','season','holiday','workingday','weather','temp','humidity','windspeed']]

rfModel = RandomForestRegressor(n_estimators=100)
rfModel.fit(x_train, y_train)
preds=rfModel.predict(X= x_test)
print(preds[:5])
submission=pd.DataFrame({"datetime":test['datetime'],"count":[max(0, x) for x in preds]})
submission.to_csv('bike_predictions_gbm_separate_without_fe.csv', index=False)
