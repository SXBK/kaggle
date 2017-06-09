import pandas as pd
import numpy as np
from pandas import Series, DataFrame
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report
from datetime import datetime
import re

train = pd.read_csv('data/train.csv')

hour=[]
day=[]
month=[]
year=[]
for d in train['datetime']:
	ru=re.findall("(.*?)-(.*?)-(.*?) (.*?):", d)
	hour.append(int(ru[0][3]))
	day.append(int(ru[0][2]))
	month.append(int(ru[0][1]))
	year.append(int(ru[0][0]))
train['hour']=hour
train['day']=day
train['month']=month
train['year']=year
train["date"] = train.datetime.apply(lambda x : x.split()[0])
train["weekday"] = train['date'].apply(lambda dateString : datetime.strptime(dateString,"%Y-%m-%d").weekday())



test=pd.read_csv('data/test.csv')

hour=[]
day=[]
month=[]
year=[]
for d in test['datetime']:
	ru=re.findall("(.*?)-(.*?)-(.*?) (.*?):", d)
	hour.append(int(ru[0][3]))
	day.append(int(ru[0][2]))
	month.append(int(ru[0][1]))
	year.append(int(ru[0][0]))
test['hour']=hour
test['day']=day
test['month']=month
test['year']=year
test["date"] = train.datetime.apply(lambda x : x.split()[0])
test["weekday"] = test['date'].apply(lambda dateString : datetime.strptime(dateString,"%Y-%m-%d").weekday())

categoryVariableList = ["year","month","day","hour","season","weather","holiday","workingday","weekday"]
for var in categoryVariableList:
	train[var]=train[var].astype("category")
	test[var]=test[var].astype("category")
'''
zcoreList = ["temp", "humidity", "windspeed"]
for var in zcoreList:
	tMean = np.average(train[var])
	tStd = np.std(train[var])
	train[var]=[(x-tMean)/tStd for x in train[var]]
	tMean = np.average(test[var])
	tStd = np.std(test[var])
	test[var]=[(x-tMean)/tStd for x in test[var]]
countMean = np.average(train[var])
countStd = np.std(train[var])
train["count"]=[(x-countMean)/countStd for x in train["count"]]
'''
x_train = train[["year","month","day",'hour','season','holiday','workingday','weather','temp','humidity','windspeed','weekday']]
#print(x_train.head())
y_train = train['count']
#print(y_train.head())

x_test = test[["year","month","day",'hour','season','holiday','workingday','weather','temp','humidity','windspeed','weekday']]

rfModel = RandomForestRegressor(n_estimators=100)
rfModel.fit(x_train, y_train)
preds=rfModel.predict(X= x_train)
preds=[int(x) for x in preds]
#preds=[x*countStd+countMean for x in preds]
print(classification_report(y_train, preds))
for i in range(10):
	print(preds[i], y_train.ix[i])
preds=rfModel.predict(X= x_test)
#preds=[x*countStd+countMean for x in preds]
print(preds[:5])
submission=pd.DataFrame({"datetime":test['datetime'],"count":[max(0, x) for x in preds]})
submission.to_csv('output/rf.csv', index=False)
