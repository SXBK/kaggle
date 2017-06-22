import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tushare as ts
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

total = ts.get_k_data('600848', start='2011-01-05').drop(['code'], axis=1)
index = ts.get_k_data('000001', start='2011-01-05').drop(['code'], axis=1)
total = total.merge(index, how='left', on='date')
total = total.drop(['date'], axis=1)
print(total.head())

#scaler = StandardScaler()
#total = scaler.fit_transform(total)
#total = (total - total.mean())/(total.var()+1e-8)

y = total['open_x'].values[1:]
x = np.reshape(total.values,(-1,1,10))[:-1]
print(x)

split = 1000
x_train, x_test, y_train, y_test = x[:split,:,:], x[split:,:,:], y[:split], y[split:]
print(x_train)
print(y_train)

#create and fit the LSTM network
model = Sequential()
#model.add(LSTM(4, input_shape=(1, 10),return_sequences=True))
model.add(LSTM(4, input_shape=(1, 10)))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=16)

pred = model.predict(x_test)
print('MSE:',mean_squared_error(pred, y_test) )
