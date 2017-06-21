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

total = ts.get_k_data('600848', start='2011-01-05')
total = total.drop(['date','code'], axis=1)
print(total.head())

#scaler = StandardScaler()
#total = scaler.fit_transform(total)
total = (total - total.mean())/total.var()
print(total.head())

y = total['open']
#x = total.drop(['open'], axis=1)
x = total

split = 1000
x_train, x_test, y_train, y_test = x[:split], x[split:], y[:split], y[split:]



#create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(1, 5)))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, nb_epoch=100, batch_size=32)

plt.figure()
#plt.plot(range(y.shape[0]), y.values)
#plt.show()
