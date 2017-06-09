# %% import modules
import numpy as np
import pandas as pd
# import matplotlib.pylab as plt
# from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

# %% loadData
train = pd.read_csv('train.csv')
clms = list(train.columns)
clms.remove('datetime')
clms.remove('casual')
clms.remove('registered')
data = np.array(train.ix[:, clms], dtype=np.float64)
atime = np.array([np.float64(time.split(' ')[1].split(':')[0])
                  for time in train.datetime]).reshape(-1, 1)
data = np.hstack((atime, data))

test = pd.read_csv('test.csv')
tclms = list(test.columns)
tclms.remove('datetime')
tdata = np.array(test.ix[:, tclms], dtype=np.float64)
tatime = np.array([np.float64(time.split(' ')[1].split(':')[0])
                   for time in test.datetime]).reshape(-1, 1)
tdata = np.hstack((tatime, tdata))

# %% knn method
# knnrgs = KNeighborsRegressor(n_neighbors=50)
# knnrgs.fit(data[:,:-1], data[:,-1])
# knnrgs.predict(data[1,:-1])

# %% random forest method
rfrgs = RandomForestRegressor()
rfrgs.fit(data[:, :-1], data[:, -1])
print("prediction", "reality")
for i in range(10):
    print(rfrgs.predict(data[i, :-1].reshape(1, -1)), data[i, -1])

result = rfrgs.predict(tdata)
rdata = pd.DataFrame(np.hstack(
    (np.array(test.datetime).reshape(-1, 1), result.reshape(-1, 1))), columns=["datetime", "count"])

rdata.to_csv("a.csv", index=False)
