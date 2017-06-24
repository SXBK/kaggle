import numpy as np
import pandas as pd
from keras.models import load_model
from myamazon import f2s, load_data

num = 500
vnum = 500

x, y, fl = load_data(num)
x_valid, y_valid = x[vnum:], y[vnum:]
x, y = x[:vnum], y[:vnum]

model=load_model('model_vgg.h5')

y_valid = y[:5].values
x_valid = x[:5]
pred = model.predict(x_valid)
print(y_valid[:,:])
pred[pred > 0.5] = 1
pred[pred <= 0.5] = 0
print(pred[:, :])
print(f2s(pred, y_valid))
