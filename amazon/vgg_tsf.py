from __future__ import absolute_import
from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, normalization
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad, Adam
from keras.utils import np_utils, generic_utils
from keras.applications.vgg16 import VGG16
import keras
import os
import numpy as np
import pandas as pd
import sys
from myamazon import f2s, load_data

num = 100
vnum = 100
times = 1

# num is the raw data num, and you'll get num * times sample
x, y, fl = load_data(num, times)
x_valid, y_valid = x[vnum:], y[vnum:]
x, y = x[:vnum], y[:vnum]

model_ORI = VGG16(weights='imagenet', include_top=True)
model = Sequential()

for i, layer in enumerate(model_ORI.layers):
    if i > len(model_ORI.layers) - 3:
        break
    layer.trainable = False
    model.add(layer)
model.add(Dense(2048, activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(2048, activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(fl, activation='sigmoid'))
model.summary()
adam = keras.optimizers.Adam(lr=1e-4, beta_1=0.9,
                             beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='categorical_crossentropy', optimizer=adam)
model.fit(x, y.values, epochs=int(sys.argv[1]))

model.save('model_vgg.h5')

y_valid = y[:5].values
x_valid = x[:5]
pred = model.predict(x_valid)
print(y_valid[:,:])
pred[pred > 0.5] = 1
pred[pred <= 0.5] = 0
print(pred[:, :])
print(f2s(pred, y_valid))
