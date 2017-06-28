
'''Transform Learning using vgg16 models
'''
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
from numpy import random
import pandas as pd
import sys
from myamazon import f2s, load_data, load_hdfdata

num = 5000

meanArray = np.array([79.4163002113, 86.8382492145, 76.2017108336])

# num is the raw data num, and you'll get num * times sample
x, y, fl = load_hdfdata(num)
print("load {} samples finished".format(num))


def make_batches(tsize, bsize):
    num_batches = int(np.ceil(tsize / float(bsize)))
    return [(i * bsize, min(tsize, (i + 1) * bsize))
            for i in range(0, num_batches)]


def generate_from_data(xtrain, ytrain, batch_size, steps_per_epoch):
    index = np.arange(num)
    random.shuffle(index)

    batches = make_batches(num, batch_size)
    bidx = 0
    while True:
        b = batches[bidx%len(batches)]
        if (bidx % len(batches) == 0 and bidx != 0):
            random.shuffle(index)

        bidx += 1

        if (bidx % steps_per_epoch) == 0:
            bidx = 0
            random.shuffle(index)

        idx = index[b[0]:b[1]]
        x = np.float32(xtrain[idx]) - meanArray
        y = ytrain[idx]
        yield x, y


model_ORI = VGG16(weights='imagenet', include_top=True)
model = Sequential()

for i, layer in enumerate(model_ORI.layers):
    if i > len(model_ORI.layers) - 3:
        break
    layer.trainable = False
    model.add(layer)
model.add(Dense(2048, activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(2048, activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(fl, activation='sigmoid'))
model.summary()
adam = keras.optimizers.Adam(lr=1e-4, beta_1=0.9,
                             beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='categorical_crossentropy', optimizer=adam)
# model.fit(x, y, epochs=int(sys.argv[1]))
model.fit_generator(generate_from_data(x, y, 32, 500), steps_per_epoch=500,
                    epochs=2, max_q_size=1, workers=1, pickle_safe=True)

model.save('model_vgg.h5')

y_valid = y[:5]
x_valid = x[:5]
pred = model.predict(x_valid)
print(y_valid[:, :])
pred[pred > 0.5] = 1
pred[pred <= 0.5] = 0
print(pred[:, :])
print(f2s(pred, y_valid))
