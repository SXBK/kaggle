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
from keras.preprocessing import image
import keras
import os
import numpy as np
import pandas as pd
import sys
import cv2


num = 600
fl = 0


def load_data():
    ytrain = pd.read_csv('train_v2.csv')[:num]
    xtrain = []
    new_style = {'grid': False}
    for f, l in ytrain.values:
        img_path = 'train-jpg/' + f + '.jpg'
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        xtrain.append(x)
    xtrain = np.array(xtrain)
    ytrain = ytrain['tags'].values
    ytrain = encodeY(ytrain)
    print(ytrain.shape, xtrain.shape)

    return xtrain, ytrain


def encodeY(tags):
    global fl
    allTags = set()
    for t in tags:
        for tag in t.split(' '):
            allTags.add(tag)
    allTags = list(allTags)
    print(allTags)
    fl = len(allTags)
    fs = np.zeros((num, fl), dtype="uint8")
    for j, c in zip(range(len(tags)), tags):
        for i in range(fl):
            fs[j, i] = 1 if c.find(allTags[i]) >= 0 else 0
    ytrain = pd.DataFrame(data=fs, columns=allTags)
    return ytrain


x, y = load_data()
x = x - x.mean()
vnum = 500
x_valid, y_valid = x[vnum:], y[vnum:]
x, y = x[:vnum], y[:vnum]
print(y.shape)

model_ORI = VGG16(weights='imagenet', include_top=True)
model = Sequential()

for i, layer in enumerate(model_ORI.layers):
    if i > len(model_ORI.layers) - 2:
        break
    layer.trainable = False
    model.add(layer)

model.add(Dense(fl, activation='sigmoid'))
model.summary()
adam = keras.optimizers.Adam(lr=1e-4, beta_1=0.9,
                             beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='categorical_crossentropy', optimizer=adam)
model.fit(x, y.values, epochs=int(sys.argv[1]))

pred = model.predict(x)
print(y.head())
q = pred[:5, :].T
q[q > 0.5] = 1
q[q <= 0.5] = 0
print(q)
