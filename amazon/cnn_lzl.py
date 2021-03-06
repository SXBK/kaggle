from __future__ import absolute_import
from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad, Adam
from keras.utils import np_utils, generic_utils
import os
import numpy as np
import pandas as pd

import cv2


num = 1000
vnum = 990
fl = 0

def load_data():
	ytrain = pd.read_csv('data/train_v2.csv')[:num]
	xtrain = []
	new_style = {'grid':False}
	for f, l in ytrain.values:
		img = cv2.imread('data/train-jpg/'+f+'.jpg')
		xtrain.append(img)
	xtrain = np.array(xtrain)
	xtrain = (xtrain - xtrain.mean())/(xtrain.std()+1e-6)
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
	allTags=list(allTags)
	print(allTags)
	fl = len(allTags)
	fs = np.zeros((num, fl), dtype="uint8")
	for j, c in zip(range(len(tags)),tags):
		for i in range(fl):
			fs[j,i] = 1 if c.find(allTags[i])>=0 else 0
	ytrain = pd.DataFrame(data=fs, columns=allTags)
	return ytrain


x, y = load_data()
x_valid , y_valid = x[vnum:], y[vnum:]
x, y = x[:vnum], y[:vnum]
print(y.shape)

#generate cnn
model = Sequential()
#first juanji
model.add(Convolution2D(4,(5,5),input_shape=(256, 256, 3), activation='relu'))

#second conv
model.add(Convolution2D(8, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2)))

#third conv
model.add(Convolution2D(16, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2)))

#quanlianjie
model.add(Flatten())
model.add(Dense(128, activation='relu'))

#softmax
model.add(Dense(fl, activation='sigmoid'))

#training
sgd = SGD(lr=0.005, decay=1e-5, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, class_mode='categorical')


model.fit(x, y.values, batch_size=50,epochs=100,shuffle=True,verbose=1, validation_split=0.1)

model.save('model.h5')
pred = model.predict(x_valid[:10])
pred[pred>0.5]=1
pred[pred<0.5]=0
print(y_valid.head(10))
print(pred[:,:])
