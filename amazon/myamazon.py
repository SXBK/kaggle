#common utils
import numpy as np
import pandas as pd
from keras.preprocessing import image

def f2s(pred, valid):
	pred = np.asarray(pred, dtype=np.int)
	valid = np.asarray(valid)
	rp = np.bitwise_and(pred, valid).sum()
	p = rp*1.0/pred.sum()
	r = rp*1.0/valid.sum()
	return 5*p*r/(4*p + r)

def load_data(num):
	ytrain = pd.read_csv('data/train_v2.csv')[:num]
	xtrain = []
	new_style = {'grid': False}
	for f, l in ytrain.values:
		img_path = 'data/train-jpg/' + f + '.jpg'
		img = image.load_img(img_path, target_size=(224, 224))
		x = image.img_to_array(img)
		xtrain.append(x)
	xtrain = np.array(xtrain)
	ytrain = ytrain['tags'].values
	ytrain, fl = encodeY(ytrain, num)
	xtrain = (xtrain - xtrain.mean())/(xtrain.std()+1e-8)
	print(ytrain.shape, xtrain.shape)

	return xtrain, ytrain, fl


def encodeY(tags, num):
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
	return ytrain, fl

