# common utils
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


def f2s(pred, valid):
    pred = np.asarray(pred, dtype=np.int)
    valid = np.asarray(valid)
    rp = np.bitwise_and(pred, valid).sum()
    p = rp * 1.0 / pred.sum()
    r = rp * 1.0 / valid.sum()
    return 5 * p * r / (4 * p + r)


def load_data(num, imgtimes=0):
    ytrain = pd.read_csv('data/train_v2.csv')[:num]
    xtrain = []
    new_style = {'grid': False}
    for f, l in ytrain.values:
        img_path = 'data/train-jpg/' + f + '.jpg'
        img = load_img(img_path, target_size=(224, 224))
        x = img_to_array(img)
        xtrain.append(x)
    xtrain = np.array(xtrain)
    ytrain = ytrain['tags'].values
    ytrain, fl = encodeY(ytrain, num)
    #xtrain = (xtrain - xtrain.mean()) / (xtrain.std() + 1e-8)
    if imgtimes > 1:
        xtrain, ytrain = imageStrong(xtrain, ytrain, imgtimes)
    print(ytrain.shape, xtrain.shape)

    return xtrain, ytrain, fl


def imageStrong(x, y, imgtimes):
    datagen = ImageDataGenerator(
        rotation_range=90,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest')
    nimg = np.array([])
    nlab = pd.DataFrame()
    for i in range(imgtimes - 1):
        for batch in datagen.flow(x, batch_size=x.shape[0]):
            if nimg.shape[0] < batch.shape[0]:
                nimg = np.array(batch, copy=True)
                #nlab = np.array(y, copy=True)
                nlab = y.copy()
            else:
                nimg = np.concatenate((nimg, np.array(batch)))
                #nlab = np.concatenate((nlab, np.array(y)))
                nlab = nlab.append(y)
            break
    x = np.concatenate((x, nimg))
    #y = np.concatenate((y, nlab))
    y = y.append(nlab)
    return x, y


def encodeY(tags, num):
    allTags = set()
    for t in tags:
        for tag in t.split(' '):
            allTags.add(tag)
    allTags = list(allTags)
    print(allTags)
    fl = len(allTags)
    fs = np.zeros((len(num), fl), dtype="uint8")
    for j, c in zip(range(len(tags)), tags):
        for i in range(fl):
            fs[j, i] = 1 if c.find(allTags[i]) >= 0 else 0
    ytrain = pd.DataFrame(data=fs, columns=allTags)
    return ytrain, fl


def load_hdfdata(num):
    if np.isscalar(num):
        num = [0, num]
    if len(num) == 2:
        num = np.arange(num[0], num[1])

    ytrain = pd.read_csv('data/train_v2.csv')['tags'].values[num]
    xtrain = np.reshape(pd.read_hdf('data/train_amazon.hdf5',
                                    'train_amazon').values,
                        (-1, 224, 224, 3))
    xtrain = xtrain[num]
    ytrain, fl = encodeY(ytrain, num)

    print(ytrain.shape, xtrain.shape)

    return xtrain, ytrain.values, fl
