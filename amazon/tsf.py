from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers import Dense, Flatten
import keras
from keras.models import Sequential

import numpy as np

model_ORI = VGG16(weights='imagenet', include_top=True)
model = Sequential()

for i, layer in enumerate(model_ORI.layers):
    if i > len(model_ORI.layers) - 2:
        break
    model.add(layer)

model.add(Dense(17, activation='softmax'))
model.summary()
adam = keras.optimizers.Adam(lr=0.01, beta_1=0.9,
                             beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='categorical_crossentropy', optimizer=adam)

img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

features = model.predict(x)
print(features.shape)
print(features)
