'''convert img to hdf5
'''
import glob
import pandas as pd
from keras.preprocessing import image
import numpy as np
from progressive.bar import Bar

imglist = glob.glob('data/train-jpg/*.jpg')
allimg = sorted(imglist, key=lambda x: int(x[21:].split('.')[0]))

imgsize = 224
totalimg = len(imglist)
alldata = np.zeros([totalimg, imgsize * imgsize * 3], dtype=np.uint8)

pb = Bar(max_value=totalimg, fallback=True)
pb.cursor.clear_lines(2)
pb.cursor.save()
for index, img in enumerate(allimg):
    img = image.load_img(img, target_size=(224, 224))
    alldata[index, :] = image.img_to_array(img).reshape(-1)
    pb.cursor.restore()
    pb.draw(index)

pddata = pd.DataFrame(alldata)
pddata.to_hdf('data/train_amazon.hdf5', 'train_amazon')
