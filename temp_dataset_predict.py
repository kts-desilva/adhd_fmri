import tensorflow as tf
import scipy.ndimage
from scipy.misc import imsave
import matplotlib.pyplot as plt
import numpy as np
import glob
import nibabel as nib #reading MR images
from sklearn.model_selection import train_test_split
import keras
from keras.models import model_from_json

fp = glob.glob('predictionDataFolder/*')

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

for i in range(0,len(fp)):
    ndata = nib.load(fp[i])
    ndata = ndata.get_data()
    ndata=np.asarray(ndata)
    nimage=ndata.reshape(1,49*58,47,1)
    prediction=loaded_model.predict_classes(nimage)
    print("prediction for {num}: {pre}".format(num=i,pre=prediction))
