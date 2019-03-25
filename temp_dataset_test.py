import tensorflow as tf
import scipy.ndimage
from scipy.misc import imsave
import matplotlib.pyplot as plt
import numpy as np
import glob
import nibabel as nib #reading MR images
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

plt.switch_backend('agg')

ff = glob.glob('preDataFolder/*')
ini_labels=[3,3,0,0,0,2,2,2,0,3,3,0,0,0,0,0,0]
#ini_labels=['inattentive','inattentive','normal','normal','normal','hyperactivity','hyperactivity']
batch_size = 4
num_classes = 4
epochs = 100

images = []
labels=[]
for f in range(len(ff)):
    a = nib.load(ff[f])
    a = a.get_data()
    print('a images shape: {shape}'.format(shape=a.shape))
    images.append(a)

images = np.asarray(images)
images=images.reshape(17,49*58,47,1)
labels = np.asarray(ini_labels)
print('images shape: {shape}'.format(shape=images.shape))
print('labels shape: {shape}'.format(shape=labels.shape))

x_train,x_test,y_train,y_test = train_test_split(images,labels,test_size=0.2,random_state=13)

print('x_train shape: {shape}'.format(shape=x_train.shape))
print('y_train shape: {shape}'.format(shape=y_train.shape))
print('y_train',y_train[:5])

 #convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train,4)
y_test = keras.utils.to_categorical(y_test,4)
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(49*58,47,1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

history=model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('accuracy.png')
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('loss.png')


# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

