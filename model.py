import keras
import tensorflow as tf


config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
session = tf.Session(config=config)
from data import get_Data, Read_Data
from PIL import Image
import numpy as np
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, Conv2DTranspose
from keras import optimizers
from keras.layers.normalization import BatchNormalization
from keras.activations import relu
from keras.layers import LocallyConnected2D, LocallyConnected1D
from keras.applications import VGG16
from keras.optimizers import SGD ,adam
import numpy as np
from keras.layers import Activation
import h5py
import theano
#
# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.3
# set_session(tf.Session(config=config))

def my_moodel(weights_path=None):

    model = Sequential()
    model.add(Convolution2D(8, (5, 5 ),padding='same', input_shape=(224, 224, 1)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))


    model.add(Convolution2D(16, (5, 5), padding='same'))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))

    model.add(Convolution2D(32, (4, 4), padding='same'))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))

    # model.add(Convolution2D(64, (4, 4), padding='same'))
    # model.add(BatchNormalization(axis=-1))
    # model.add(Activation('relu'))
    #
    # model.add(Convolution2D(128, (3, 3), padding='same'))
    # model.add(BatchNormalization(axis=-1))
    # model.add(Activation('relu'))
    #
    # model.add(Conv2DTranspose(64, (3, 3), padding='same'))
    # model.add(BatchNormalization(axis=-1))
    # model.add(Activation('relu'))
    #
    # model.add(Conv2DTranspose(32, (4, 4), padding='same'))
    # model.add(BatchNormalization(axis=-1))
    # model.add(Activation('relu'))


    model.add(Conv2DTranspose(16, (4, 4), padding='same'))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))


    model.add(Conv2DTranspose(8, (5, 5), padding='same'))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))

    model.add(Conv2DTranspose(3, (5, 5), padding='same'))


    if weights_path:
        model.load_weights(weights_path)

    return model



def run_model():
    # get the data from the file
    dataset = get_Data(400 , 0 , 0)
    model1 = my_moodel()
    # datasetInput, datasetOutput = Read_Data()



    X_train = dataset['X_train']
    y_train = dataset['y_train']
    X_val = dataset['X_val']
    y_val = dataset['y_val']
    X_test = dataset['X_test']
    y_test = dataset['y_test']

    meany = dataset['meany']

    # print(meany.shape)


    print (model1.output)
    model1.summary()
    sgd = optimizers.SGD(lr=0.001, decay=1e-3, momentum=0.9, nesterov=True)
    adam = optimizers.adam(lr=5e-2)

    model1.compile(loss='mean_squared_error', optimizer=adam)



    model1.fit(X_train[0:100], y_train[0:100], batch_size=4, epochs=125, shuffle=False)



    # model1.predict(X_test)

    #score = model1.evaluate(datasetInput[0:50], datasetOutput[0:50], batch_size=1)
   # print(score)
    #print(hist)
    # hist = model.fit(X_train ,y_train ,validation_split=0.2 )

    arr = model1.predict(X_train[0:200], batch_size=8)
    # print(arr.shape)
    arr += meany
    for i in range(100):
        img = Image.fromarray(arr[i].astype(np.uint8), 'RGB')
        img.save('/home/ziad/Documents/image colorization/train/' + str(i) + 'my.png')

    for i in range(100 ,200):
        img = Image.fromarray(arr[i].astype(np.uint8), 'RGB')
        img.save('/home/ziad/Documents/image colorization/test/' + str(i) + 'my.png')

    model1.save_weights('/home/ziad/Documents/image colorization/weights.h5')


run_model()
