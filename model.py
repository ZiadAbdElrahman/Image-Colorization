
from data import get_Data, Read_Data
from PIL import Image
import numpy as np
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, Conv2DTranspose
from keras import optimizers
from keras.layers.normalization import BatchNormalization
from keras.layers import LocallyConnected2D, LocallyConnected1D
from keras.applications import VGG16
from keras.optimizers import SGD ,adam
import numpy as np
import h5py


def my_moodel(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 1), data_format="channels_last"))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))

    model.add(BatchNormalization(axis=-1))
    # model.add(MaxPooling2D((2,2), strides=(2,2)))


    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(Convolution2D(128, 3, 3, activation='relu'))


    model.add(BatchNormalization(axis=-1))
    # model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Convolution2D(256, 2, 2, activation='relu'))
    model.add(Convolution2D(256, 2, 2, activation='relu'))

    model.add(BatchNormalization(axis=-1))


    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(Convolution2D(512, 3, 3, activation='relu'))

    model.add(BatchNormalization(axis=-1))


    model.add(Conv2DTranspose(256, 4, 4, activation='relu'))
    model.add(Conv2DTranspose(256, 3, 3, activation='relu'))
    model.add(Conv2DTranspose(256, 2, 2, activation='relu'))

    model.add(BatchNormalization(axis=-1))

    model.add(Conv2DTranspose(128, 3, 3, activation='relu'))
    model.add(Conv2DTranspose(3, 3, 3, activation='relu'))

    # model.add(MaxPooling2D((3,3), strides=(2,2)))
    # model.add(BatchNormalization(axis=-1))

    # model.add(Convolution2D(256, 2, 2, activation='relu'))
    # model.add(Convolution2D(256, 2, 2, activation='relu'))
    # model.add(Convolution2D(256, 2, 2, activation='relu'))
    #

    # model.add(Convolution2D(256, 224, 224, activation='relu'))

    if weights_path:
        model.load_weights(weights_path)

    return model


def extra(base_model):
    model = Sequential()

    model.add(LocallyConnected2D(224, (1, 1), input_shape=(base_model.output_shape[1], 1)))
    return model


def run_model():
    # get the data from the file
    # dataset = get_Data()
    model1 = my_moodel()
    datasetInput, datasetOutput = Read_Data()

    # model1 = VGG16(weights=None,input_shape=(224, 224, 3))
    # model1.load_weights('vgg16_weights_tf_dim_ordering_tf_kernels.h5')

    # model2 = extra(model1)

    #
    # X_train = dataset['X_train']
    # y_train = dataset['y_train']
    # X_val = dataset['X_val']
    # y_val = dataset['y_val']
    # X_test = dataset['X_test']
    # y_test = dataset['y_test']


    for i in range(50):
        print("hi ", datasetOutput[i].shape)
        img = Image.fromarray(datasetOutput[i].astype(np.uint8), 'RGB')
        img.save('/home/ziad/Documents/image colorization/data_set/y_train/' + str(i) + 'my.png')

    print (model1.output)
    model1.summary()
    sgd = optimizers.SGD(lr=0.001, decay=1e-3, momentum=0.9, nesterov=True)
    adam = optimizers.adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.09, amsgrad=False)
    model1.compile(loss='mean_absolute_error', optimizer=adam)
    hist = model1.fit(datasetInput[0:40], datasetOutput[0:40], batch_size=2, epochs=10, shuffle=True)
    # model1.predict(X_test)

    score = model1.evaluate(datasetInput[0:50], datasetOutput[0:50], batch_size=1)
    print(score)
    print(hist)
    # hist = model.fit(X_train ,y_train ,validation_split=0.2 )
    arr = model1.predict(datasetInput[0:50])

    for i in range(50):
        img = Image.fromarray(arr[i].astype(np.uint8), 'RGB')
        img.save('/home/ziad/Documents/image colorization/data_set/' + str(i) + 'my.png')

    model1.save_weights('/home/ziad/Documents/image colorization/weights.h5')


run_model()
