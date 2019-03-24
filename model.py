import keras
import tensorflow as tf
import matplotlib.pyplot as plt

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
from skimage import io, color
from  scipy.misc import imsave
from keras import regularizers
# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.3
# set_session(tf.Session(config=config))

def my_moodel(weights_path=None):

    model = Sequential()
    model.add(Convolution2D(8, (3, 3 ),padding='same',strides=(2,2), input_shape=(224, 224, 1)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))


    model.add(Convolution2D(16, (3, 3), padding='same',strides=(2,2)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))

    model.add(Convolution2D(32, (3, 3), padding='same',strides=(2,2)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))

    model.add(Convolution2D(64, (3, 3), padding='same',strides=(2,2)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))

    model.add(Convolution2D(128, (3, 3), padding='same',strides=(2,2)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))

    # model.add(Convolution2D(256, (3, 3), padding='same', strides=(2, 2)))
    # # model.add(BatchNormalization(axis=-1))
    # model.add(Activation('relu'))

    model.add(Conv2DTranspose(128, (3, 3), padding='same', strides=(2, 2)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))

    model.add(Conv2DTranspose(64, (3, 3), padding='same',strides=(2,2)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))

    model.add(Conv2DTranspose(32, (3, 3), padding='same',strides=(2,2)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))


    model.add(Conv2DTranspose(16, (3, 3), padding='same',strides=(2,2)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))


    model.add(Conv2DTranspose(2, (3, 3), padding='same',strides=(2,2)))
    # model.add(BatchNormalization(axis=-1))



    #
    # model.add(Conv2DTranspose(2, (3, 3), padding='same',strides=(2,2)))

    # model.add(regularizers.l2(1e-3))
    if weights_path:
        model.load_weights(weights_path)

    return model


def plot_history(history):
    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
    acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]

    if len(loss_list) == 0:
        print('Loss is missing in history')
        return

        ## As loss always exists
    epochs = range(1, len(history.history[loss_list[0]]) + 1)

    ## Loss
    plt.figure(1)
    for l in loss_list:
        plt.plot(epochs, history.history[l], 'b',
                 label='Training loss (' + str(str(format(history.history[l][-1], '.5f')) + ')'))
    for l in val_loss_list:
        plt.plot(epochs, history.history[l], 'g',
                 label='Validation loss (' + str(str(format(history.history[l][-1], '.5f')) + ')'))

    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    ## Accuracy
    plt.figure(2)
    for l in acc_list:
        plt.plot(epochs, history.history[l], 'b',
                 label='Training accuracy (' + str(format(history.history[l][-1], '.5f')) + ')')
    for l in val_acc_list:
        plt.plot(epochs, history.history[l], 'g',
                 label='Validation accuracy (' + str(format(history.history[l][-1], '.5f')) + ')')

    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


def run_model():
    # get the data from the file
    dataset = get_Data(3500 , 0 , 1000,norm=False)
    model1 = my_moodel()
    # datasetInput, datasetOutput = Read_Data()



    X_train = dataset['X_train']
    y_train = dataset['y_train']
    X_val = dataset['X_val']
    y_val = dataset['y_val']
    X_test = dataset['X_test']
    y_test = dataset['y_test']

    # meany = dataset['meany']
    # meanX = dataset['meanX']
    # # print(meany.shape)


    print (model1.output)
    model1.summary()
    sgd = optimizers.SGD(lr=0.001, decay=1e-3, momentum=0.9, nesterov=True)
    adam = optimizers.adam(lr=1e-5)

    model1.compile(loss='mean_squared_error', optimizer=adam)



    history = model1.fit(X_train[0:500], y_train[0:500], batch_size=4, epochs=750, shuffle=False)

    plot_history(history)

    # model1.predict(X_test)

    #score = model1.evaluate(datasetInput[0:50], datasetOutput[0:50], batch_size=1)
   # print(score)
    #print(hist)
    # hist = model.fit(X_train ,y_train ,validation_split=0.2 )

    arr = model1.predict(X_train[0:1000], batch_size=8)
    # print(arr.shape)
    arr *= 128
    # X_train += meanX

    for i in range(1000):

        cur = np.zeros((224, 224, 3))
        Xre = X_train[i].reshape(224, 224)
        cur[:, :, 0] = Xre
        cur[:, :, 1:] = arr[i]
        end = color.lab2rgb(cur)
        end *= 255


        img = Image.fromarray(end.astype(np.uint8),'RGB')
        # imsave('/home/ziad/Documents/image colorization/train/' + str(i) + 'my.png', img)
        img.save('/home/ziad/Documents/image colorization/train/' + str(i) + 'my.png')

    # for i in range(700 ,1000):
    #     cur = np.zeros((224, 224, 3))
    #     Xre = X_train[i].reshape(224, 224)
    #     cur[:, :, 0] = Xre
    #
    #     cur[:, :, 1:] = arr[i]
    #
    #     end = color.lab2rgb(cur)*255
    #
    #     img = Image.fromarray(end.astype(np.uint8), 'RGB')
    #     #     imsave('/home/ziad/Documents/image colorization/test/' + str(i) + 'my.png', end)
    #
    #     img.save('/home/ziad/Documents/image colorization/train/' + str(i) + 'my.png')
    #
    #     # end.save('/home/ziad/Documents/image colorization/test/' + str(i) + 'my.png')

    model1.save_weights('/home/ziad/Documents/image colorization/weights.h5')


run_model()
