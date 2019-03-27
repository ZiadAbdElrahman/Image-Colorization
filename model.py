import tensorflow as tf
import matplotlib.pyplot as plt

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
session = tf.Session(config=config)
from data import get_Data ,Dowenloading_Data
from PIL import Image
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, Conv2DTranspose
from keras import optimizers
from keras.layers.normalization import BatchNormalization
import numpy as np
from keras.layers import Activation
from skimage import color
from keras.regularizers import l2

def my_moodel(weights_path=None):
    model = Sequential()
    model.add(Convolution2D(32, (5, 5), padding='same', strides=(2, 2), input_shape=(224, 224, 1),activity_regularizer=l2(9e-7)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))

    model.add(Convolution2D(64, (5, 5), padding='same', strides=(2, 2),activity_regularizer=l2(1e-7)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))

    model.add(Convolution2D(128, (5, 5), padding='same', strides=(2, 2),activity_regularizer=l2(1e-7)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))

    model.add(Convolution2D(256, (5, 5), padding='same', strides=(2, 2),activity_regularizer=l2(1e-7)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))

    model.add(Convolution2D(512, (5, 5), padding='same', strides=(2, 2),activity_regularizer=l2(1e-7)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))

    # model.add(Convolution2D(256, (3, 3), padding='same', strides=(2, 2)))
    # # model.add(BatchNormalization(axis=-1))
    # model.add(Activation('relu'))

    model.add(Conv2DTranspose(512, (5, 5), padding='same', strides=(2, 2),activity_regularizer=l2(1e-7)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))

    model.add(Conv2DTranspose(256, (5, 5), padding='same', strides=(2, 2),activity_regularizer=l2(1e-7)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))

    model.add(Conv2DTranspose(128, (5, 5), padding='same', strides=(2, 2),activity_regularizer=l2(1e-7)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('relu'))

    model.add(Conv2DTranspose(64, (5, 5), padding='same', strides=(2, 2),activity_regularizer=l2(1e-7)))
    # model.add(BatchNormalization(axis=-1))
    # model.add(Activation('relu'))

    model.add(Conv2DTranspose(2, (5, 5), padding='same', strides=(2, 2),activity_regularizer=l2(1e-7)))
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
    dataset = get_Data(4000,600,0,norm=False)
    model1 = my_moodel()

    # datasetInput, datasetOutput = Read_Data()
    #
    X_train = dataset['X_train']
    y_train = dataset['y_train']
    X_val = dataset['X_val']
    y_val = dataset['y_val']
    X_test = dataset['X_test']
    y_test = dataset['y_test']



    print (model1.output)
    model1.summary()
    sgd = optimizers.SGD(lr=0.001, decay=1e-3, momentum=0.9, nesterov=True)
    adam = optimizers.adam(lr=1e-4)

    model1.compile(loss='mean_squared_error', optimizer=adam)

    history = model1.fit(X_train[0:3500], y_train[0:3500], batch_size=16, epochs=25 , shuffle=True,validation_data=(X_val,y_val))

    plot_history(history)

    # model1.predict(X_test)

    # score = model1.evaluate(datasetInput[0:50], datasetOutput[0:50], batch_size=1)
    # print(score)
    # print(hist)
    # hist = model.fit(X_train ,y_train ,validation_split=0.2 )

    arrTrain = model1.predict(X_train[0:500], batch_size=4)
    arrVal = model1.predict(X_val[0:500], batch_size=4)
    # arrVal = model1.predict(datasetInput[2000:2100], batch_size=4)

    # print(arr.shape)
    # arrTrain *= 128
    # arrVal *= 128


    for i in range(len(arrTrain)):
        cur = np.zeros((224, 224, 3))
        Xre = X_train[i].reshape(224, 224)
        cur[:, :, 0] = Xre
        cur[:, :, 1:] = arrTrain[i]
        end = color.lab2rgb(cur)
        end *= 255

        img = Image.fromarray(end.astype(np.uint8), 'RGB')
        # imsave('/home/ziad/Documents/image colorization/train/' + str(i) + 'my.png', img)
        img.save('/home/ziad/Documents/image colorization/train/' + str(i) + 'my.png')

    for i in range(len(arrTrain)):
        cur = np.zeros((224, 224, 3))
        Xre = X_val[i].reshape(224, 224)
        cur[:, :, 0] = Xre
        cur[:, :, 1:] = arrVal[i]
        end = color.lab2rgb(cur)
        end *= 255

        img = Image.fromarray(end.astype(np.uint8), 'RGB')
        # imsave('/home/ziad/Documents/image colorization/train/' + str(i) + 'my.png', img)
        img.save('/home/ziad/Documents/image colorization/test/' + str(i) + 'my.png')


    model1.save_weights('/home/ziad/Documents/image colorization/weights.h5')



# run_model()

Dowenloading_Data("dogs",2000)
Dowenloading_Data("cats",2000)
Dowenloading_Data("Mountains",2000)
Dowenloading_Data("beach",1000)
Dowenloading_Data("Park",2000)
Dowenloading_Data("Nature",2000)
Dowenloading_Data("town",2000)
Dowenloading_Data("houses",2000)
Dowenloading_Data("street",2000)