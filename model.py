import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
session = tf.Session(config=config)

import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential, model_from_yaml
from keras.layers.convolutional import Convolution2D, Conv2DTranspose
from keras import optimizers
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation, Dropout
from keras.regularizers import l2
from keras.utils import plot_model


class model:

    def __init__(self, image=None, arrayName=None, pathForModel=None, pathForwights=None):

        self.image = image
        self.dataset = image.get_Data(arrayName)

        self.mode = image.mode

        self.adam = None

        if pathForModel != None:
            yaml_file = open(pathForModel, 'r')
            loaded_model_yaml = yaml_file.read()
            yaml_file.close()
            self.my_model = model_from_yaml(loaded_model_yaml)
            self.my_model.summary()
        else:
            self.my_model = self.moodel()

        if pathForwights != None:
            self.my_model.load_weights(pathForwights)

        # plot_model(self.my_model, to_file='model.png')

        if (self.mode == "training"):
            self.X_train = self.dataset['X_train']
            self.y_train = self.dataset['y_train']
            self.X_test = self.dataset['X_test']
            self.y_test = self.dataset['y_test']
        else:
            self.input = self.dataset

    def moodel(self):

        # reg = 1e-6
        # reg = 1e-8
        # reg = 8e-10
        reg = 0

        drp = 0

        model = Sequential()

        model.add(Convolution2D(32, (3, 3), padding='same', strides=(2, 2), input_shape=(224, 224, 1),
                                activity_regularizer=l2(reg)))
        model.add(Dropout(drp))
        model.add(BatchNormalization(axis=-1))
        model.add(Activation('relu'))

        model.add(Convolution2D(32, (3, 3), padding='same', strides=(2, 2), activity_regularizer=l2(reg)))
        model.add(Dropout(drp))
        model.add(BatchNormalization(axis=-1))
        model.add(Activation('relu'))

        model.add(Convolution2D(32, (3, 3), padding='same', strides=(2, 2), activity_regularizer=l2(reg)))
        model.add(Dropout(drp))
        model.add(BatchNormalization(axis=-1))
        model.add(Activation('relu'))

        model.add(Convolution2D(64, (3, 3), padding='same', strides=(1, 1), activity_regularizer=l2(reg)))
        model.add(Dropout(drp))
        model.add(BatchNormalization(axis=-1))
        model.add(Activation('relu'))

        model.add(Convolution2D(64, (3, 3), padding='same', strides=(1, 1), activity_regularizer=l2(reg)))
        model.add(Dropout(drp))
        model.add(BatchNormalization(axis=-1))
        model.add(Activation('relu'))

        model.add(Convolution2D(64, (3, 3), padding='same', strides=(1, 1), activity_regularizer=l2(reg)))
        model.add(Dropout(drp))
        model.add(BatchNormalization(axis=-1))
        model.add(Activation('relu'))

        model.add(Convolution2D(64, (3, 3), padding='same', strides=(1, 1), activity_regularizer=l2(reg)))
        model.add(Dropout(drp))
        model.add(BatchNormalization(axis=-1))
        model.add(Activation('relu'))

        model.add(Conv2DTranspose(32, (3, 3), padding='same', strides=(2, 2), activity_regularizer=l2(reg)))
        model.add(Dropout(drp))
        model.add(BatchNormalization(axis=-1))
        model.add(Activation('relu'))

        model.add(Conv2DTranspose(32, (3, 3), padding='same', strides=(2, 2), activity_regularizer=l2(reg)))
        model.add(Dropout(drp))
        model.add(BatchNormalization(axis=-1))
        model.add(Activation('relu'))

        model.add(Conv2DTranspose(32, (3, 3), padding='same', strides=(2, 2), activity_regularizer=l2(reg)))
        model.add(Dropout(drp))
        model.add(BatchNormalization(axis=-1))
        model.add(Activation('relu'))

        model.add(Conv2DTranspose(2, (3, 3), padding='same', strides=(1, 1), activity_regularizer=l2(reg)))

        return model

    def train(self, batch_size=4, epochs=30, learning_rate=1e-4):

        # print the
        self.my_model.summary()

        # lr = 6e-5)
        self.adam = optimizers.adam(lr=learning_rate)
        self.my_model.compile(loss='mean_squared_error', optimizer=self.adam)

        history = self.my_model.fit(self.X_train, self.y_train, batch_size=batch_size, epochs=epochs, shuffle=True,
                                    validation_data=(self.X_test, self.y_test))

        self.plot_history(history)

        self.my_model.save_weights('weights.h5')

        model_yaml = self.my_model.to_yaml()
        with open("model.yaml", "w") as yaml_file:
            yaml_file.write(model_yaml)

    def predict(self, X, SavePath):
        # path to save the predicted image in

        predicted = self.my_model.predict(X, batch_size=16)

        self.image.Save_Image(predicted, X, SavePath)

    def plot_history(self, history):
        loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
        val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]

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
        plt.savefig("LOSS")
        plt.show()
