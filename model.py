import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
session = tf.Session(config=config)
from keras.models import model_from_yaml
import matplotlib.pyplot as plt
from image import image
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, Conv2DTranspose
from keras import optimizers
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation
from keras.regularizers import l2

class model :

    def __init__(self, image, pathFormodel= None, pathForwights = None):

        self.my_model = None
        self.mode = image.mode
        self.dataset = image.get_Data()
        if(self.mode == "training") :
            self.X_train = self.dataset['X_train']
            self.y_train = self.dataset['y_train']
            self.X_test = self.dataset['X_test']
            self.y_test = self.dataset['y_test']
        else :
            self.input = self.dataset
        # if(not pathFormodel == None):


    def moodel(self):
        reg = 9e-7
        model = Sequential()

        model.add(Convolution2D(32, (5, 5), padding='same', strides=(2, 2), input_shape=(224, 224, 1),activity_regularizer=l2(reg)))
        model.add(BatchNormalization(axis=-1))
        model.add(Activation('relu'))

        model.add(Convolution2D(64, (5, 5), padding='same', strides=(2, 2),activity_regularizer=l2(reg)))
        model.add(BatchNormalization(axis=-1))
        model.add(Activation('relu'))

        model.add(Convolution2D(128, (5, 5), padding='same', strides=(2, 2),activity_regularizer=l2(reg)))
        model.add(BatchNormalization(axis=-1))
        model.add(Activation('relu'))

        model.add(Convolution2D(256, (5, 5), padding='same', strides=(2, 2),activity_regularizer=l2(reg)))
        model.add(BatchNormalization(axis=-1))
        model.add(Activation('relu'))

        model.add(Convolution2D(512, (5, 5), padding='same', strides=(2, 2),activity_regularizer=l2(reg)))
        model.add(BatchNormalization(axis=-1))
        model.add(Activation('relu'))


        model.add(Conv2DTranspose(512, (5, 5), padding='same', strides=(2, 2),activity_regularizer=l2(reg)))
        model.add(BatchNormalization(axis=-1))
        model.add(Activation('relu'))

        model.add(Conv2DTranspose(256, (5, 5), padding='same', strides=(2, 2),activity_regularizer=l2(reg)))
        model.add(BatchNormalization(axis=-1))
        model.add(Activation('relu'))

        model.add(Conv2DTranspose(128, (5, 5), padding='same', strides=(2, 2),activity_regularizer=l2(reg)))
        model.add(BatchNormalization(axis=-1))
        model.add(Activation('relu'))

        model.add(Conv2DTranspose(64, (5, 5), padding='same', strides=(2, 2),activity_regularizer=l2(reg)))

        model.add(Conv2DTranspose(2, (5, 5), padding='same', strides=(2, 2),activity_regularizer=l2(reg)))

        return model


    def train(self,batch_size=16, epochs=35, learning_rate=6e-5):

        self.my_model = self.moodel()
        self.my_model.summary()

        # lr = 6e-5)
        self.adam = optimizers.adam(lr=learning_rate)
        self.my_model.compile(loss='mean_squared_error', optimizer=self.adam)

        history = self.my_model.fit(self.X_train, self.y_train, batch_size=batch_size, epochs=epochs , shuffle=True,
                                    validation_data=(self.X_test,self.y_test))


        self.plot_history(history)



        self.my_model.save_weights('weights.h5')

        model_yaml = self.my_model.to_yaml()
        with open("model.yaml", "w") as yaml_file:
            yaml_file.write(model_yaml)


    def predict(self,SavePath):
        # path to save the predicted image in

        if(self.mode == "training") :
            predicted = self.my_model.predict(self.X_test,batch_size=16)
            image.Save_Image(predicted , self.X_test, SavePath)
        else:
            predicted = self.my_model.predict(self.input, batch_size=16)
            image.Save_Image(predicted, self.input, SavePath)




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

        plt.show()



