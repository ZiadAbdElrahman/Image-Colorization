from abc import abstractmethod
import os
import numpy as np
from tempfile import TemporaryFile


class Data:
    def __init__(self, width, hight, mode):
        self.width = width
        self.hight = hight
        self.mode = mode

    @abstractmethod
    def Read_Data(self):
        return 0

    def get_Data(self, arrayName=None):

        # Spilting the dato into three categry (training, testing) .


        # read the data from the file and convert it into array
        if self.mode == "predecting":
            datasetInput, datasetOutput = self.Read_Data()

        else:
            try:
                # datasetInput = np.load("data/" + arrayName + 'I' + '.npy')
                print ("start1")
                X_train = np.load("data/" + arrayName + 'XT' + '.npy')
                X_test = np.load("data/" + arrayName + 'XS' + '.npy')

                y_train = np.load("data/" + arrayName + 'yT' + '.npy')
                y_test = np.load("data/" + arrayName + 'yS' + '.npy')
                print ("end1")

            except:
                try:
                    datasetInput = np.load("data/" + arrayName + 'I' + '.npy')

                except:
                    datasetInput, datasetOutput = self.Read_Data()
                    np.save("data/" + arrayName + 'I' + '.npy', datasetInput)
                    np.save("data/" + arrayName + 'O' + '.npy', datasetOutput)

                TrainingMask = list(range(0, int(len(datasetInput) * 0.8)))

                TestMask = list(range(len(TrainingMask), int(len(datasetInput))))

                X_train = datasetInput[TrainingMask]
                np.save("data/" + arrayName + 'XT' + '.npy', X_train)

                X_test = datasetInput[TestMask]
                np.save("data/" + arrayName + 'XS' + '.npy', X_test)



                datasetOutput = np.load("data/" + arrayName + 'O' + '.npy')




                y_train = datasetOutput[TrainingMask]
                np.save("data/" + arrayName + 'yT' + '.npy', y_train)


                y_test = datasetOutput[TestMask]
                np.save("data/" + arrayName + 'yS' + '.npy', y_test)



        if (self.mode == "predecting"):
            return datasetInput
        else:
            return {
                'X_train': X_train, 'y_train': y_train,
                'X_test': X_test, 'y_test': y_test
            }
