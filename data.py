from abc import  abstractmethod
import os


class Data :
    def __init__(self, width, hight, mode):
        self.width = width
        self.hight = hight
        self.mode = mode

    @abstractmethod
    def  Read_Data(self):
        return 0

    def get_Data(self):

        # Spilting the dato into three categry (training, testing) .

        # read the data from the file and convert it into array
        datasetInput, datasetOutput = self.Read_Data()


        TrainingMask = list(range(int(len(datasetInput)*0.7)))
        TestMask = list(range(int(len(datasetInput)*0.3)))



        X_train = datasetInput[TrainingMask]
        y_train = datasetOutput[TrainingMask]

        X_test = datasetInput[TestMask]
        y_test = datasetOutput[TestMask]

        if (self.mode == "predecting"):
            return datasetInput
        else:
            return {
                'X_train': X_train, 'y_train': y_train,
                'X_test': X_test, 'y_test': y_test
            }


