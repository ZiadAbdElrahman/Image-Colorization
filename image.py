from google_images_download import google_images_download
from resizeimage import resizeimage
import os
import numpy as np
from keras.preprocessing.image import img_to_array, load_img
from skimage.color import rgb2lab
from PIL import Image
from skimage import color

class image :


    def __init__(self, width, hight, resizingFolder, mode, DownloadingFolder=None):
        # Downloading folder the folder will downloading in, the resizing folder the folder  will used to save in after resizing

        self.width = width
        self.hight = hight
        self.DownFolder = DownloadingFolder
        self.ReSizingFolder = resizingFolder
        self.mode = mode

    # Dowenloding the images from google
    def Dowenloading_Data(self, kewword, NumOfphoto):
        response = google_images_download.googleimagesdownload()
        absolute_image_paths = response.download({"keywords": kewword,"output_directory": self.DownFolder, "limit": NumOfphoto, "color_type": "full-color",
                                                  "chromedriver": "/home/ziad/Downloads/chromedriver"})

    # converting all image to the same size
    def resizing(self):
        faild = 0
        for ind, image_path in enumerate(os.listdir(self.DownFolder)):
            input_path = os.path.join(self.DownFolder, image_path)
            try:
                with open(input_path, 'r+b') as f:
                    with Image.open(f) as image:
                        cover = resizeimage.resize_cover(image, [self.width, self.hight])
                        newName = self.ReSizingFolder + "/" + str(ind)  + '.jpg'
                        cover.save(newName, image.format)
            except:
                faild += 1
        print("sorry, faild to resize " + faild + " image")


    def Read_Data(self):

        # Load the Dataset from the file data and return it .

        folder = self.ReSizingFolder

        images = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
        print("Files in train_files: %d" % len(images))

        # Dimensions
        image_width = 224
        image_height = 224

        # dataset input is the black and white image, sot it's just one channel "L"
        datasetInput = np.ndarray(shape=(len(images), image_height, image_width, 1), dtype=np.float32)

        # dataset output is other two channel "a and b"
        datasetOutput = np.ndarray(shape=(len(images), image_height, image_width, 2), dtype=np.float32)

        i = 0
        faild = 0
        for _file in images:
            # loading the image and convert it into array
            img = load_img(folder + "/" + _file)
            img = np.array(img, dtype=float)

            try:
                if(self.mode == "training") :
                    # converting int lab image
                    L = rgb2lab(1.0 / 255 * img)[:, :, 0]
                    A_B = rgb2lab(1.0 / 255 * img)[:, :, 1:]

                    # Normlizing
                    # A_B /= 128

                    datasetInput[i, ..., 0] = L
                    datasetOutput[i] = A_B
                if(self.mode == "predecting") :
                    # converting int lab image
                    L = rgb2lab(1.0 / 255 * img)[:, :, 0]

                    datasetInput[i, ..., 0] = L

                i += 1
            except:
                faild += 1

            if (i % 1000 == 0):
                print (str(i) + " !!! ")

        print("All images to array!")
        print("sorry, faild to reed " + faild + " image")
        return datasetInput, datasetOutput

    def get_Data(self, num_training=4000, num_test=1845):

        # Spilting the dato into three categry (training, validation, testing) .

        # read the data from the file and convert it into array
        datasetInput, datasetOutput = self.Read_Data()

        TrainingMask = list(range(0, num_training))
        TestMask = list(range(num_training, num_training + num_test))

        X_train = datasetInput[TrainingMask]
        y_train = datasetOutput[TrainingMask]

        X_test = datasetInput[TestMask]
        y_test = datasetOutput[TestMask]

        if(self.mode == "predecting") :
            return datasetInput
        else :
            return {
                'X_train': X_train, 'y_train': y_train,
                'X_test': X_test, 'y_test': y_test
            }

    def Save_Image(self, predicted, L, path):
        for i in range(len(predicted)):
            cur = np.zeros((224, 224, 3))
            Xre = L[i].reshape(224, 224)
            cur[:, :, 0] = Xre
            cur[:, :, 1:] = predicted[i]
            end = color.lab2rgb(cur)
            end *= 255

            img = Image.fromarray(end.astype(np.uint8), 'RGB')
            img.save(path + str(i) + '.png')
