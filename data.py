from google_images_download import google_images_download
from resizeimage import resizeimage
import os
from PIL import Image
import numpy as np
from keras.preprocessing.image import img_to_array, load_img
from skimage.color import rgb2lab
from skimage import io, color

# Dowenloding the images from google
def Dowenloading_Data(kewword,NumOfphoto) :
    response = google_images_download.googleimagesdownload()
    absolute_image_paths = response.download({"keywords":kewword , "limit" : NumOfphoto , "color_type" : "full-color" , "aspect_ratio": "square" ,"chromedriver":"/home/ziad/Downloads/chromedriver"})


# converting all image to the same size
def resizeing(path,newepath,newWidth,newHight) :
    faild = 0
    for ind,image_path in enumerate(os.listdir(path)):
        input_path = os.path.join(path, image_path)
        try:
            with open(input_path, 'r+b') as f :
                with Image.open(f) as image:
                    cover = resizeimage.resize_cover(image, [newWidth, newHight])
                    newName = newepath + "/" + str(ind)+'.jpg'
                    cover.save( newName, image.format)
        except :
            faild+=1
    print(faild)




def Read_Data() :

    # Load the Dataset from the file data and return it .

    folder = 'Dataset'

    images = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    print("Files in train_files: %d" % len(images))

    # Dimensions
    image_width = 224
    image_height = 224


    # dataset input is the black and white image, sot it's just one channel "L"
    datasetInput = np.ndarray(shape=(len(images), image_height, image_width, 1),dtype=np.float32)

    # dataset output is other two channel "a and b"
    datasetOutput = np.ndarray(shape=(len(images), image_height, image_width, 2), dtype=np.float32)

    # datasetInput = []
    # datasetOutput = []
    i = 0
    for _file in images:
        # loading the image and convert it into array
        img = load_img(folder + "/" + _file)
        img = np.array(img,dtype=float)


        # converting int lab image
        L = rgb2lab(1.0/255*img)[:, :, 0]
        A_B = rgb2lab(1.0/255*img)[:, :, 1:]


        # Normlizing
        # A_B /= 128


        datasetInput[i, ..., 0] = L
        datasetOutput[i] = A_B

        i += 1

    print("All images to array!")

    return datasetInput, datasetOutput


def get_Data(num_training=3500, num_validation=500, num_test=500 ,norm = True):

    # Spilting the dato into three categry (training, validation, testing) .

    # read the data from the file and convert it into array
    datasetInput, datasetOutput = Read_Data()


    TrainingMask = list(range(0, num_training))
    ValidationMask = list(range(num_training , num_training+num_validation))
    TestMask = list(range( num_training+num_validation ,  num_training + num_validation + num_test))



    X_train = datasetInput[TrainingMask]
    y_train = datasetOutput[TrainingMask]

    X_val = datasetInput[ValidationMask]
    y_val = datasetOutput[ValidationMask]

    X_test = datasetInput[TestMask]
    y_test = datasetOutput[TestMask]


    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test
    }
