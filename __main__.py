from image import image
from model import model
#
# # path = "facesOnlypart"
# # path = "facesOnly"
# # path = "me"
# # path = "ALLdata"
path = "people"
# # path = "temp2"
# # path = "images"
#
# image = image(width , hight, file ,mode: "trainning" / "predecting", file to read the row image from )
image = image(224, 224, "training", path)
# # image.Dowenloading_Data("hi",10,"http://farm4.static.flickr.com/3175/2737866473_7958dc8760.jpg")
#
#
# # image.resizing()
#
#
moodel = model(image, path, None, "nice++/weights.h5")
#
# # model.train( Batch Size,epoch, learning rate )
# # model.train(4, 150, 5e-4)
#
# # model.my_model.load_weights('best.h5')
# model.predict(model.X_test, "test/")
from keras.preprocessing.image import img_to_array, load_img
from tkinter import *
from tkinter import font
import numpy as np
from PIL import Image
from skimage.color import rgb2lab
import random
from model import model
import matplotlib.pyplot as plt
import tkFileDialog
from skimage import color

welcomeWindow = Tk()
mainWindow = Tk()
mainWindow.withdraw()
IMGfile = []

def browse_file():

    file = tkFileDialog.askopenfilename(filetypes = (("Template files", "*.type"), ("All files", "*")))
    print (file)
    IMGfile.append(file)

    return file

    root = Tk.Tk()
    root.wm_title("Browser")
    broButton = Tk.Button(master = root, text = 'Browse', width = 6, command=browse_file)
    broButton.pack(side=Tk.LEFT, padx = 2, pady=2)

    Tk.mainloop()



def show_loss():
    img = Image.open('nice++/LOSS.png')
    img.show()

def predict_from_test():

    temp = random.randint(0, moodel.X_test.shape[0])
    img = (moodel.X_test[temp]).reshape(1, 224, 224, 1)



    img = (moodel.X_test[temp]).reshape(224, 224)
    plt.subplot(111)
    plt.imshow(img, cmap=plt.get_cmap('gray'))
    plt.show()
    y = moodel.predict(img, str(temp))


def browse():
    file = browse_file()
    IMGfile.append(file)

def predict():

    if IMGfile[0] != None:
        img = load_img(IMGfile[0])
        plt.subplot(111)
        plt.imshow(img, cmap=plt.get_cmap('gray'))
        plt.show()
        img = np.array(img, dtype=float)
        L = rgb2lab(1.0 / 255 * img)[:, :, 0]
        L = L.reshape(1, 224, 224, 1)
        moodel.predict(L, "")
        IMGfile.pop()

def back_main():
    mainWindow.withdraw()
    welcomeWindow.deiconify()


def openMainWindow():
    welcomeWindow.withdraw()
    mainWindow.deiconify()


# /---------------------------------------------------------------------------------------------------------------------/

welcomeWindow.geometry("500x500")

labelFont = font.Font(family='Helvetica', size=30, weight='bold', slant='italic')
welcome = Label(welcomeWindow, text="Welcome To\nImage Colorization", font=labelFont, fg="#7da1c1").place(x=40, y=110)

mainWindowButton = Button(welcomeWindow, text="Start", command=openMainWindow, width=35, height=2, bg="#232323",
                          fg="white").place(x=95, y=355)

# /---------------------------------------------------------------------------------------------------------------------/

buttonFont = font.Font(family='Helvetica', size=40, weight='bold', slant='italic')

mainWindow.geometry("500x500")
predictButton = Button(mainWindow, text="Predict", command=predict, bg="#7da1c1", fg="black", font=buttonFont)
predictButton.place(bordermode=OUTSIDE, height=225, width=250, x=0, y=0)

testButton = Button(mainWindow, text="Test", command=predict_from_test, bg="#eaea60", fg="black", font=buttonFont)
testButton.place(bordermode=OUTSIDE, height=225, width=250, x=250, y=0)

showLoss = Button(mainWindow, text="Loss", command=show_loss, bg="#ff5b5b", fg="black", font=buttonFont)
showLoss.place(bordermode=OUTSIDE, height=225, width=250, x=0, y=225)

showAccuracy = Button(mainWindow, text="browse", command=browse, bg="#b5ff5b", fg="black", font=buttonFont)
showAccuracy.place(bordermode=OUTSIDE, height=225, width=250, x=250, y=225)

backButton = Button(mainWindow, text="BACK", command=back_main, bg="#232323", fg="white", font=buttonFont)
backButton.place(bordermode=OUTSIDE, height=50, width=500, x=0, y=450)

# /---------------------------------------------------------------------------------------------------------------------/

welcomeWindow.mainloop()