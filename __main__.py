from image import image
from model import model

path = "facesOnly"
image = image(224, 224, path, "training", "try")

# image.resizing()


# image.Dowenloading_Data("mountains wallpaper", 2000)
#

model = model(image)
model.train(4,25,2e-4)
model.my_model.load_weights('best.h5')
model.predict("test/")
