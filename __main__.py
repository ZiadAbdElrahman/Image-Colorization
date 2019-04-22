from image import image
from model import model

path = "ALLdata"
# image = image(width , hight, file ,mode: "trainning" / "predecting", file to read the row image from )
image = image(224, 224, path, "predecting", "hi")
# image.Dowenloading_Data("hi",10,"http://farm4.static.flickr.com/3175/2737866473_7958dc8760.jpg")


# image.resizing()


model = model("training",image)

# model.train( Batch Size,epoch, learning rate )
model.train(2, 25, 1e-2)

# model.my_model.load_weights('best.h5')
# model.predict("test/")
