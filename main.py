from image import image
from model import model

path = "ALLdata"
image = image(224, 224, path, "training", None)



model = model(image)
model.train()

