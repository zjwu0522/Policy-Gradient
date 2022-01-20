import numpy as np

def preprocess(image, constant):
    image = image[34:194, :, :] # 160, 160, 3
    image = np.mean(image, axis=2, keepdims=False) # 160, 160
    image = image[::2, ::2] # 80, 80
    image = image/256
    image = image - constant/256# remove background
    return image