import math
from param import *


def combine_images(generated_images):

    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num) / width))
    shape = generated_images.shape[1:3]
    image = np.zeros((height * shape[0], width * shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index / width)
        j = index % width
        image[i * shape[0]:(i + 1) * shape[0],
              j * shape[1]:(j + 1) * shape[1]] = img[:, :, 0]
    return image


def load_mnist():

    # Load dataset and roughly rescale to [-1, 1]
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = (x_train.astype(np.float32) - SCALE) / SCALE
    # Add channel axis
    x_train = x_train[:, :, :, np.newaxis]
    x_test = x_test[:, :, :, np.newaxis]

    return x_train, y_train, x_test, y_test
