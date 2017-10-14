import argparse
import os

import numpy as np
from keras.datasets import mnist
from keras.optimizers import SGD
from PIL import Image

# Define global parameters, treated like config file

SCALE = 127.5       # Scaling factor

LATENT_SIZE = 100
BATCH_SIZE = 128
N_EPOCH = 50
IMG_DIR = "./vis/"
