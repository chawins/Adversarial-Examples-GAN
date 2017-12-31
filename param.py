import argparse
import os
import pickle
from collections import defaultdict

import numpy as np
import keras
from keras.datasets import mnist
from keras.optimizers import SGD
from keras.utils.generic_utils import Progbar
from PIL import Image

# Define global parameters, treated like config file

VIS_DIR = "./vis/"
WEIGHT_DIR = "./weights/"

SCALE = 127.5               # Data scaling factor
LATENT_SIZE = 100           # Dimensions of latent variables
INPUT_SHAPE = (28, 28, 1)   # Shape of input data
N_CLASSES = 10              # Number of classes/labels

N_EPOCH = 50                # Max. epochs for training
BATCH_SIZE = 128            # Batch size
N_DIS = 5                   # Number of iterations of discriminator training
                            # per one iteration of generator training
