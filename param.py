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

SCALE = 127.5       # Scaling factor

LATENT_SIZE = 100
BATCH_SIZE = 128
N_EPOCH = 50
VIS_DIR = "./vis/"
WEIGHT_DIR = "./weights/"
