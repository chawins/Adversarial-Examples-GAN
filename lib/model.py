from keras.layers import Conv2D, Dense, MaxPooling2D, Reshape, UpSampling2D, Input, Embedding, merge, LeakyReLU, Dropout
from keras.layers.core import Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, Model
from param import *


# ----------------------------- Generator models ----------------------------- #


def dcgan_mnist_model_g():

    model = Sequential()
    model.add(Dense(input_dim=100, output_dim=1024))
    model.add(Activation('tanh'))
    model.add(Dense(128 * 7 * 7))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Reshape((7, 7, 128), input_shape=(128 * 7 * 7,)))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(64, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(1, (5, 5), padding='same'))
    model.add(Activation('tanh'))

    return model


def acgan_mnist_model_g():

    model = Sequential()

    model.add(Dense(1024, input_dim=LATENT_SIZE, activation='relu'))
    model.add(Dense(128 * 7 * 7, activation='relu'))
    model.add(Reshape((128, 7, 7)))

    # upsample to (..., 14, 14)
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(256, 5, 5, border_mode='same',
                     activation='relu', init='glorot_normal'))

    # upsample to (..., 28, 28)
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(128, 5, 5, border_mode='same',
                     activation='relu', init='glorot_normal'))

    # take a channel axis reduction
    model.add(Conv2D(1, 2, 2, border_mode='same',
                     activation='tanh', init='glorot_normal'))

    # this is the z space commonly refered to in GAN papers
    latent = Input(shape=(LATENT_SIZE, ))

    # this will be our label
    image_class = Input(shape=(1, ), dtype='int32')

    # 10 classes in MNIST
    cls = Flatten()(Embedding(10, LATENT_SIZE,
                              init='glorot_normal')(image_class))

    # hadamard product between z-space and a class conditional embedding
    h = merge([latent, cls], mode='mul')

    fake_image = model(h)

    return Model(input=[latent, image_class], output=fake_image)


# --------------------------- Discriminator models --------------------------- #


def dcgan_mnist_model_d():

    model = Sequential()
    model.add(
        Conv2D(64, (5, 5),
               padding='same',
               input_shape=(28, 28, 1))
    )
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (5, 5)))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('tanh'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model


def acgan_mnist_model_d():

    # build a relatively standard conv net, with LeakyReLUs as suggested in
    # the reference paper
    model = Sequential()

    model.add(Conv2D(32, 3, 3, border_mode='same', subsample=(2, 2),
                     input_shape=(1, 28, 28)))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    model.add(Conv2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    model.add(Conv2D(128, 3, 3, border_mode='same', subsample=(2, 2)))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    model.add(Conv2D(256, 3, 3, border_mode='same', subsample=(1, 1)))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    model.add(Flatten())

    image = Input(shape=(1, 28, 28))

    features = model(image)

    # first output (name=generation) is whether or not the discriminator
    # thinks the image that is being shown is fake, and the second output
    # (name=auxiliary) is the class that the discriminator thinks the image
    # belongs to.
    fake = Dense(1, activation='sigmoid', name='generation')(features)
    aux = Dense(10, activation='softmax', name='auxiliary')(features)

    return Model(input=image, output=[fake, aux])


# ------------------------------- Model Utils -------------------------------- #


def generator_containing_discriminator(g, d):

    model = Sequential()
    model.add(g)
    d.trainable = False
    model.add(d)

    return model
