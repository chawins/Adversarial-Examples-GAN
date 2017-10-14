from keras.layers import (Conv2D, Dense, MaxPooling2D, Reshape, UpSampling2D,
                          Input, Embedding, Multiply, LeakyReLU, Dropout,
                          Activation, Flatten)
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

    latent = Input(shape=(LATENT_SIZE, ))
    image_class = Input(shape=(1, ), dtype='int32')
    embed = Embedding(
        10, LATENT_SIZE, embeddings_initializer='glorot_normal')(image_class)
    flatten = Flatten()(embed)
    # hadamard product between z-space and a class conditional embedding
    mul = Multiply()([latent, flatten])

    model = Sequential()

    model.add(Dense(1024, input_dim=LATENT_SIZE, activation='relu'))
    model.add(Dense(128 * 7 * 7, activation='relu'))
    model.add(Reshape((7, 7, 128)))

    # upsample to (..., 14, 14)
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(256, (5, 5), padding='same', activation='relu',
                     kernel_initializer='glorot_normal'))

    # upsample to (..., 28, 28)
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(128, (5, 5), padding='same', activation='relu',
                     kernel_initializer='glorot_normal'))

    # take a channel axis reduction
    model.add(Conv2D(1, (2, 2), padding='same', activation='tanh',
                     kernel_initializer='glorot_normal'))

    fake_image = model(mul)

    return Model(inputs=[latent, image_class], outputs=fake_image)


# --------------------------- Discriminator models --------------------------- #


def dcgan_mnist_model_d():

    model = Sequential()
    model.add(Conv2D(64, (5, 5), padding='same', input_shape=(28, 28, 1)))
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

    image = Input(shape=(28, 28, 1))

    model = Conv2D(32, (3, 3), strides=(2, 2), padding='same',
                   input_shape=(28, 28, 1))(image)
    model = LeakyReLU()(model)
    model = Dropout(0.3)(model)

    model = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(model)
    model = LeakyReLU()(model)
    model = Dropout(0.3)(model)

    model = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(model)
    model = LeakyReLU()(model)
    model = Dropout(0.3)(model)

    model = Conv2D(256, (3, 3), strides=(1, 1), padding='same')(model)
    model = LeakyReLU()(model)
    model = Dropout(0.3)(model)

    model = Flatten()(model)

    # first output (name=generation) is whether or not the discriminator
    # thinks the image that is being shown is fake, and the second output
    # (name=auxiliary) is the class that the discriminator thinks the image
    # belongs to.
    fake = Dense(1, activation='sigmoid', name='generation')(model)
    aux = Dense(10, activation='softmax', name='auxiliary')(model)

    return Model(inputs=image, outputs=[fake, aux])

# def acgan_mnist_model_d():

#     # build a relatively standard conv net, with LeakyReLUs as suggested in
#     # the reference paper
#     model = Sequential()

#     model.add(Conv2D(32, (3, 3), stride=(2, 2), padding='same',
#                      input_shape=(28, 28, 1)))
#     model.add(LeakyReLU())
#     model.add(Dropout(0.3))

#     model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same'))
#     model.add(LeakyReLU())
#     model.add(Dropout(0.3))

#     model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
#     model.add(LeakyReLU())
#     model.add(Dropout(0.3))

#     model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same'))
#     model.add(LeakyReLU())
#     model.add(Dropout(0.3))

#     model.add(Flatten())

#     image = Input(shape=(28, 28, 1))

#     features = model(image)

#     # first output (name=generation) is whether or not the discriminator
#     # thinks the image that is being shown is fake, and the second output
#     # (name=auxiliary) is the class that the discriminator thinks the image
#     # belongs to.
#     fake = Dense(1, activation='sigmoid', name='generation')(features)
#     aux = Dense(10, activation='softmax', name='auxiliary')(features)

#     return Model(input=image, output=[fake, aux])


# ------------------------------- Model Utils -------------------------------- #


def generator_containing_discriminator(g, d):

    model = Sequential()
    model.add(g)
    d.trainable = False
    model.add(d)

    return model


def combine_acgan(g, d):

    latent = Input(shape=(LATENT_SIZE, ))
    image_class = Input(shape=(1, ), dtype='int32')
    fake = g([latent, image_class])

    # we only want to be able to train generation for the combined model
    d.trainable = False
    fake, aux = d(fake)
    return Model(inputs=[latent, image_class], outputs=[fake, aux])
