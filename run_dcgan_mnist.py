from lib.model import *
from lib.utils import *
from param import *

# Set CUDA visible device to GPU:0
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def train(prog=True):

    # Load MNIST
    x_train, _, _, _ = load_mnist()

    # Build model
    d = dcgan_mnist_model_d()
    g = dcgan_mnist_model_g()

    # Set optimizer
    d_opt = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    g_opt = SGD(lr=0.0005, momentum=0.9, nesterov=True)

    # Set loss function and compile models
    g.compile(loss='binary_crossentropy', optimizer="SGD")
    d.compile(loss='binary_crossentropy', optimizer=d_opt)
    combined = generator_containing_discriminator(g, d)
    combined.compile(loss='binary_crossentropy', optimizer=g_opt)

    n_batch = int(x_train.shape[0] / BATCH_SIZE)
    for epoch in range(N_EPOCH):
        print("Epoch {} of {}".format(epoch + 1, N_EPOCH))
        for index in range(n_batch):

            # ---------------- Train discriminator --------------------------- #
            # Generate samples from g
            z = np.random.uniform(-1, 1, (BATCH_SIZE, 100))
            x_g = g.predict(z, verbose=0)
            # Combine with real samples
            x_real = x_train[index * BATCH_SIZE:(index + 1) * BATCH_SIZE]
            x_d = np.concatenate((x_real, x_g))
            y_d = [1] * BATCH_SIZE + [0] * BATCH_SIZE
            d_loss = d.train_on_batch(x_d, y_d)

            # ---------------- Train generator ------------------------------- #
            z = np.random.uniform(-1, 1, (BATCH_SIZE, 100))
            d.trainable = False
            g_loss = combined.train_on_batch(z, [1] * BATCH_SIZE)
            d.trainable = True

            # Save weights
            if index % 10 == 9:
                g.save_weights(WEIGHT_DIR + 'dcgan_mnist_g.h5', True)
                d.save_weights(WEIGHT_DIR + 'dcgan_mnist_d.h5', True)

            if prog:
                if index % 20 == 0:
                    # Save generated samples
                    image = combine_images(x_g)
                    image = image * SCALE + SCALE
                    Image.fromarray(image.astype(np.uint8)).save(
                        VIS_DIR + str(epoch) + "_" + str(index) + ".png")
                    # Print losses
                    print("batch %d d_loss : %f     batch %d g_loss : %f" %
                          (index, d_loss, index, g_loss))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-prog", dest="prog", action="store_false")
    parser.set_defaults(prog=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    train(prog=args.prog)
