from keras.optimizers import Adam
from lib.model import *
from lib.utils import *
from param import *

# Set CUDA visible device to GPU:0
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Adam parameters suggested in https://arxiv.org/abs/1511.06434
adam_lr = 0.0002
adam_beta_1 = 0.5


def train(prog=True):
    """
    Code is based on https://github.com/lukedeo/keras-acgan
    """

    # Load MNIST
    x_train, y_train, x_test, y_test = load_mnist()

    # Build model
    d = dcgan_mnist_model_d()
    g = dcgan_mnist_model_g()

    # Set up optimizers
    adam = Adam(lr=adam_lr, beta_1=adam_beta_1)

    # Set loss function and compile models
    g.compile(optimizer=adam, loss='binary_crossentropy')
    d.compile(optimizer=adam, loss=['binary_crossentropy'])
    combined = generator_containing_discriminator(g, d)
    combined.compile(optimizer=adam, loss=['binary_crossentropy'])

    train_history = defaultdict(list)
    test_history = defaultdict(list)

    n_batch = int(x_train.shape[0] / BATCH_SIZE)
    for epoch in range(N_EPOCH):
        print('Epoch {} of {}'.format(epoch + 1, N_EPOCH))
        progress_bar = Progbar(target=n_batch)

        epoch_g_loss = []
        epoch_d_loss = []

        for index in range(n_batch):
            progress_bar.update(index, force=True)

            # ---------------- Train discriminator --------------------------- #
            # Generate samples from g
            z = np.random.uniform(-1, 1, (BATCH_SIZE, LATENT_SIZE))
            x_g = g.predict(z, verbose=0)

            # Combine with real samples
            x_real = x_train[index * BATCH_SIZE:(index + 1) * BATCH_SIZE]
            x_d = np.concatenate((x_real, x_g))
            y_d = np.array([1] * BATCH_SIZE + [0] * BATCH_SIZE)

            epoch_d_loss.append(d.train_on_batch(x_d, y_d))

            # ---------------- Train generator ------------------------------- #
            # Generate 2 * BATCH_SIZE samples to match d's batch size
            z = np.random.uniform(-1, 1, (2 * BATCH_SIZE, LATENT_SIZE))
            y_g = np.ones(2 * BATCH_SIZE)

            epoch_g_loss.append(combined.train_on_batch(z, y_g))

        print('\nTesting for epoch {}:'.format(epoch + 1))
        n_test = x_test.shape[0]

        # ---------------- Test discriminator -------------------------------- #
        z = np.random.uniform(-1, 1, (n_test, LATENT_SIZE))
        x_g = g.predict(z, verbose=0)

        x_d = np.concatenate((x_test, x_g))
        y_d = np.array([1] * n_test + [0] * n_test)

        d_test_loss = d.evaluate(x_d, y_d, verbose=0)
        d_train_loss = np.mean(np.array(epoch_d_loss), axis=0)

        # ---------------- Test generator ------------------------------------ #
        z = np.random.uniform(-1, 1, (2 * n_test, LATENT_SIZE))
        y_g = np.ones(2 * n_test)

        g_test_loss = combined.evaluate(z, y_g, verbose=0)
        g_train_loss = np.mean(np.array(epoch_g_loss), axis=0)

        # generate an epoch report on performance
        train_history['generator'].append(g_train_loss)
        train_history['discriminator'].append(d_train_loss)
        test_history['generator'].append(g_test_loss)
        test_history['discriminator'].append(d_test_loss)

        print('{0:<22s} | {1:4s}'.format(
            'component', *d.metrics_names))
        print('-' * 35)

        ROW_FMT = '{0:<22s} | {1:<4.4f}'
        print(ROW_FMT.format('generator (train)',
                             train_history['generator'][-1]))
        print(ROW_FMT.format('generator (test)',
                             test_history['generator'][-1]))
        print(ROW_FMT.format('discriminator (train)',
                             train_history['discriminator'][-1]))
        print(ROW_FMT.format('discriminator (test)',
                             test_history['discriminator'][-1]))

        # Aave weights every epoch
        g.save_weights("{}weight_g_epoch_{:03d}.hdf5".format(
            WEIGHT_DIR, epoch), True)
        d.save_weights("{}weight_d_epoch_{:03d}.hdf5".format(
            WEIGHT_DIR, epoch), True)

        # generate some digits to display
        noise = np.random.uniform(-1, 1, (100, LATENT_SIZE))
        # get a batch to display
        generated_images = g.predict(noise, verbose=0)

        # arrange them into a grid
        img = (np.concatenate([r.reshape(-1, 28)
                               for r in np.split(generated_images, 10)
                               ], axis=-1) * SCALE + SCALE).astype(np.uint8)

        Image.fromarray(img).save(
            '{}plot_epoch_{:03d}_generated.png'.format(VIS_DIR, epoch))

    pickle.dump({'train': train_history, 'test': test_history},
                open('acgan-history.pkl', 'wb'))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-prog", dest="prog", action="store_false")
    parser.set_defaults(prog=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    train(prog=args.prog)
