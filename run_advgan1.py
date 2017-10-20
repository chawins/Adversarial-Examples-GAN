from keras.optimizers import Adam
from lib.model import *
from lib.utils import *
from param import *

# Set CUDA visible device to GPU:0
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Adam parameters suggested in https://arxiv.org/abs/1511.06434
adam_lr = 0.0002
adam_beta_1 = 0.5


def random_sample(size):

    # Generate samples from g
    z = np.random.uniform(-1, 1, (size, LATENT_SIZE))
    # Sampled labels
    y_sampled = np.random.randint(0, 10, size)
    # Target labels
    y_t = np.random.randint(0, 10, size)
    return z, y_sampled, y_t


def generate_random(g, size):

    z, y_sampled, y_t = random_sample(size)
    x_g = g.predict([z, y_sampled.reshape((-1, 1)),
                     y_t.reshape((-1, 1))], verbose=0)
    return x_g, y_sampled, y_t


def train(prog=True):

    # Load MNIST
    x_train, y_train, x_test, y_test = load_mnist()

    # Build model
    d = advgan1_mnist_model_d()
    g = advgan1_mnist_model_g()

    # Set up optimizers
    adam = Adam(lr=adam_lr, beta_1=adam_beta_1)

    # Set loss function and compile models
    g.compile(optimizer=adam, loss='sparse_categorical_crossentropy')
    d.compile(optimizer=adam, loss='sparse_categorical_crossentropy')
    combined = combine_advgan1(g, d)
    combined.compile(optimizer=adam, loss='sparse_categorical_crossentropy')

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
            x_g, y_sampled, y_t = generate_random(g, BATCH_SIZE)

            # Combine with real samples
            x_real = x_train[index * BATCH_SIZE:(index + 1) * BATCH_SIZE]
            x_d = np.concatenate((x_real, x_g))
            # Conditional labels
            y_real = y_train[index * BATCH_SIZE:(index + 1) * BATCH_SIZE]
            y_d = np.concatenate((y_real, y_sampled), axis=0)

            epoch_d_loss.append(d.train_on_batch(x_d, y_d))

            # ---------------- Train generator ------------------------------- #
            # Generate 2 * BATCH_SIZE samples to match d's batch size
            z, y_sampled, y_t = random_sample(2 * BATCH_SIZE)

            epoch_g_loss.append(combined.train_on_batch(
                [z, y_sampled.reshape((-1, 1)), y_t.reshape((-1, 1))], y_t))

        print('\nTesting for epoch {}:'.format(epoch + 1))
        n_test = x_test.shape[0]

        # ---------------- Test discriminator -------------------------------- #
        x_g, y_sampled, y_t = generate_random(g, n_test)

        x_d = np.concatenate((x_test, x_g))
        y_d = np.concatenate((y_test, y_sampled), axis=0)

        d_test_loss = d.evaluate(x_d, y_d, verbose=0)
        d_train_loss = np.mean(np.array(epoch_d_loss), axis=0)

        # ---------------- Test generator ------------------------------------ #
        z, y_sampled, y_t = random_sample(2 * n_test)

        g_test_loss = combined.evaluate(
            [z, y_sampled.reshape((-1, 1)), y_t.reshape((-1, 1))], y_t, verbose=0)
        g_train_loss = np.mean(np.array(epoch_g_loss), axis=0)

        # generate an epoch report on performance
        train_history['generator'].append(g_train_loss)
        train_history['discriminator'].append(d_train_loss)
        test_history['generator'].append(g_test_loss)
        test_history['discriminator'].append(d_test_loss)

        print('{0:<22s} | {1:4s}'.format(
            'component', *d.metrics_names))
        print('-' * 65)

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

        sampled_labels = np.array([[i] * 10 for i in range(10)]).reshape(-1, 1)
        sampled_target = np.array([[i for i in range(10)] * 10]).reshape(-1, 1)

        # get a batch to display
        generated_images = g.predict(
            [noise, sampled_labels, sampled_target], verbose=0)

        # Get classification on generated images
        y_pred = []
        for i, x in enumerate(generated_images):
            tmp = d.predict(x.reshape(1, 28, 28, 1))
            y_pred.append(np.argmax(tmp[0]))
            if (i + 1) % 10 == 0:
                print(y_pred)
                y_pred = []

        # Get classification on real images
        y_pred = np.argmax(d.predict(x_test), axis=1)
        n_correct = np.sum(y_pred == y_test)
        print("Accuracy: " + str(float(n_correct) / n_test))

        # arrange them into a grid
        img = (np.concatenate([r.reshape(-1, 28)
                               for r in np.split(generated_images, 10)
                               ], axis=-1) * 127.5 + 127.5).astype(np.uint8)

        Image.fromarray(img).save(
            '{}plot_epoch_{:03d}_generated.png'.format(VIS_DIR, epoch))

    pickle.dump({'train': train_history, 'test': test_history},
                open('acgan-history.pkl', 'wb'))


def generate(nice=False):

    g = acgan_mnist_model_g()
    g.compile(loss='binary_crossentropy', optimizer="SGD")
    g.load_weights('generator')

    if nice:

        d = acgan_mnist_model_d()
        d.compile(loss='binary_crossentropy', optimizer="SGD")
        d.load_weights('discriminator')

        noise = np.random.uniform(-1, 1, (BATCH_SIZE * 20, 100))
        generated_images = g.predict(noise, verbose=1)
        d_pret = d.predict(generated_images, verbose=1)
        index = np.arange(0, BATCH_SIZE * 20)
        index.resize((BATCH_SIZE * 20, 1))
        pre_with_index = list(np.append(d_pret, index, axis=1))
        pre_with_index.sort(key=lambda x: x[0], reverse=True)
        nice_images = np.zeros(
            (BATCH_SIZE,) + generated_images.shape[1:3], dtype=np.float32)
        nice_images = nice_images[:, :, :, None]
        for i in range(BATCH_SIZE):
            idx = int(pre_with_index[i][1])
            nice_images[i, :, :, 0] = generated_images[idx, :, :, 0]
        image = combine_images(nice_images)
    else:
        noise = np.random.uniform(-1, 1, (BATCH_SIZE, 100))
        generated_images = g.predict(noise, verbose=1)
        image = combine_images(generated_images)

    image = image * SCALE + SCALE
    Image.fromarray(image.astype(np.uint8)).save(
        VIS_DIR + "generated_image.png")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--nice", dest="nice", action="store_true")
    parser.add_argument("--no-prog", dest="prog", action="store_false")
    parser.set_defaults(nice=False)
    parser.set_defaults(prog=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    if args.mode == "train":
        train(prog=args.prog)
    elif args.mode == "generate":
        generate(nice=args.nice)
