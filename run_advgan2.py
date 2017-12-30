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
# Number of batches to train discriminator before training generator for 1 batch
k = 16


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


def collage(images):
    img = (np.concatenate([np.concatenate([s for s in r], axis=1)
                           for r in np.split(images, 10)], axis=0) *
           SCALE + SCALE).astype(np.uint8)
    return np.squeeze(img)


def print_progress(g, d, train_history, test_history, losses, data, epoch):

    # generate an epoch report on performance
    train_history['generator'].append(losses['g_train_loss'])
    train_history['discriminator'].append(losses['d_train_loss'])
    test_history['generator'].append(losses['g_test_loss'])
    test_history['discriminator'].append(losses['d_test_loss'])

    print('{0:<22s} | {1:4s} | {2:15s} | {3:5s}'.format(
        'component', *d.metrics_names))
    print('-' * 65)

    ROW_FMT = '{0:<22s} | {1:<4.2f} | {2:<15.2f} | {3:<5.2f}'
    print(ROW_FMT.format('generator (train)',
                         *train_history['generator'][-1]))
    print(ROW_FMT.format('generator (test)',
                         *test_history['generator'][-1]))
    print(ROW_FMT.format('discriminator (train)',
                         *train_history['discriminator'][-1]))
    print(ROW_FMT.format('discriminator (test)',
                         *test_history['discriminator'][-1]))

    # Save weights every epoch
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
        y_pred.append([tmp[0][0, 0], np.argmax(tmp[1])])
        if (i + 1) % 10 == 0:
            print(y_pred)
            y_pred = []

    # Get classification on real images
    _, _, x_test, y_test = data
    y_pred = np.argmax(d.predict(x_test)[1], axis=1)
    n_correct = np.sum(y_pred == y_test)
    print("Accuracy: " + str(float(n_correct) / len(x_test)))

    # Arrange them into a grid
    Image.fromarray(collage(generated_images)).save(
        '{}plot_epoch_{:03d}_generated.png'.format(VIS_DIR, epoch))

    return train_history, test_history


def print_progress_index(g, d, epoch, index):

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
        y_pred.append([tmp[0][0, 0], np.argmax(tmp[1])])
        if (i + 1) % 10 == 0:
            print(y_pred)
            y_pred = []

    # Arrange them into a grid
    Image.fromarray(collage(generated_images)).save(
        '{}plot_epoch_{:03d}_index_{:03d}_generated.png'.format(VIS_DIR, epoch,
                                                                index))


def train(prog=True):

    # Load MNIST
    data = load_mnist()
    x_train, y_train, x_test, y_test = data

    # Build model
    d = advgan2_mnist_model_d()
    g = advgan1_mnist_model_g()

    # Set up optimizers
    adam = Adam(lr=adam_lr, beta_1=adam_beta_1)

    # Set loss function and compile models
    g.compile(optimizer=adam, loss='binary_crossentropy')
    d.compile(optimizer=adam, loss=[
              'binary_crossentropy', 'sparse_categorical_crossentropy'])
    combined = combine_advgan1(g, d)
    combined.compile(optimizer=adam, loss=[
                     'binary_crossentropy', 'sparse_categorical_crossentropy'])

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
            for _ in range(k):
                x_g, y_sampled, y_t = generate_random(g, BATCH_SIZE)

                # Combine with real samples
                # ind = np.random.randint(0, len(x_train), BATCH_SIZE)
                ind = [i for i in range(
                    index * BATCH_SIZE, (index + 1) * BATCH_SIZE)]
                x_real = x_train[ind]
                x = np.concatenate((x_real, x_g))
                # Conditional labels
                y_d = np.array([1] * BATCH_SIZE + [0] * BATCH_SIZE)
                y_real = y_train[ind]
                y = np.concatenate((y_real, y_sampled), axis=0)

                d_loss = d.train_on_batch(x, [y_d, y])

            epoch_d_loss.append(d_loss)

            # ---------------- Train generator ------------------------------- #
            # Generate 2 * BATCH_SIZE samples to match d's batch size
            z, y_sampled, y_t = random_sample(2 * BATCH_SIZE)
            y_d = np.array([1] * 2 * BATCH_SIZE)

            epoch_g_loss.append(combined.train_on_batch(
                [z, y_sampled.reshape((-1, 1)), y_t.reshape((-1, 1))], [y_d, y_t]))

            # if (index + 1) % 20 == 0:
            #     # Print progress and samples
            #     print_progress_index(g, d, epoch, index)

        print('\nTesting for epoch {}:'.format(epoch + 1))
        n_test = x_test.shape[0]

        # ---------------- Test discriminator -------------------------------- #
        x_g, y_sampled, y_t = generate_random(g, n_test)

        x = np.concatenate((x_test, x_g))
        y = np.concatenate((y_test, y_sampled), axis=0)
        y_d = np.array([1] * n_test + [0] * n_test)

        d_test_loss = d.evaluate(x, [y_d, y], verbose=0)
        d_train_loss = np.mean(np.array(epoch_d_loss), axis=0)

        # ---------------- Test generator ------------------------------------ #
        z, y_sampled, y_t = random_sample(2 * n_test)
        y_d = np.array([1] * n_test * 2)

        g_test_loss = combined.evaluate(
            [z, y_sampled.reshape((-1, 1)), y_t.reshape((-1, 1))], [y_d, y_t], verbose=0)
        g_train_loss = np.mean(np.array(epoch_g_loss), axis=0)

        losses = {"g_train_loss": g_train_loss, "d_train_loss": d_train_loss,
                  "g_test_loss": g_test_loss, "d_test_loss": d_test_loss}
        # Print progress, evaluate every epoch
        print_progress(g, d, train_history, test_history, losses, data, epoch)

    pickle.dump({'train': train_history, 'test': test_history},
                open('advgan1-history.pkl', 'wb'))


def generate(size=1, weight_path=None, **kwargs):

    y_sampled = kwargs.get("y_sampled")
    y_t = kwargs.get("y_t")

    g = advgan1_mnist_model_g()
    g.compile(optimizer="SGD", loss='sparse_categorical_crossentropy')

    if weight_path is None:
        weight_path = WEIGHT_DIR + "weights_best.h5"
    g.load_weights(weight_path)

    z = np.random.uniform(-1, 1, (size, LATENT_SIZE))
    x_g = g.predict([z, y_sampled.reshape((-1, 1)),
                     y_t.reshape((-1, 1))], verbose=0)
    return x_g


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
