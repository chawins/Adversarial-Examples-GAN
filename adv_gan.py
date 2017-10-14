from lib.model import *
from lib.utils import *
from param import *

# Set CUDA visible device to GPU:0
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def train(batch_size=BATCH_SIZE, prog=True):

    # Load dataset and roughly rescale to [-1, 1]
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = (x_train.astype(np.float32) - SCALE) / SCALE
    # Add channel axis
    x_train = x_train[:, :, :, np.newaxis]
    x_test = x_test[:, :, :, np.newaxis]

    # Build model
    d = acgan_mnist_model_d()
    g = acgan_mnist_model_g()
    d_on_g = generator_containing_discriminator(g, d)

    # Set optimizer
    d_opt = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    g_opt = SGD(lr=0.0005, momentum=0.9, nesterov=True)

    # Set loss function
    g.compile(loss='binary_crossentropy', optimizer="SGD")
    d_on_g.compile(loss='binary_crossentropy', optimizer=g_opt)
    d.trainable = True
    d.compile(loss='binary_crossentropy', optimizer=d_opt)

    n_batch = int(x_train.shape[0] / BATCH_SIZE)
    for epoch in range(N_EPOCH):
        print("Epoch {} of {}".format(epoch, N_EPOCH))
        for step in range(n_batch):

            # Generate samples from generator
            noise = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))
            generated_images = g.predict(noise, verbose=0)

            # --------------------- Train discriminator ---------------------- #
            image_batch = x_train[step * BATCH_SIZE:(step + 1) * BATCH_SIZE]
            x_d = np.concatenate((image_batch, generated_images))
            y = [1] * BATCH_SIZE + [0] * BATCH_SIZE
            d_loss = d.train_on_batch(x_d, y)

            # ----------------------- Train generator ------------------------ #
            noise = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))
            d.trainable = False
            g_loss = d_on_g.train_on_batch(noise, [1] * BATCH_SIZE)
            d.trainable = True

            # Save weights
            if step % 10 == 9:
                g.save_weights('generator', True)
                d.save_weights('discriminator', True)

            if prog:
                if step % 20 == 0:
                    # Save generated samples
                    image = combine_images(generated_images)
                    image = image * SCALE + SCALE
                    Image.fromarray(image.astype(np.uint8)).save(
                        IMG_DIR + str(epoch) + "_" + str(step) + ".png")
                    # Print losses
                    print("batch %d d_loss : %f" % (step, d_loss))
                    print("batch %d g_loss : %f" % (step, g_loss))


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
        IMG_DIR + "generated_image.png")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--nice", dest="nice", action="store_true")
    parser.set_defaults(nice=False)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    if args.mode == "train":
        train(batch_size=args.batch_size)
    elif args.mode == "generate":
        generate(batch_size=args.batch_size, nice=args.nice)
