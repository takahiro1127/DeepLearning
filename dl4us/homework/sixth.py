import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_mnist():
    # 学習データ
    x_train = np.load('/root/userspace/public/lesson6/data/x_train.npy')
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
    return x_train


x_train = load_mnist()
generated_image_num = 50


def plot_mnist(n_ex=10, dim=(2, 5), figsize=(8, 4)):
    noise = np.random.uniform(0, 1, size=[n_ex, 100])
    generated_images = generator.predict(noise)

    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        img = generated_images[i, :, :, 0]
        plt.imshow(img, cmap='binary')
        plt.axis('off')
    plt.tight_layout()
    plt.show()


from keras.layers import Input
from keras.layers.core import Reshape, Dense, Dropout, Flatten, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.models import Model
from tqdm import tqdm
import csv


def Generator():
    nch = 200
    model_input = Input(shape=[100])
    x = Dense(nch*14*14, kernel_initializer='glorot_normal')(model_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Reshape([14, 14, nch])(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(int(nch/2), (3, 3), padding='same',
               kernel_initializer='glorot_uniform')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(int(nch/4), (3, 3), padding='same',
               kernel_initializer='glorot_uniform')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(1, (1, 1), padding='same',
               kernel_initializer='glorot_uniform')(x)
    model_output = Activation('sigmoid')(x)
    model = Model(model_input, model_output)

    return model


def Discriminator(shape, dropout_rate=0.25, opt=Adam(lr=1e-4)):
    model_input = Input(shape=shape)
    x = Conv2D(256, (5, 5), padding='same',
               kernel_initializer='glorot_uniform', strides=(2, 2))(model_input)
    x = LeakyReLU(0.2)(x)
    x = Dropout(dropout_rate)(x)
    x = Conv2D(512, (5, 5), padding='same',
               kernel_initializer='glorot_uniform', strides=(2, 2))(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(dropout_rate)(x)
    x = Flatten()(x)
    x = Dense(256)(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(dropout_rate)(x)
    model_output = Dense(2, activation='softmax')(x)
    model = Model(model_input, model_output)
    model.compile(loss='categorical_crossentropy', optimizer=opt)

    return model


def combined_network(generator, discriminator, opt=Adam(lr=1e-3)):
    gan_input = Input(shape=[100])
    x = generator(gan_input)
    gan_output = discriminator(x)
    model = Model(gan_input, gan_output)
    model.compile(loss='categorical_crossentropy', optimizer=opt)

    return model


def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val


def train(step=10000, BATCH_SIZE=128):
    for e in tqdm(range(step)):
        image_batch = x_train[np.random.randint(
            0, x_train.shape[0], size=BATCH_SIZE), :, :, :]
        noise_gen = np.random.uniform(0, 1, size=[BATCH_SIZE, 100])
        generated_images = generator.predict(noise_gen)

        make_trainable(discriminator, True)

        X = np.concatenate((image_batch, generated_images))
        y = np.zeros([2*BATCH_SIZE, 2])
        y[:BATCH_SIZE, 1] = 1
        y[BATCH_SIZE:, 0] = 1

        discriminator.train_on_batch(X, y)

        make_trainable(discriminator, False)

        noise_gen = np.random.uniform(0, 1, size=[BATCH_SIZE, 100])
        y2 = np.zeros([BATCH_SIZE, 2])
        y2[:, 1] = 1

        GAN.train_on_batch(noise_gen, y2)


generator = Generator()
discriminator = Discriminator(x_train.shape[1:])
make_trainable(discriminator, False)
GAN = combined_network(generator, discriminator)

train(step=100)

noise = np.random.uniform(0, 1, size=[generated_image_num, 100])
generated_images = generator.predict(noise)

with open('/root/userspace/submission.csv', 'w') as file:
    writer = csv.writer(file, lineterminator='\n')
    writer.writerows(generated_images.reshape(-1, 28*28).tolist())
