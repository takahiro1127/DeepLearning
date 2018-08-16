import numpy as np
import pandas as pd


def load_cifar10():

    # 学習データ
    x_train = np.load('/root/userspace/public/lesson2/data/x_train.npy')
    y_train = np.load('/root/userspace/public/lesson2/data/y_train.npy')

    # テストデータ
    x_test = np.load('/root/userspace/public/lesson2/data/x_test.npy')

    x_train = x_train / 255.
    x_test = x_test / 255.

    y_train = np.eye(10)[y_train]

    return (x_train, x_test, y_train)


x_train, x_test, y_train = load_cifar10()
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator


model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same', input_shape=(32, 32, 3)))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))


model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])
datagen = ImageDataGenerator(
    width_shift_range=0.2,  # 3.1.1 左右にずらす
    height_shift_range=0.2,  # 3.1.2 上下にずらす
    horizontal_flip=True,  # 3.1.3 左右反転
    # 3.2.1 Global Contrast Normalization (GCN) (Falseに設定しているのでここでは使用していない)
    samplewise_center=False,
    samplewise_std_normalization=False,
    zca_whitening=False)

model.fit(x=x_train, y=y_train, batch_size=32, epochs=5, validation_split=0.1)

y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, 1)

submission = pd.Series(y_pred, name='label')
submission.to_csv('/root/userspace/submission.csv',
                  header=True, index_label='id')
