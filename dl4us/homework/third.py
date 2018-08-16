import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical


def load_dataset():
    # 学習データ
    x_train = np.load('/root/userspace/public/lesson3/data/x_train.npy')
    y_train = np.load('/root/userspace/public/lesson3/data/y_train.npy')
    y_train = to_categorical(y_train[:, np.newaxis], num_classes=5)

    # テストデータ
    x_test = np.load('/root/userspace/public/lesson3/data/x_test.npy')

    return (x_train, x_test, y_train)


x_train, x_test, y_train = load_dataset()

import tensorflow.python.keras
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM
from keras import optimizers
import pandas as pd

hid_dim = 10
adam1 = optimizers.Adam(lr=0.01, clipnorm=1.)
model = Sequential()

# input_shape=(系列長T, x_tの次元), output_shape=(系列長T, units(=hid_dim))
model.add(LSTM(hid_dim, input_shape=x_train.shape[1:]))
model.add(Dense(y_train.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=adam1, metrics=['accuracy'])
model.fit(x_train, y_train, epochs=50, batch_size=100,
          verbose=2, validation_split=0.2)

y_pred = np.argmax(model.predict(x_test), 1)
# 性能評価

submission = pd.Series(y_pred, name='label')
submission.to_csv('/root/userspace/submission.csv',
                  header=True, index_label='id')
