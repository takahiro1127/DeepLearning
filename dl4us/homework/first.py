import numpy as np
import pandas as pd
import os

def load_mnist():

    # 学習データ
    x_train = np.load('/root/userspace/public/lesson1/data/x_train.npy')
    y_train = np.load('/root/userspace/public/lesson1/data/y_train.npy')
    
    # テストデータ
    x_test = np.load('/root/userspace/public/lesson1/data/x_test.npy')

    x_train = x_train.reshape(-1, 784).astype('float32') / 255
    x_test = x_test.reshape(-1, 784).astype('float32') / 255
    y_train = np.eye(10)[y_train.astype('int32').flatten()]

    return (x_train, x_test, y_train)

x_train, x_test, y_train = load_mnist()
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import regularizers

model = Sequential()

model.add(Dense(512, input_shape=(784,), activation='relu', use_bias=True, kernel_initializer='he_normal'))
model.add(Dropout(0.1))
model.add(Dense(256, activation='relu', use_bias=True, kernel_initializer='he_normal'))
model.add(Dropout(0.1))
model.add(Dense(128, activation='relu', use_bias=True, kernel_initializer='he_normal'))
model.add(Dropout(0.1))
model.add(Dense(64, activation='relu', use_bias=True, kernel_initializer='he_normal'))
model.add(Dropout(0.1))
model.add(Dense(32, activation='relu', use_bias=True, kernel_initializer='he_normal'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=32, epochs=35, validation_split=0.1)

pred_y = model.predict(x_test)
pred = np.argmax(pred_y, 1)
    
submission = pd.Series(pred, name='label')
submission.to_csv('/root/userspace/submissionjikken23.csv', header=True, index_label='id')
