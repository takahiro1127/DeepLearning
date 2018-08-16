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

from keras.applications.vgg16 import VGG16
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD


# VGG-16モデルの構造と重みをロード
# include_top=Falseによって、VGG16モデルから全結合層を削除
input_tensor = Input(shape=(32, 32, 3))
vgg16_model = VGG16(include_top=False, weights='imagenet',
                    input_tensor=input_tensor)

# 全結合層の構築
top_model = Sequential()
top_model.add(Flatten(input_shape=vgg16_model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Activation("relu"))
top_model.add(Dropout(0.5))
top_model.add(Dense(10, activation='softmax'))

# 全結合層を削除したVGG16モデルと上で自前で構築した全結合層を結合
model = Model(input=vgg16_model.input, output=top_model(vgg16_model.output))

# 図3における14層目までのモデル重みを固定（VGG16のモデル重みを用いる）
for layer in model.layers[:15]:
        layer.trainable = False

# モデルのコンパイル
model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True),
              metrics=['accuracy'])
model.fit(x=x_train, y=y_train, batch_size=32, epochs=5, validation_split=0.1)

y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, 1)

submission = pd.Series(y_pred, name='label')
submission.to_csv('/root/userspace/submission.csv',
                  header=True, index_label='id')
73%