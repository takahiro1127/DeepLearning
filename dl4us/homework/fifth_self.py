import os
import sys
import math
import copy
from keras.applications.vgg16 import VGG16
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Dense, Embedding, LSTM, Activation, Flatten, Reshape, dot, Permute, Lambda, CuDNNLSTM
from keras.optimizers import RMSprop
from keras.utils import np_utils
from keras import backend as K
from keras.layers import Lambda
from tqdm import tqdm
import csv

sys.path.append('/root/userspace/public/lesson5/master')

from utils import Node

np.random.seed(1234)

batch_size = 32
emb_dim = 32
hid_dim = 128

### Encoder ###
x = Input(shape=(224, 224, 3))
x_normalized = Lambda(lambda x: x / 255.)(x)  # [0, 255) -> [0, 1)
encoder = VGG16(weights='imagenet', include_top=False,
                input_tensor=x_normalized)

for layer in encoder.layers:
    layer.trainable = False

u = Flatten()(encoder.output)

### Decoder ###
u_map = Reshape((7*7, 512))(u)

h_0 = Dense(hid_dim)(u)
cell_0 = Dense(hid_dim)(u)

y = Input(shape=(None,), dtype='int32')
y_in = Lambda(lambda x: x[:, :-1])(y)
y_out = Lambda(lambda x: x[:, 1:])(y)

mask = Lambda(lambda x: K.cast(K.not_equal(x, w2i['<pad>']), 'float32'))(y_out)

embedding = Embedding(vocab_size, emb_dim)
lstm = CuDNNLSTM(hid_dim, return_sequences=True, return_state=True)

y_emb = embedding(y_in)
h, _, _ = lstm(y_emb, initial_state=[h_0, cell_0])
h = Activation('tanh')(h)

### Attention ###
dense_att = Dense(hid_dim)
_u_map = dense_att(u_map)
score = dot([_u_map, h], axes=-1)

permute_att1 = Permute((2, 1))
activation_att = Activation('softmax')
score = permute_att1(score)
a = activation_att(score)

permute_att2 = Permute((2, 1))
context = dot([u_map, a], axes=(1, 2))
context = permute_att2(context)

dense_output1 = Dense(hid_dim)
dense_output2 = Dense(vocab_size)
softmax = Activation('softmax')
h_tilde = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=2))([h, context])
h_tilde = dense_output1(h_tilde)
y_pred = dense_output2(h_tilde)
y_pred = softmax(y_pred)

### Learning ###


def cross_entropy(y_true, y_pred):
    return -K.mean(K.sum(K.sum(y_true * K.log(K.clip(y_pred, 1e-10, 1)), axis=-1) * mask, axis=1))


model = Model([x, y], y_pred)

model.compile(loss=cross_entropy, optimizer='rmsprop')


def generator(data_X, data_y, batch_size=32):

    n_batches = math.ceil(len(data_X) / batch_size)

    while True:
        for i in range(n_batches):
            start = i * batch_size
            end = (i + 1) * batch_size

            data_x_mb = data_X[start:end]
            data_y_mb = data_y[start:end]

            data_x_mb = np.array(data_x_mb).astype('float32')
            data_y_mb = pad_sequences(
                data_y_mb, dtype='int32', padding='post', value=w2i['<pad>'])
            data_y_mb_oh = np.array([np_utils.to_categorical(
                datum_y, vocab_size) for datum_y in data_y_mb[:, 1:]])

            yield [data_x_mb, data_y_mb], data_y_mb_oh


n_batches_train = math.ceil(len(x_train) / batch_size)

try:
    model.fit_generator(
        generator(x_train, y_train),
        epochs=10,
        steps_per_epoch=n_batches_train,
    )
except KeyboardInterrupt:
    pass

### Predict ###
encoder_model = Model([x], [u_map, h_0, cell_0])

u_map_inpt = Input(shape=(7*7, 512,))
h_tm1 = Input(shape=(hid_dim,))
cell_tm1 = Input(shape=(hid_dim,))
y_t = Input(shape=(1,))
y_emb_t = embedding(y_t)
_, h_t, cell_t = lstm(y_emb_t, initial_state=[h_tm1, cell_tm1])
h_t = Lambda(lambda x: x[:, None, :])(h_t)
cell_t = Lambda(lambda x: x[:, None, :])(cell_t)

_u_map = dense_att(u_map_inpt)
score_t = dot([_u_map, h_t], axes=-1)

score_t = permute_att1(score_t)
a_t = activation_att(score_t)

context_t = dot([u_map_inpt, a_t], axes=(1, 2))
context_t = permute_att2(context_t)

h_tilde_t = Lambda(lambda x: K.concatenate(
    [x[0], x[1]], axis=2))([h_t, context_t])
h_tilde_t = dense_output1(h_tilde_t)
pred_t = dense_output2(h_tilde_t)

decoder_model = Model([y_t, h_tm1, cell_tm1, u_map_inpt], [
                      pred_t, h_t, cell_t, a_t])


def np_log(x):
    return np.log(np.clip(x, 1e-10, x))


def beam_search(x, max_len=100, k=3):
    u_map, h_tm1, cell_tm1 = encoder_model.predict(x)
    y_tm1 = np.array([w2i['<s>']])

    root = Node(w2i['<s>'])

    # [score, y_tm1, h_tm1, cell_tm1, a, y_pred]
    candidates = [[0, y_tm1, h_tm1, cell_tm1, [], y_tm1]]

    t = 0
    while t < max_len:
        root.depth += 1
        t += 1

        # すべての候補を一時的に保管するリスト
        tmp_candidates = []

        # </s>がすべての候補で出力されたかどうかのフラッグ
        end_flag = True
        for score_tm1, y_tm1, h_tm1, cell_tm1, a, y_pred in candidates:
            a = copy.deepcopy(a)
            if y_tm1[0] == w2i['</s>']:
                tmp_candidates.append(
                    [score_tm1, y_tm1, h_tm1, cell_tm1, a, y_pred])
            else:
                end_flag = False
                y_t, h_t, cell_t, a_t = decoder_model.predict(
                    [y_tm1, h_tm1, cell_tm1, u_map])
                h_t, cell_t = h_t[:, 0], cell_t[:, 0]
                a.append(a_t.flatten())

                # 対数化
                y_t = np_log(y_t.flatten())

                # 確率の高い単語とそのidを取得
                y_t, s_t = np.argsort(y_t)[::-1][:k], np.sort(y_t)[::-1][:k]

                # スコア (対数尤度) を蓄積
                score_t = score_tm1 + s_t

                # すべての候補を一時的に保管
                tmp_candidates.extend(
                    [[score_tk, np.array([y_tk]), h_t, cell_t, a, np.append(y_pred, [y_tk])]
                        for score_tk, y_tk in zip(score_t, y_t)]
                )
        if end_flag:
            break

        candidates = sorted(tmp_candidates, key=lambda x: -x[0]/len(x[-1]))[:k]

    return candidates[0][-1]


pred_list = []
for i in tqdm(range(x_test.shape[0])):
    pred_y = beam_search(x_test[i:i+1], k=3, max_len=40)

    pred_list.append(list(pred_y[1:-1]))

with open('/root/userspace/submission.csv', 'w') as file:
    writer = csv.writer(file, lineterminator='\n')
    writer.writerows(pred_list)
