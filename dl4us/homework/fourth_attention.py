import numpy as np
from keras.models import Model
from keras.layers import Input, Embedding, Dense, LSTM, CuDNNLSTM, concatenate, dot, Activation
import csv

emb_dim = 256
hid_dim = 256

en_vocab_size = len(tokenizer_en.word_index) + 1
ja_vocab_size = len(tokenizer_ja.word_index) + 1

seqX_len = len(x_train[0])
seqY_len = len(y_train[0])

encoder_inputs = Input(shape=(seqX_len,))
encoder_embedded = Embedding(en_vocab_size, emb_dim)(encoder_inputs)
encoder_seq = CuDNNLSTM(hid_dim, return_sequences=True)(encoder_embedded)
encoded_seq, *encoder_states = CuDNNLSTM(hid_dim, return_sequences=True, return_state=True)(encoder_output)

decoder_inputs = Input(shape=(seqY_len,))
decoder_embedding = Embedding(ja_vocab_size, emb_dim)
decoder_embedded = decoder_embedding(decoder_inputs)
decoder_lstm = CuDNNLSTM(hid_dim, return_sequences=True, return_state=True)
decoded_seq, _, _ = decoder_lstm(
    decoder_embedded, initial_state=encoder_states)
# decoder_dense = Dense(ja_vocab_size, activation='softmax')
# decoder_outputs = decoder_dense(decoder_outputs0)

att_dim = 256
score_dense = Dense(hid_dim)
# shape: (seqY_len, hid_dim) -> (seqY_len, hid_dim)
score = score_dense(decoded_seq)
# shape: [(seqY_len, hid_dim), (seqX_len, hid_dim)] -> (seqY_len, seqX_len)
score = dot([score, encoded_seq], axes=(2, 2))
# shape: (seqY_len, seqX_len) -> (seqY_len, seqX_len)
attention = Activation('softmax')(score)
# shape: [(seqY_len, seqX_len), (seqX_len, hid_dim)] -> (seqY_len, hid_dim)
context = dot([attention, encoded_seq], axes=(2, 1))
# shape: [(seqY_len, hid_dim), (seqY_len, hid_dim)] -> (seqY_len, 2*hid_dim)
concat = concatenate([context, decoded_seq], axis=2)
attention_dense = Dense(att_dim, activation='tanh')
# shape: (seqY_len, 2*hid_dim) -> (seqY_len, att_dim)
attentional = attention_dense(concat)
output_dense = Dense(ja_vocab_size, activation='softmax')
# shape: (seqY_len, att_dim) -> (seqY_len, ja_vocab_size)
outputs = output_dense(attentional)
print(encoder_inputs)
print(decoder_inputs)
print(outputs)
model = Model([encoder_inputs, decoder_inputs], outputs)
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')

train_target = np.hstack(
    (y_train[:, 1:], np.zeros((len(y_train), 1), dtype=np.int32)))
model.fit([x_train, y_train], np.expand_dims(train_target, -1),
          batch_size=128, epochs=1, validation_split=0.2)

encoder_model = Model(encoder_inputs, encoder_states)

decoder_states_inputs = [Input(shape=(hid_dim,)), Input(shape=(hid_dim,))]
decoder_inputs = Input(shape=(1,))
decoder_embedded = decoder_embedding(decoder_inputs)
decoder_outputs, *decoder_states = decoder_lstm(decoder_embedded, initial_state=decoder_states_inputs)
decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = Model([decoder_inputs] + decoder_states_inputs,
                      [decoder_outputs] + decoder_states)

# Tensor("input_3:0", shape=(?, 18), dtype=float32)
# Tensor("input_4:0", shape=(?, 18), dtype=float32)
# Tensor("dense_6/truediv:0", shape=(?, 18, 8777), dtype=float32)
