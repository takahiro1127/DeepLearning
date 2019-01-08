# 候補　単純に層を深くする　1.6
# attenstionは必須
# 時系列のデータはそれ自身のデータを取ってくるようにするやつもいいと思う。
# dropout 0.5 56
# beam search 3 -> 56
import numpy as np
from keras.models import Model
from keras.layers import Input, Embedding, Dense, LSTM
import csv

emb_dim = 256
hid_dim = 256

en_vocab_size = len(tokenizer_en.word_index) + 1
ja_vocab_size = len(tokenizer_ja.word_index) + 1

seqX_len = len(x_train[0])
seqY_len = len(y_train[0])

encoder_inputs = Input(shape=(seqX_len,))
encoder_embedded = Embedding(
    en_vocab_size, emb_dim, mask_zero=True)(encoder_inputs)
_, *encoder_states = LSTM(hid_dim, return_state=True)(encoder_embedded)

decoder_inputs = Input(shape=(seqY_len,))
decoder_embedding = Embedding(ja_vocab_size, emb_dim)
decoder_embedded = decoder_embedding(decoder_inputs)
decoder_lstm = LSTM(hid_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(
    decoder_embedded, initial_state=encoder_states)
decoder_dense = Dense(ja_vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
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


def decode_sequence(input_seq, bos_eos, max_output_length):
    states_value = encoder_model.predict(input_seq)

    target_seq = np.array(bos_eos[0])
    output_seq = bos_eos[0][:]

    while True:
        output_tokens, *states_value = decoder_model.predict([target_seq] + states_value)
        sampled_token_index = [np.argmax(output_tokens[0, -1, :])]
        output_seq += sampled_token_index

        if (sampled_token_index == bos_eos[1] or len(output_seq) > max_output_length):
            break

        target_seq = np.array(sampled_token_index)

    return output_seq
