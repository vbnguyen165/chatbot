from data_processing import *
import numpy as np
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Embedding, LSTM, Input

# Load embedding matrix built from the vocabulary.
# Refer to 'embedding.py' for the process of creating the embedding matrix.
embedding_matrix = np.load("embedding_matrix.dat")

embed = Embedding(VOCAB_SIZE,
                  100,
                  trainable=True, mask_zero=True)

embed.build((None,))
embed.set_weights([embedding_matrix])

# The two below models were inspired by keras' tutorial on Seq2Seq model
# https://keras.io/examples/nlp/lstm_seq2seq/

# TRAINING RNN ENCODER-DECODER MODEL

# Create encoder layer for the RNN Encoder-Decoder Model
# input layer
enc_inputs = Input(shape=(None,))
# embedding layer
enc_embedding = embed(enc_inputs)
# LSTM layer
enc_outputs, state_h, state_c = LSTM(400, return_state=True)(enc_embedding)
# discard `encoder_outputs` and only keep the states.
enc_states = [state_h, state_c]

# Create decoder layer for the RNN Encoder-Decoder Model, using 'enc_states' as
# the initial state

# input layer
dec_inputs = Input(shape=(None,))
# embedding layer
dec_embedding = embed(dec_inputs)
# LSTM layer
dec_lstm = LSTM(400, return_state=True, return_sequences=True)
# Return full output sequences and internal states. We don't use the
# return states in the training model, but we will use them in inference.
dec_outputs, _, _ = dec_lstm(dec_embedding,
                             initial_state=enc_states)
dec_dense = Dense(VOCAB_SIZE, activation='softmax')
# dense layer
output = dec_dense(dec_outputs)

# RNN Encoder-Decoder Model that will turn  `enc_inputs` & `dec_inputs`
# into `dec_outputs`
model = Model([enc_inputs, dec_inputs], output)

model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['acc'])
model.summary()

# # The below chunk of code is for training;
# # Running it will overide the current model
# model.fit([encoder_input_data, decoder_input_data], decoder_output_data,
#           batch_size=64, epochs=15, validation_split=0.3,)
# model.save_weights('chatbot_wo_attention.h5')

# current pre-trained model
weight_file = 'chatbot_wo_attention.h5'

if os.path.isfile(weight_file):
    model.load_weights(weight_file)

# INFERENCE RNN ENCODER-DECODER MODEL

# The encoder for inference takes the input question
# represented as a padded vector
enc_inputs = model.input[0]  # input_1
# It outputs the internal state vectors of the LSTM layer
# into a context vector for the decoder
enc_outputs, state_h_enc, state_c_enc = model.layers[3].output  # lstm_1
enc_states = [state_h_enc, state_c_enc]
enc_model = Model(enc_inputs, enc_states)

# The decoder for inference takes the context vector
# and the 1-word sequence as inputs
dec_inputs = model.input[1]
# Connects the 1-word sequence to the Embedding layer
dec_embedding = embed(dec_inputs)
dec_state_input_h = Input(shape=(400,))
dec_state_input_c = Input(shape=(400,))
dec_states_inputs = [dec_state_input_h, dec_state_input_c]
dec_lstm = model.layers[4]
dec_outputs, state_h_dec, state_c_dec = \
    dec_lstm(dec_embedding, initial_state=dec_states_inputs)
dec_states = [state_h_dec, state_c_dec]
# Output of the decoder is the next predicted word in one-hot encoding format
dec_outputs = dec_dense(dec_outputs)
dec_model = Model(
    inputs=[dec_inputs] + dec_states_inputs,
    outputs=[dec_outputs] + dec_states)


for _ in range(100):
    input_seq = input('you: ')
    # Clean the input sentence
    clean_input = clean_text(input_seq)

    # Convert the input to an integer vector using padding
    words = clean_input.split()
    tokens_list = list()
    for current_word in words:
        current_word_index = tokenizer.word_index.get(current_word, '')
        if current_word_index != '':
            tokens_list.append(current_word_index)

    padded_input = pad_sequences([tokens_list], maxlen=maxlen_questions,
                                 padding='post')

    # Feed the input to the encoder model for inference
    # and save the states (context vector)
    context = enc_model.predict(padded_input)
    # Start with a target sequence of size 1 - the start-of-sequence word
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = tokenizer.word_index['start']
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        # Feed the state vectors and 1-word target sequence
        # to the decoder to produce predictions for the next word
        dec_outputs, h, c = dec_model.predict([target_seq]
                                              + context)

        # Next word is the one with the highest probability
        sampled_word_index = np.argmax(dec_outputs[0, -1, :])
        sampled_word = None
        # Append the sampled word to the target sequence
        for word, index in tokenizer.word_index.items():
            if sampled_word_index == index:
                if word != 'end':
                    decoded_sentence += ' {}'.format(word)
                sampled_word = word

        # Repeat until the model generate the end-of-sequence word 'end'
        # or the sentence reaches the length limit
        if sampled_word == 'end' \
                or len(decoded_sentence.split()) \
                > maxlen_answers:
            stop_condition = True

        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_word_index
        context = [h, c]
    print('chatbot:' + decoded_sentence)
