"""
This file creates the recurrent neural network (RNN).
An RNN is a better version of the FFNN model that predicts an output with given inputs.
The architecture of the RNN  can be seen in res/models/rnn/model_figures/RNN_TEST_MODEL.png.
"""

from tensorflow.keras import layers
from tensorflow.keras.models import Model
from keras.utils.vis_utils import plot_model


def create_rnn(model_name, save_path, plot_save_path, embedding_dim, neurons, dropout, source_dict, target_dict):
    """
    Function that creates the recurrent neural network, and plots the architecture.
    :param model_name: String - Model name
    :param save_path: String - Model save path
    :param plot_save_path: String - Model architecture plot path
    :param embedding_dim: Integer - Embedding/Output dimension of Layer
    :param neurons: Integer - Amount of neurons
    :param dropout: Float - Amount of dropout for layers
    :param source_dict: Dictionary - Loaded dictionary of source language
    :param target_dict: Dictionary - Loaded dictionary of target language
    :return: Functional - FFNN model
    """
    encoder_vocab = source_dict.get_size()
    decoder_vocab = target_dict.get_size()

    # Encode Data, Input shape for model is (batch_size, max_sequence_length, vocab_size)
    encoder_input = layers.Input(shape=(None,))

    # Embedding layer, parallelize embedding for whole words for training
    encoder_embedding = layers.Embedding(input_dim=encoder_vocab, output_dim=embedding_dim, mask_zero=True)

    enc_emb_output = encoder_embedding(encoder_input)
    encoder_mask = encoder_embedding.compute_mask(encoder_input)

    # Bidirectional Encoder with GRU Layer,
    # backward_layer is generated automatically from layer input if backward_layer=None
    bi_encoder_gru = layers.Bidirectional(
        layer=layers.GRU(units=neurons, return_state=False, return_sequences=True, dropout=dropout),
        # , recurrent_dropout=0.4
        merge_mode='sum', backward_layer=None)

    bi_encoder_output = bi_encoder_gru(enc_emb_output, initial_state=None, mask=encoder_mask, training=True)

    # Add Dense for better performance
    encoder_dense = layers.Dense(neurons)(bi_encoder_output)
    encoder_dense = layers.Dense(neurons)(encoder_dense)

    encoder_output = layers.Dropout(rate=dropout, noise_shape=(None, 1, neurons))(encoder_dense)

    # Decode Data
    decoder_input = layers.Input(shape=(None,))

    # Embedding layer, parallelize embedding for whole words for training
    decoder_embedding = layers.Embedding(input_dim=decoder_vocab, output_dim=embedding_dim, mask_zero=True)

    dec_emb_output = decoder_embedding(decoder_input)
    decoder_mask = decoder_embedding.compute_mask(decoder_input)

    decoder_gru = layers.GRU(units=neurons, return_state=False, return_sequences=True,
                             dropout=dropout)

    decoder_gru_output = decoder_gru(dec_emb_output, initial_state=None, mask=decoder_mask)

    decoder_output = layers.Dropout(rate=dropout, noise_shape=(None, 1, neurons))(decoder_gru_output)

    # Attention layer with representations encoder_output and last decoder hidden states decoder_output
    context_vector = layers.AdditiveAttention()([decoder_output, encoder_output], mask=[decoder_mask, encoder_mask])

    # Concat attention input and decoder GRU output
    decoder_concat_input = layers.Concatenate()([decoder_output, context_vector])

    # Projection Layer
    projection = layers.Dense(decoder_vocab, activation=None)(decoder_concat_input)

    # Softmax Layer
    decoder_outputs = layers.Softmax()(projection, mask=None)

    rnn_model = Model([encoder_input, decoder_input], decoder_outputs)

    rnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])

    rnn_model.save(filepath=f"{save_path}/{model_name}")

    rnn_model.summary()

    plot_model(rnn_model, to_file=plot_save_path, show_shapes=True)

    return rnn_model
