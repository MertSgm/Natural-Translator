"""
This file contains the self attention neural network (SATNN), which is based on the paper
    A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A.N. Gomez, Ł. Kaiser, I. Polosukhin:
    Attention is all you need.
    In Advances in Neural Information Processing Systems, pp. 6000–6010, 2017.
The model that is created here is not suited for translation, as it is faulty.
However, the paper and the associated model is still very interesting - so it is displayed for example purposes.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from src.preprocessing.data_prepocessing_rnn import get_padded_sequences


def encoder_layers(x, encoder_mask, embedding_dim, dropout):
    """
    Function that creates one instance of an encoder block.
    :param x: KerasTensor - Encoder output
    :param encoder_mask: KerasTensor, Encoder mask
    :param embedding_dim: Integer - Embedding dimension
    :param dropout: Dropout - dropout layer
    :return: KerasTensor - normalized encoder output
    """
    mha = layers.MultiHeadAttention(num_heads=8, key_dim=8, dropout=dropout)

    layer_norm1 = layers.LayerNormalization(epsilon=1e-6)
    layer_norm2 = layers.LayerNormalization(epsilon=1e-6)

    dropout1 = layers.Dropout(rate=dropout)

    ffnn_dense1 = layers.Dense(embedding_dim, activation='relu',
                               kernel_initializer='he_normal')  # (batch_size, seq_len, dff)
    ffnn_dense2 = layers.Dense(embedding_dim)

    att_output = mha(x, x, x, attention_mask=encoder_mask, training=True)
    out1 = layer_norm1(x + att_output)  # (batch_size, input_seq_len, d_model)

    ffnn_output = ffnn_dense1(out1)
    ffnn_output = ffnn_dense2(ffnn_output)
    ffnn_output = dropout1(ffnn_output)
    out2 = layer_norm2(out1 + ffnn_output)

    return out2


def decoder_layers(x, encoder_output, decoder_mask, look_ahead_mask, embedding_dim, dropout):
    """
    Function that creates one instance of a decoder block.
    :param x: KerasTensor - Decoder output
    :param encoder_output: Any - Encoder output
    :param decoder_mask: KerasTensor - Decoder mask
    :param look_ahead_mask: KerasTensor - Look-ahead mask
    :param embedding_dim: Integer - Embedding dimension
    :param dropout: Dropout - dropout layer
    :return: KerasTensor - normalized decoder output
    """
    mha1 = layers.MultiHeadAttention(num_heads=8, key_dim=8, dropout=dropout)
    mha2 = layers.MultiHeadAttention(num_heads=8, key_dim=8, dropout=dropout)

    layer_norm1 = layers.LayerNormalization(epsilon=1e-6)
    layer_norm2 = layers.LayerNormalization(epsilon=1e-6)
    layer_norm3 = layers.LayerNormalization(epsilon=1e-6)

    dropout1 = layers.Dropout(rate=dropout)

    ffnn_dense1 = layers.Dense(embedding_dim, activation='relu',
                               kernel_initializer='he_normal')  # shape (batch_size, seq_len, dff)
    ffnn_dense2 = layers.Dense(embedding_dim)

    attn1 = mha1(x, x, attention_mask=look_ahead_mask, training=True)
    out1 = layer_norm1(x + attn1)  # shape (batch_size, input_seq_len, d_model)

    attn2 = mha2(encoder_output, encoder_output, out1, attention_mask=decoder_mask, training=True)

    out2 = layer_norm2(attn2 + out1)  # shape (batch_size, input_seq_len, d_model)

    ffnn_output = ffnn_dense1(out2)
    ffnn_output = ffnn_dense2(ffnn_output)
    ffnn_output = dropout1(ffnn_output)
    out3 = layer_norm3(ffnn_output + out2)

    return out3


def encoder(input1, num_layers, i_max, embedding_dim, dropout, source_dict):
    """
    Function that creates the encoder block for the transformator.
    The encoder block is executed num_layers times.
    :param input1: Layer - Input layer
    :param num_layers: Integer - amount of encoder blocks
    :param i_max: Integer - maximum sequence length
    :param embedding_dim: Integer - Embedding dimension
    :param dropout: Float - dropout %
    :param source_dict: Dictionary - Loaded dictionary of source language
    :return: KerasTensor - Encoder output
    """
    encoder_vocab = source_dict.get_size()

    encoder_input = input1

    encoder_embedding = layers.Embedding(input_dim=encoder_vocab, output_dim=embedding_dim, mask_zero=True)
    # encoder_mask = encoder_embedding.compute_mask(encoder_input)
    encoder_mask = create_padding_mask(encoder_input)

    positional_embeddings = np.zeros((i_max, embedding_dim))

    for position in range(i_max):
        for i in range(0, embedding_dim, 2):
            positional_embeddings[position, i] = (
                np.sin(position / (10000 ** ((2 * i) / embedding_dim)))
            )
            positional_embeddings[position, i + 1] = (
                np.cos(position / (10000 ** ((2 * (i + 1)) / embedding_dim)))
            )

    dropout = layers.Dropout(rate=dropout)

    # adding embedding and position encoding
    encoder_output = encoder_embedding(encoder_input)  # (batch_size, input_seq_len, d_model)

    x = encoder_output + positional_embeddings

    x = dropout(x, training=True)

    for i in range(num_layers):
        x = encoder_layers(x=x, encoder_mask=encoder_mask, embedding_dim=embedding_dim, dropout=dropout)

    return x


def decoder(input2, num_layers, encoder_output, i_max, embedding_dim, dropout, target_dict):
    """
    Function that creates the encoder block for the transformator.
    The encoder block is executed num_layers times.
    :param input2: Layer - Input layer
    :param num_layers: Integer - amount of encoder blocks
    :param encoder_output: Any - Encoder output
    :param i_max: Integer - Maximum sequence length
    :param embedding_dim: Integer - Embedding dimension
    :param dropout: Float - Dropout
    :param target_dict: Dictionary - Loaded dictionary of target language
    :return: Any - decoder_output + positional_embeddings
    """
    decoder_vocab = target_dict.get_size()

    decoder_input = input2

    decoder_embedding = layers.Embedding(input_dim=decoder_vocab, output_dim=embedding_dim, mask_zero=True)
    # decoder_mask = decoder_embedding.compute_mask(decoder_input)
    decoder_mask = create_padding_mask(decoder_input)

    look_ahead_mask = create_look_ahead_mask(i_max)

    positional_embeddings = np.zeros((i_max, embedding_dim))

    for position in range(i_max):
        for i in range(0, embedding_dim, 2):
            positional_embeddings[position, i] = (
                np.sin(position / (10000 ** ((2 * i) / embedding_dim)))
            )
            positional_embeddings[position, i + 1] = (
                np.cos(position / (10000 ** ((2 * (i + 1)) / embedding_dim)))
            )

    dropout = layers.Dropout(rate=dropout)

    decoder_output = decoder_embedding(decoder_input)
    x = decoder_output + positional_embeddings

    x = dropout(x, training=True)

    for i in range(num_layers):
        x = decoder_layers(x=x,
                           encoder_output=encoder_output,
                           decoder_mask=decoder_mask,
                           look_ahead_mask=look_ahead_mask,
                           embedding_dim=embedding_dim,
                           dropout=dropout)

    return x


def transformator(model_name,
                  source_path,
                  target_path,
                  source_dict,
                  target_dict,
                  save_path,
                  num_layers,
                  embedding_dim,
                  dropout):
    """
    Function that creates the self-attention neural network.
    :param model_name: String - Model name
    :param source_path: Path - BPE encoded source path (training)
    :param target_path:Path - BPE encoded target path (training)
    :param source_dict: Dictionary - Loaded dictionary of source language
    :param target_dict: Dictionary - Loaded dictionary of target language
    :param save_path: Path - model save path
    :param num_layers: Integer - Amount of Encoder/Decoder block repetitions
    :param embedding_dim: Integer - Embedding Dimension
    :param dropout: Float - Dropout %
    :return: Functional - SATNN model
    """
    decoder_vocab = target_dict.get_size()

    source, _, _ = get_padded_sequences(source_path=source_path,
                                        target_path=target_path,
                                        source_dict=source_dict,
                                        target_dict=target_dict)

    i_max = np.array(source.shape[1])

    input1 = layers.Input(shape=(None,))
    input2 = layers.Input(shape=(None,))

    enc_output = encoder(input1=input1,
                         num_layers=num_layers,
                         i_max=i_max,
                         embedding_dim=embedding_dim,
                         dropout=dropout,
                         source_dict=source_dict)

    decoder_output = decoder(input2=input2,
                             num_layers=num_layers,
                             i_max=i_max,
                             encoder_output=enc_output,
                             embedding_dim=embedding_dim,
                             dropout=dropout,
                             target_dict=target_dict)

    projection = layers.Dense(decoder_vocab)(decoder_output)
    softmax = layers.Softmax()(projection, mask=None)

    model = Model([input1, input2], softmax)

    # warmup_steps = 4000
    # step_num = 40000
    # learning_rate = embedding_dim**(-0.5) * min(step_num**(-0.5), step_num * warmup_steps**(-1.5))
    #
    # optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
    #                                      epsilon=1e-9)

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])

    model.save(filepath=f"{save_path}/{model_name}")

    model.summary()

    return model


def create_padding_mask(seq):
    """
    Function that creates a padding mask.
    :param seq: KerasTensor - Sequence
    :return: KerasTensor - Padded Sequence
    """
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    x = seq[:, tf.newaxis, tf.newaxis, :]

    x = tf.cast(tf.math.equal(x, 0), tf.float32)

    return x


def create_look_ahead_mask(i_max):
    """
    Function that creates a look-ahead mask.
    :param i_max: Integer - maximum sequence length
    :return: KerasTensor - Look-ahead mask
    """
    mask = 1 - tf.linalg.band_part(tf.ones((i_max, i_max), dtype=tf.dtypes.int32), -1, 0, )

    x = tf.cast(tf.math.equal(mask, 0), tf.float32)
    return x
