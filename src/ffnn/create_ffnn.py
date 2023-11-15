"""
This file creates the feed forward neural network (FFNN).
An FFNN is a simple model that that predicts an output with given inputs.
The architecture of the FFNN (which can be seen in res/models/ffnn/model_figures/FFNN_TEST_MODEL.png) is a
first prototype. As such, the architecture in itself is not great (i.e., no Dropout)
and is not well suited for translation.
"""

from keras.utils.vis_utils import plot_model
from tensorflow.keras.models import Model
from tensorflow.keras import layers


def create_ffnn(model_name, save_path, plot_save_path, embedding_dim, neurons, source_dict, target_dict):
    """
    Function that creates the FFNN model and plots its architecture.
    :param model_name: String - Model name
    :param save_path: String - Model save path
    :param plot_save_path: String - Model architecture plot path
    :param embedding_dim: Integer - Embedding/Output dimension of layers
    :param neurons: Integer - Amount of neurons
    :param source_dict: Dictionary - Loaded dictionary of source language
    :param target_dict: Dictionary - Loaded dictionary of target language
    :return: Functional - FFNN model
    """
    window_size_source = 3
    window_size_target = 1

    # Build Model
    input_source = layers.Input(shape=(window_size_source,), name='input_source')
    input_target = layers.Input(shape=(window_size_target,), name='input_target')

    # embedded Layer (fully connected layer without activation and bias)
    # source window w=3, target window w=1
    embedding_source = layers.Embedding(input_dim=source_dict.get_size(), output_dim=embedding_dim,
                                        input_length=window_size_source,
                                        name='embedding_layer_origin')(input_source)
    embedding_target = layers.Embedding(input_dim=target_dict.get_size(), output_dim=embedding_dim,
                                        input_length=window_size_target,
                                        name='embedding_layer_target')(input_target)

    flat1 = layers.Flatten()(embedding_source)
    flat2 = layers.Flatten()(embedding_target)

    # fully-connected layer for source and target
    dense1 = layers.Dense(neurons, activation='relu', kernel_initializer='he_normal',
                          bias_initializer='zero', name='fully_connected_source')(flat1)
    dense2 = layers.Dense(neurons, activation='relu', kernel_initializer='he_normal',
                          bias_initializer='zero', name='fully_connected_target')(flat2)

    # concat layer  Vector [source, target]
    concat1 = layers.Concatenate()([dense1, dense2])

    # fully connected layer 1
    dense = layers.Dense(neurons, activation='relu', kernel_initializer='he_normal',
                         bias_initializer='zero')(concat1)

    # fully connected layer 2 (projection-layer), output size = size of vocabulary, no activation-func
    projection_layer = layers.Dense(target_dict.get_size(), kernel_initializer='he_normal',
                                    bias_initializer='zero', name='projection_layer')(dense)

    # Output/Softmax-layer
    softmax_layer = layers.Softmax()(projection_layer)

    # Optimizer func = Adam
    # Activation func = ReLu
    # Loss func = Cross-Entropy
    model = Model(inputs=[input_source, input_target], outputs=softmax_layer)

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])

    model.save(filepath=f"{save_path}/{model_name}")

    model.summary()

    plot_model(model, to_file=plot_save_path, show_shapes=True)

    return model
