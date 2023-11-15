"""
This file trains the recurrent neural network, as well as evaluates it.
Training: The weights of one layer is passed on to the next layer from bottom to top,
as well as from left to right.
(considering the 'adam' optimizer) - until training stops. Input is of dynamic length.
Evaluation: The evaluation considers three metrics: Accuracy, Cross-Entropy and Perplexity.
    Accuracy: % of correct target label predictions
    Cross-Entropy: Loss function, compares two probability distributions
    Perplexity: How well the model predicts
"""

import matplotlib.pyplot as plt
import src.preprocessing.data_prepocessing_rnn as dp
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping


def train_rnn(model_path,
              save_weights_path,
              save_plot_path,
              batch_size,
              epochs,
              train_source_path,
              train_target_path,
              eval_source_path,
              eval_target_path,
              source_dict,
              target_dict):
    """
    Function that trains the RNN model with training data and evaluates the trained model on evaluate data.
    Saves the best weights during training, stops early if no improvement.
    :param model_path: String - FFNN path
    :param save_weights_path: String - Path to save model weights
    :param save_plot_path: String - Path to save model training history plot
    :param batch_size: Integer - Batch size
    :param epochs: Integer - Amount of epochs to train
    :param train_source_path: String - BPE encoded training source path
    :param train_target_path: String - BPE encoded training target path
    :param eval_source_path: String - BPE encoded evaluate source path
    :param eval_target_path: String - BPE encoded evaluate target path
    :param source_dict: Dictionary - Loaded dictionary of source language
    :param target_dict: Dictionary - Loaded dictionary of target language
    :return: Functional - Trained model
    """

    dataset_train, dataset_val = dp.get_datasets(train_source_path=train_source_path,
                                                 train_target_path=train_target_path,
                                                 eval_source_path=eval_source_path,
                                                 eval_target_path=eval_target_path,
                                                 source_dict=source_dict,
                                                 target_dict=target_dict,
                                                 batch_size=batch_size)
    model = load_model(filepath=model_path)

    print(f'Loaded Model.')
    print("Starting training...")

    # half learning rate if loss does not get better for patience-amount of epochs
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=0, verbose=1,
                                  mode='min', min_delta=0.005, cooldown=0, min_lr=0)

    # save best model checkpoint during training
    save_checkpoint = ModelCheckpoint(filepath=save_weights_path, monitor='val_loss', verbose=1,
                                      save_best_only=True, save_weights_only=True, mode='min', save_freq='epoch')

    # train until no significant improvements are seen. Stop to prevent overfit
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=1,
                                   mode='min', baseline=None, restore_best_weights=True)

    # train model
    history = model.fit(dataset_train, validation_data=dataset_val, verbose=1, epochs=epochs,
                        callbacks=[reduce_lr, save_checkpoint, early_stopping])

    print("Finished training.")

    # plot training history
    plt.plot(history.history['val_loss'], label='evaluation data')
    plt.plot(history.history['loss'], label='train data')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_plot_path, bbox_inches='tight')
    plt.close()
