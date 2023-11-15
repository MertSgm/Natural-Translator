"""
This file trains the feed forward neural network, as well as evaluates it.
Training: The weights of one layer is passed on to the next layer from bottom to top
(considering the 'adam' optimizer) - until training stops. Input is of fixed length.
Evaluation: The evaluation considers three metrics: Accuracy, Cross-Entropy and Perplexity.
    Accuracy: % of correct target label predictions
    Cross-Entropy: Loss function, compares two probability distributions
    Perplexity: How well the model predicts
"""

import random
import numpy as np
import src.preprocessing.batches as batches
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping


def train_ffnn(model_path,
               save_weights_path,
               batch_size,
               epochs,
               print_eval_info_freq,
               show_eval_info,
               eval_freq,
               train_source_path,
               train_target_path,
               eval_source_path,
               eval_target_path,
               source_dict,
               target_dict):
    """
    Function that trains the FFNN model with training data and evaluates the trained model on evaluate data.
    Saves the best weights during training, stops early if no improvement.
    :param model_path: String - FFNN path
    :param save_weights_path: String - Path to save model weights
    :param batch_size: Integer - Batch size
    :param epochs: Integer - Amount of epochs to train
    :param print_eval_info_freq: Integer - Print loss, accuracy and perplexity every x Batches
    :param show_eval_info: Boolean - Shows print_info if true
    :param eval_freq: Integer -Evaluation sequence
    :param train_source_path: String - BPE encoded training source path
    :param train_target_path: String - BPE encoded training target path
    :param eval_source_path: String - BPE encoded evaluate source path
    :param eval_target_path: String - BPE encoded evaluate target path
    :param source_dict: Dictionary - Loaded dictionary of source language
    :param target_dict: Dictionary - Loaded dictionary of target language
    :return: Functional - Trained model
    """
    model = load_model(model_path)

    train_source, train_target, train_labels = batches.create_batches(train_source_path,
                                                                      train_target_path,
                                                                      source_dict,
                                                                      target_dict,
                                                                      batch_size)
    eval_source, eval_target, eval_labels = batches.create_batches(eval_source_path,
                                                                   eval_target_path,
                                                                   source_dict,
                                                                   target_dict,
                                                                   batch_size)
    mid_perplexity = []

    complete_batches = list(zip(train_source, train_target, train_labels))

    random.shuffle(complete_batches)

    train_source_c, train_target_c, train_labels_c = zip(*complete_batches)

    # half learning rate if loss does not get better for patience-amount of epochs
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=0, verbose=0,
                                  mode='min', min_delta=0.005, cooldown=0, min_lr=0)

    # save best model checkpoint during training
    save_checkpoint = ModelCheckpoint(filepath=save_weights_path, monitor='loss', verbose=0,
                                      save_best_only=True, save_weights_only=True, mode='min', save_freq=batch_size)

    # train until no significant improvements are seen. Stop to prevent overfit
    early_stopping = EarlyStopping(monitor='loss', min_delta=0, patience=0, verbose=0,
                                   mode='min', baseline=None, restore_best_weights=True)

    for i in range(1, epochs + 1):

        batch_count = 1
        for big_s, big_t, big_l in zip(train_source_c, train_target_c, train_labels_c):
            big_s = np.array(big_s)
            big_t = np.array(big_t)
            big_l = np.array(big_l)

            # train model
            train_result = model.fit(x=[big_s, big_t], y=big_l, batch_size=batch_size, verbose=0,
                                     callbacks=[reduce_lr, save_checkpoint, early_stopping])

            if batch_count % print_eval_info_freq == 0 and show_eval_info:
                train_result = train_result.history
                perplexity = train_result['loss'][0]
                perplexity = 2 ** perplexity
                mid_perplexity.append(perplexity)
                print(f'Batch {batch_count} in Epoch: {i}, Loss: {train_result["loss"]}, Accuracy:'
                      f' {train_result["sparse_categorical_accuracy"]}, '
                      f'Perplexity: [{perplexity}]')

                # if current batch is in evaluate sequence
                if batch_count % eval_freq == 0:

                    print(f'Evaluating batch {batch_count}...')
                    evaluate_model(model=model, batch_entries=[eval_source, eval_target, eval_labels])

            batch_count += 1

    return model


def evaluate_model(model, batch_entries):
    """
    Function that evaluates the FFNN model by showing its accuracy, cross-entropy and perplexity.
    :param model: Functional - FFNN model
    :param batch_entries: List - Source windows; List, Target windows; List, Target labels
    :return: Float | Float | Float | Float - Mid-loss, Mid-Accuracy, Mid-Perplexity, Min-Perplexity
    """
    source = batch_entries[0]
    target = batch_entries[1]
    labels = batch_entries[2]

    total_acc = 0
    total_loss = 0
    total_perplexity = 0
    min_perplexity = -1
    number_of_test_batches = 0

    for big_s, big_t, big_l in zip(source, target, labels):
        big_s = np.array(big_s)
        big_t = np.array(big_t)
        big_l = np.array(big_l)
        acc_loss = model.evaluate(x=[big_s, big_t], y=big_l, batch_size=big_s.shape[0], verbose=0)

        perplexity = 2 ** acc_loss[0]

        if min_perplexity == -1:
            min_perplexity = perplexity
        min_perplexity = min(min_perplexity, perplexity)

        total_loss += acc_loss[0]
        total_perplexity += perplexity
        total_acc += acc_loss[1]
        number_of_test_batches += 1

    result = [total_loss / number_of_test_batches, total_acc / number_of_test_batches,
              total_perplexity / number_of_test_batches, min_perplexity]

    print(f"Mid loss: {result[0]}, Mid Accuracy: {result[1]}, "
          f"Mid Perplexity: {result[2]}, Min Perplexity: {result[3]} on Test Data")

    return result
