"""
This file has all necessary functions to create the input needed for translation and for training the RNN.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences


def calc_longest_sequence(*paths):
    """
    Function that calculates the longest sequence of source and target files.
    :param paths: *String - Paths of files
    :return: Integer - longest sequence
    """
    max_sequence = 0
    for path in paths:
        if path is None:
            break
        else:
            with open(path, 'r', encoding='utf-8') as f:
                context = f.readlines()
            new_max_sequence = np.amax([len(s.split()) for s in context])
            if new_max_sequence > max_sequence:
                max_sequence = new_max_sequence

    return max_sequence


def text_to_dict_indices(sentences, dict_for_sentences):
    """
    Function that converts String words to Integer indices of the given Dictionary.
    :param sentences: List - String sentences
    :param dict_for_sentences: Dictionary - Dictionary of the sentence
    :return: List - Dictionary indices of sentences
    """
    result = []

    for s in sentences:
        s_array = s.split()

        sentence_dict_index = []

        for w in s_array:
            sentence_dict_index.append(dict_for_sentences.word_to_index(w))

        result.append(sentence_dict_index)
    return result


def get_padded_sequences(source_path, target_path, source_dict, target_dict, source_path_2=None, target_path_2=None):
    """
    Function that pads the input, filling with zeros.
    :param source_path: String - Path of source file
    :param target_path: String - Path of target file
    :param source_dict: Dictionary - Loaded dictionary of source language
    :param target_dict: Dictionary - Loaded dictionary of target language
    :param source_path_2: String - Path of source file 2 (Optional, to calculate the longest sequence)
    :param target_path_2: String - Path of target file 2 (Optional, to calculate the longest sequence)
    :return: Array | Array | Array - padded source, padded target, padded labels
    """
    with open(source_path, 'r', encoding='utf-8') as f:
        source_context = f.readlines()
    with open(target_path, 'r', encoding='utf-8') as f:
        target_context = f.readlines()

    source = text_to_dict_indices(source_context, source_dict)
    target = text_to_dict_indices(target_context, target_dict)
    labels = text_to_dict_indices(target_context, target_dict)

    t_max = calc_longest_sequence(source_path, target_path, source_path_2, target_path_2)

    # add "time-shift" for decoder input and output
    for target_line in target:
        target_line.insert(0, 2)  # 2 is <s>

    for label_line in labels:
        label_line.append(3)  # 3 is </s>

    padded_source = [pad_sequences(source, maxlen=t_max, padding='post')]
    padded_target = [pad_sequences(target, maxlen=t_max, padding='post')]
    padded_labels = [pad_sequences(labels, maxlen=t_max, padding='post')]

    padded_source = np.squeeze(np.array(padded_source), axis=0)
    padded_target = np.squeeze(np.array(padded_target), axis=0)
    padded_labels = np.squeeze(np.array(padded_labels), axis=0)

    return padded_source, padded_target, padded_labels


def get_rnn_greedy_input(source_path, source_dict, i_max):
    """
    Function that prepares and gets the greedy search input for RNN.
    :param source_path: String - Path of source file
    :param source_dict: Dictionary - Loaded dictionary of source language
    :param i_max: Integer - Maximum sequence length
    :return: Array - Padded source
    """
    with open(source_path, 'r', encoding='utf-8') as f:
        source_context = f.readlines()

    source = text_to_dict_indices(source_context, source_dict)

    for source_line in source:
        source_line.append(3)

    padded_source = [pad_sequences(source, maxlen=i_max, padding='post')]

    padded_source = np.squeeze(np.array(padded_source), axis=0)

    return padded_source


def get_rnn_beam_input(source_path, source_dict, batch_size, i_max):
    """
    Function that prepares and gets the beam search input for RNN.
    :param source_path: String - Path of source file
    :param source_dict: Dictionary - Loaded dictionary of source language
    :param batch_size: Integer - Batch size
    :param i_max: Integer - Maximum sequence length
    :return: Array - source batch entries, target batch entries
    """
    with open(source_path, 'r', encoding='utf-8') as f:
        source_context = f.readlines()

    source = text_to_dict_indices(source_context, source_dict)

    for source_line in source:
        source_line.append(3)

    padded_source = [pad_sequences(source, maxlen=i_max, padding='post')]
    padded_source = np.squeeze(np.array(padded_source), axis=0)
    padded_target = np.zeros(padded_source.shape)
    for p_t in padded_target:
        p_t[0] = 2
    cnt = 0
    batches_source = []
    batches_target = []
    batch_s = []
    batch_t = []
    for p_s, p_t in zip(padded_source, padded_target):
        batch_s.append(p_s)
        batch_t.append(p_t)
        cnt += 1
        if cnt == batch_size:
            cnt = 0
            batches_source.append(np.array(batch_s))
            batches_target.append(np.array(batch_t))
            batch_s = []
            batch_t = []

    if cnt != batch_size:
        batches_source.append(np.array(batch_s))
        batches_target.append(np.array(batch_t))

    return batches_source, batches_target


def get_datasets(train_source_path, train_target_path,
                 eval_source_path, eval_target_path,
                 source_dict, target_dict,
                 batch_size):
    """
    Function that prepares the datasets for RNN model training and evaluation.
    :param train_source_path: String - Path of source training file
    :param train_target_path: String - Path of target training file
    :param eval_source_path: String - Path of source evaluation file
    :param eval_target_path: String - Path of target evaluation file
    :param source_dict: Dictionary - Loaded dictionary of source language
    :param target_dict: Dictionary - Loaded dictionary of target language
    :param batch_size: Integer - Size of Batch
    :return: DatasetV1 | DatasetV2 - training dataset, evaluation dataset
    """
    # Pad inputs for our model
    train_source, train_target, train_labels = get_padded_sequences(source_path=train_source_path,
                                                                    target_path=train_target_path,
                                                                    source_dict=source_dict,
                                                                    target_dict=target_dict)

    eval_source, eval_target, eval_labels = get_padded_sequences(source_path=eval_source_path,
                                                                 target_path=eval_target_path,
                                                                 source_dict=source_dict,
                                                                 target_dict=target_dict)

    # Create Tensorflow Datasets
    dataset_train_input = tf.data.Dataset.from_tensor_slices((train_source, train_target))
    dataset_train_label = tf.data.Dataset.from_tensor_slices(train_labels)
    dataset_train = tf.data.Dataset.zip((dataset_train_input, dataset_train_label))

    dataset_train = dataset_train.batch(batch_size, drop_remainder=False)  # if last batch should be cancelled or not
    dataset_train.shuffle(35000, reshuffle_each_iteration=True)

    dataset_val_input = tf.data.Dataset.from_tensor_slices((eval_source, eval_target))
    dataset_val_label = tf.data.Dataset.from_tensor_slices(eval_labels)
    dataset_eval = tf.data.Dataset.zip((dataset_val_input, dataset_val_label))
    dataset_eval = dataset_eval.batch(batch_size, drop_remainder=False)

    return dataset_train, dataset_eval
