"""
This file utilizes two algorithms to translate a source file to a target language.
The first algorithm is greedy search, which simply takes the best prediction value for the translation.
The second algorithm is beam search, which is a better version of greedy search.
Rather than looking only at the best value, beam search looks at the top k values - possibly finding
a better translation.
"""

import numpy as np
import tensorflow as tf
import src.preprocessing.bpe as bpe
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from src.preprocessing.data_prepocessing_rnn import get_rnn_greedy_input, get_rnn_beam_input


def greedy_search_rnn(model_path,
                      model_weights_path,
                      source_path,
                      save_translation_path,
                      source_dict,
                      target_dict,
                      i_max):
    """
    Function that translates a source file to the target language using greedy search with a neural model.
    :param model_path: String - RNN model path
    :param model_weights_path: String - RNN model weights path
    :param source_path: String - Path of source file that will be translated
    :param save_translation_path: String - Output path of translated file
    :param source_dict: Dictionary - Loaded dictionary of source language
    :param target_dict: Dictionary - Loaded dictionary of target language
    :param i_max: Integer - maximum sentence length
    :return: List - Translated sentences
    """
    model = load_model(filepath=model_path)
    model.load_weights(filepath=model_weights_path)
    print('Successfully loaded model weights.')

    padded_source = get_rnn_greedy_input(source_path, source_dict, i_max + 1)

    sentence_count = 0
    result = []

    print("Starting translation...")

    for sentence in padded_source:

        target_sentence = [[2]]
        target_sentence = [pad_sequences(target_sentence, maxlen=i_max + 1, padding='post')]
        last_word_index = 2
        word_count = 0

        sentence_translation = []
        target_sentence = target_sentence[0][0]
        while word_count < i_max and last_word_index not in [0, 3]:
            prediction = model.predict([sentence, target_sentence])

            top_word = tf.math.top_k(prediction[word_count][0], k=1)

            last_word_index = int(top_word[1][0])
            last_word_string = target_dict.index_to_word(last_word_index)

            target_sentence[word_count + 1] = last_word_index
            sentence_translation.append(last_word_string)
            word_count += 1

        sentence_translation = " ".join(str(t) for t in sentence_translation)
        result.append(sentence_translation)

        last_progress = int(sentence_count / len(padded_source) * 100)
        sentence_count += 1
        new_progress = int(sentence_count / len(padded_source) * 100)

        if new_progress != last_progress:
            print(new_progress, "% complete \n", end='')

    result = bpe.decode_data_string(result)

    output_file = open(save_translation_path, "w+")
    for sentence in result:
        output_file.write(f'{sentence}\n')
    output_file.close()

    print("Finished translation.")

    return result


def beam_search_rnn(model_path,
                    model_weights_path,
                    source_path,
                    save_translation_path,
                    source_dict,
                    target_dict,
                    batch_size,
                    i_max,
                    k):
    """
    Function that translates a source file to the target language using beam search with a neural model.
    :param model_path: String - RNN model path
    :param model_weights_path: String - RNN model weights path
    :param source_path: String - Path of source file that will be translated
    :param save_translation_path: String - Output path of translated file
    :param source_dict: Dictionary - Loaded dictionary of source language
    :param target_dict: Dictionary - Loaded dictionary of target language
    :param batch_size: Integer - Batch size
    :param i_max: Integer - maximum sentence length
    :param k: Integer - Top k results in beam search
    :return: List - Translated sentences
    """
    model = load_model(filepath=model_path)
    model.load_weights(filepath=model_weights_path)
    print('Successfully loaded model weights.')

    batches_source, batches_target = get_rnn_beam_input(source_path, source_dict, batch_size, i_max + 1)

    print("Starting translation...")

    translations = []
    total_sentence_count = 0
    for batch in batches_source:
        total_sentence_count += len(batch)

    for batch_s, batches_t in zip(batches_source, batches_target):

        batch_entries_paths = []  # = [ [[path], prob_of_path], ...]
        batch_entries_probabilities = []

        for s_t in batches_t:
            batch_entries_paths.append([s_t])
            batch_entries_probabilities.append([np.log(1)])

        batch_entries_probabilities = np.array(batch_entries_probabilities)
        batch_entries_probabilities = batch_entries_probabilities.reshape(1, len(batches_t))

        batch_entries_paths = np.array(batch_entries_paths)
        batch_entries_paths = batch_entries_paths.reshape(1, len(batches_t), i_max + 1)

        translated_batch_path = np.zeros([len(batch_s), k, i_max + 1])
        translated_batch_prob = np.zeros([len(batch_s), k])
        finished_sentence_in_batch = np.zeros(len(batches_t))  # <- if sentence finished entry = 1
        number_finished_sentences = np.zeros(len(batches_t))
        word_cnt = 0

        batch_s_copy = batch_s.copy()
        finished_sentences_count = 0
        while word_cnt < i_max and finished_sentences_count < len(batch_s):

            new_paths = []
            new_probs = []
            for paths, probs in zip(batch_entries_paths, batch_entries_probabilities):
                prediction_first = model.predict(x=[batch_s_copy, paths])
                top_k_predicts = tf.math.top_k(prediction_first, k=k)
                highest_indices = top_k_predicts.indices.numpy()
                highest_values = top_k_predicts.values.numpy()

                for l in range(k):
                    new_path = paths.copy()
                    old_prob = probs.copy()
                    new_prob = []
                    sentence_counter = 0
                    for n_pa, prob, h_i, h_v in zip(new_path, old_prob, highest_indices, highest_values):
                        n_pa[word_cnt + 1] = h_i[word_cnt][l]
                        new_prob.append(np.add(prob, np.log(h_v[word_cnt][l])))

                        sentence_counter += 1

                    new_paths.append(new_path)
                    new_probs.append(new_prob)

            new_paths = np.array(new_paths)
            new_probs = np.array(new_probs)

            # reshape
            reshaped_paths = np.zeros([len(batch_s_copy), len(new_paths), i_max + 1])
            reshaped_probs = np.zeros([len(batch_s_copy), len(new_paths)])

            for index_sentences in range(len(reshaped_paths)):
                for index_k in range(len(new_paths)):
                    reshaped_paths[index_sentences][index_k] = new_paths[index_k][index_sentences]
                    reshaped_probs[index_sentences][index_k] = new_probs[index_k][index_sentences]

            new_paths = reshaped_paths
            new_probs = reshaped_probs

            top_k_paths = np.zeros([k, len(batch_s_copy), i_max + 1])
            top_k_probs = np.full((k, len(batch_s_copy)), -float("inf"))

            sentences_counter = 0
            for paths, probs in zip(new_paths, new_probs):
                finished_trans_in_sentences = int(number_finished_sentences[sentences_counter])
                top_k_paths_probs = sorted(zip(paths, probs), key=lambda l: l[1], reverse=True)[
                                    :(k - finished_trans_in_sentences)]

                for index_k in range(len(top_k_paths_probs)):
                    top_k_paths[index_k][sentences_counter] = top_k_paths_probs[index_k][0]
                    top_k_probs[index_k][sentences_counter] = top_k_paths_probs[index_k][1]

                    # top_k_paths[index_k][sentences_counter]
                    if top_k_paths[index_k][sentences_counter][word_cnt + 1] in [0, 3]:
                        translated_batch_path[sentences_counter][int(number_finished_sentences[sentences_counter])] = \
                            top_k_paths[index_k][sentences_counter]
                        translated_batch_prob[sentences_counter][int(number_finished_sentences[sentences_counter])] = \
                            top_k_probs[index_k][sentences_counter]
                        number_finished_sentences[sentences_counter] += 1
                        top_k_probs[index_k][sentences_counter] = -float("inf")

                        if number_finished_sentences[sentences_counter] == k:
                            finished_sentences_count += 1
                            finished_sentence_in_batch[sentences_counter] = 1

                sentences_counter += 1

            batch_entries_paths = top_k_paths
            batch_entries_probabilities = top_k_probs
            word_cnt += 1

        for f_s_index in range(len(finished_sentence_in_batch)):
            if finished_sentence_in_batch[f_s_index] == 0:
                for index_k in range(k):
                    translated_batch_path[f_s_index][index_k] = batch_entries_paths[index_k][f_s_index]
                    translated_batch_prob[f_s_index][index_k] = batch_entries_probabilities[index_k][f_s_index]

        for t_path, t_prob in zip(translated_batch_path, translated_batch_prob):
            for index_k in range(len(t_path)):
                t_prob[index_k] = t_prob[index_k] / len(t_path[index_k])
            top_translation = sorted(zip(t_path, t_prob), key=lambda l: l[1], reverse=True)[
                              :1][0][0]
            sentence = ""

            for w in top_translation:
                sentence += target_dict.index_to_word(w)
            sentence = sentence.replace("<PAD>", "")
            translations.append(sentence)

        progress = int(len(translations) / total_sentence_count * 100)
        print(progress, "% complete \n", end='')

    result_words = bpe.decode_data_string(translations)

    output_file = open(save_translation_path, "w+")
    for sentence in result_words:
        output_file.write(f'{sentence}\n')
    output_file.close()

    print("Finished translation.")

    return translations
