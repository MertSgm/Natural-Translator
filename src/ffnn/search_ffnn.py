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


def greedy_search_ffnn(model_path,
                       model_weights_path,
                       source_path,
                       save_translation_path,
                       source_dict,
                       target_dict,
                       i_max):
    """
    Function that translates a source file to the target language using greedy search with a neural model.
    :param model_path: String - FFNN model path
    :param model_weights_path: String - FFNN model weights path
    :param source_path: String - Path of source file that will be translated
    :param save_translation_path: String - Output path of translated file
    :param source_dict: Dictionary - Loaded dictionary of source language
    :param target_dict: Dictionary - Loaded dictionary of target language
    :param i_max: Integer - maximum sentence length
    :return: List - Translated sentences
    """
    model = load_model(filepath=model_path)
    model.load_weights(filepath=model_weights_path)
    print(f'Successfully loaded model weights.')

    data = open(source_path, encoding="utf8").readlines()

    sentence_count = 0
    total_english_sentences = []

    print("Starting translation...")

    for sentence in data:
        last_word = "<s>"
        english_sentence = []

        sentence_split = sentence.split()
        word_count = 0
        english_sentence.append(last_word)

        while english_sentence[word_count] != "</s>" and word_count <= i_max:
            word_count += 1

            if not word_count < len(sentence_split) - 2:
                sentence_split.append("</s>")
                sentence_split.append("</s>")

            source = np.array([[source_dict.word_to_index(sentence_split[word_count - 1]),
                                source_dict.word_to_index(sentence_split[word_count]),
                                source_dict.word_to_index(sentence_split[word_count + 1]),
                                ]])
            target = np.array([target_dict.word_to_index(last_word)])

            last_word = target_dict.index_to_word(np.argmax(model.predict(x=[source, target])))
            english_sentence.append(last_word)

        finished_sentence = ' '.join(str(e) for e in english_sentence)
        finished_sentence = finished_sentence.replace("<s>", "")
        finished_sentence = finished_sentence.replace("</s>", "")

        last_progress = int(sentence_count / len(data) * 100)
        sentence_count += 1
        new_progress = int(sentence_count / len(data) * 100)

        if new_progress != last_progress:
            print(new_progress, "% complete \n", end='')

        print(int(sentence_count / len(data) * 100), "% complete \r", end='')

        total_english_sentences.append(finished_sentence)

    total_english_sentences = bpe.decode_data_string(total_english_sentences)

    output_file = open(f'{save_translation_path}', "w+")
    for sentence in total_english_sentences:
        output_file.write(f'{sentence}\n')
    output_file.close()

    print("Finished translation.")

    return total_english_sentences


def beam_search_ffnn(model_path,
                     model_weights_path,
                     source_path,
                     save_translation_path,
                     source_dict,
                     target_dict,
                     i_max,
                     k):
    """
    Function that translates a source file to the target language using beam search with a neural model.
    :param model_path: String - FFNN model path
    :param model_weights_path: String - FFNN model weights path
    :param source_path: String - Path of source file that will be translated
    :param save_translation_path: String - Output path of translated file
    :param source_dict: Dictionary - Loaded dictionary of source language
    :param target_dict: Dictionary - Loaded dictionary of target language
    :param i_max: Integer - maximum sentence length
    :param k: Integer - Top k results in beam search
    :return: List - Translated sentences
    """
    model = load_model(filepath=model_path)
    model.load_weights(filepath=model_weights_path)
    print(f'Successfully loaded model weights.')

    data = open(source_path, encoding="utf8").readlines()

    result_indices = []

    sentence_count = 0
    print("Starting translation...")
    for sentence in data:

        sentence = sentence.split()
        complete_sentence_indices = []

        for word in sentence:
            complete_sentence_indices.append(source_dict.word_to_index(word))

        align = 3

        entries_proba = [[[2], np.log(1)]]  # = [ [[path], prob_of_path],...]
        cnt = 0
        all_possibilities_finished = False
        cnt_finished_beams = 0

        while (not all_possibilities_finished) and cnt < i_max and cnt_finished_beams < k:
            new_entries_prob = []
            cnt_finished_beams = 0

            for path_and_prob in entries_proba:
                if path_and_prob[0][-1] == 3:
                    cnt_finished_beams += 1
                    new_entries_prob.append(path_and_prob)
                else:
                    if cnt + align > len(complete_sentence_indices):
                        complete_sentence_indices.append(3)
                        complete_sentence_indices.append(3)
                        complete_sentence_indices.append(3)

                    source = complete_sentence_indices[cnt:align + cnt]
                    target = path_and_prob[0][-1]

                    source = np.array([source])
                    target = np.array([target])

                    prediction_first = model.predict(x=[source, target])[0]
                    top_k_predicts = tf.math.top_k(prediction_first, k=k)
                    highest_indices = top_k_predicts.indices.numpy().tolist()
                    highest_values = top_k_predicts.values.numpy().tolist()

                    for h_i, h_v in zip(highest_indices, highest_values):
                        # curr = path_and_prob
                        new_path = path_and_prob[0].copy()
                        new_path.append(h_i)

                        new_prob = np.add(path_and_prob[1], np.log(h_v))
                        new_entries_prob.append([new_path, new_prob])

            cnt += 1

            entries_proba = sorted(new_entries_prob, key=lambda l: l[1], reverse=True)[:k]

        last_progress = int(sentence_count / len(data) * 100)
        sentence_count += 1
        new_progress = int(sentence_count / len(data) * 100)

        if new_progress != last_progress:
            print(new_progress, "% complete \n", end='')

        finished_sentence = sorted(new_entries_prob, key=lambda l: l[1], reverse=True)[:1][0][0]
        result_indices.append(finished_sentence)

    result_words = []
    for sentence in result_indices:
        s = ""
        for word in sentence:
            if word == 3:
                break
            if not word == 2:
                s += target_dict.index_to_word(word)
        result_words.append(s)

    result_words = bpe.decode_data_string(result_words)

    output_file = open(f'{save_translation_path}', "w+")
    for sentence in result_words:
        output_file.write(f'{sentence}\n')
    output_file.close()

    print("Finished translation.")

    return result_words
