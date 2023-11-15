"""
This file is used for preprocessing training data using sub-word units.
Sub-words denote groups of letters that can be smaller than complete words.
BPE (Byte-Pair Encoding) is a decomposition strategy for sub-words based on occurrences.
"""

import time
import pickle
import numpy as np


def create_bpe_model(model_name, save_path, n, file_path_train_data, file_path_train_data_two=None):
    """
    Function that creates a BPE model of either one language or two languages as a combined version.
    The amount of operations affect the vocabulary size.
    :param model_name: String - BPE model name
    :param save_path: String - Output path of created BPE model
    :param n: Integer - BPE operations
    :param file_path_train_data: String - File path training data
    :param file_path_train_data_two: String - File path training data 2 (optional, necessary for combined BPE)
    :return: None
    """
    start_time = time.time()
    lines_list = []

    # two paths necessary to create combined BPE (e.g., english and german)
    if not file_path_train_data_two:
        file = open(file_path_train_data, encoding="utf8")
        lines_list.extend(file.readlines())
        file.close()
    else:
        for file_path in [file_path_train_data, file_path_train_data_two]:
            file = open(file_path, encoding="utf8")
            lines_list.extend(file.readlines())
            file.close()

    # create frequency dictionary:
    freq_dict = {}
    for line in lines_list:
        words_of_line = line.split()

        for word in words_of_line:

            if word in freq_dict:
                freq_dict[word]["number"] += 1

            # create new entry with number = 1
            else:
                freq_dict.update({word: {"number": 1}})

    # encode the words
    # store the disassembled word in freq_dict
    for w in freq_dict:
        disassembled_word = list(w)

        # pushing the word-breaker to the last char of a word
        disassembled_word.append(disassembled_word.pop() + "</w>")
        freq_dict[w]['disassembled'] = disassembled_word

    # iterate over n and create the highest frequent pair
    # note: difference between freq_dict vs. frequency_table
    # former one are all words with their frequency, latter is pair frequency
    result = []
    for r in range(n):

        # create frequency table
        frequency_table = {}

        for word in freq_dict:
            dis_word = freq_dict[word]['disassembled']

            for i in range(len(dis_word) - 1):
                pair = dis_word[i] + dis_word[i + 1]

                # if pair is ready in frequency_table, add number of words with pair
                if pair in frequency_table:
                    frequency_table[pair] += freq_dict[word]['number']
                else:
                    frequency_table.update({pair: freq_dict[word]['number']})

        # get table entry with the highest score
        # if more than one, pick in alphabetic order
        highest_value = max(frequency_table.values())
        highest_frequent_pairs = [k for k, v in frequency_table.items() if float(v) == highest_value]
        highest_frequent_pairs = sorted(highest_frequent_pairs)
        highest_frequent_pair = highest_frequent_pairs[0]
        result.append(highest_frequent_pair)

        # create new temp for all words and for each word.
        for word in freq_dict:
            dis_word = freq_dict[word]['disassembled']

            freq_dict[word]['disassembled'] = merge_pair(highest_frequent_pair, dis_word)

    file_output = open(f"{save_path}/{model_name}", "wb")
    clear_file(file_output)
    pickle.dump(result, file_output)
    file_output.close()
    end_time = time.time()
    print(f"Finished for {n} Operations. Needed Time: {end_time - start_time}.")


def encode_data(file_path_bpe_model, file_path_input, file_path_output=None):
    """
    Function that encodes a text file with BPE.
    :param file_path_bpe_model: String - BPE model file path
    :param file_path_input: String - Input file path (File to encode)
    :param file_path_output: String - Output file path (optional, necessary to create BPE encoded file)
    :return: String - BPE encoded text
    """
    file_input = open(file_path_input, encoding="utf8")
    lines_list = file_input.readlines()
    file_input.close()

    result = encode_data_string(file_path_bpe_model, lines_list)

    if file_path_output:
        file_output = open(file_path_output, "a", encoding="utf8")
        clear_file(file_output)
        file_output.write(result)
        file_output.close()

    return result


def encode_data_string(file_path_bpe_model, sentences):
    """
    Function that encodes an array of sentences with BPE.
    :param file_path_bpe_model: String - BPE model file path
    :param sentences: Array - String sentences
    :return: Array - BPE encoded sentences
    """
    file_model = open(file_path_bpe_model, "rb")
    model = pickle.load(file_model)
    file_model.close()
    result = ""
    model = np.array(model)
    for line in sentences:

        for word in line.split():
            dis_word = list(word)
            dis_word[-1] += "</w>"

            for m in model:
                dis_word = merge_pair(m, dis_word)

            for d in range(len(dis_word) - 1):
                result += dis_word[d] + "@@ "

            result += dis_word[-1] + " "

        result = result[:-1] + "\n"

    return result


def decode_data(file_path_input, file_path_output):
    """
    Function that decodes a text file with BPE.
    :param file_path_input: String - Input file path
    :param file_path_output: String - Output file path
    :return: None
    """
    file_input = open(file_path_input, encoding="utf8")
    lines_list = file_input.readlines()
    file_input.close()

    file_output = open(file_path_output, "a", encoding="utf8")

    temp_res = decode_data_string(lines_list)
    res = ""

    for line in temp_res:
        res += line

    clear_file(file_output)
    file_output.write(res)
    file_output.close()


def decode_data_string(sentences):
    """
    Function that decodes a list of sentences with BPE.
    :param sentences: List - String sentences
    :return: List - BPE decoded String sentences
    """
    result = []

    for line in sentences:
        line = line.replace('@@ ', '')
        line = line.replace('@@', '')
        line = line.replace('</w> ', ' ')
        line = line.replace('</w>', ' ')
        line = line.replace('<UNK>', ' ')
        line = line.replace('</s>', '')
        line = line.replace('<s>', '')
        result.append(line)

    return result


def merge_pair(pair, word):
    """
    Function that merges BPE pairs.
    :param pair: String - Highest frequency BPE pair
    :param word: List - Disassembled String word
    :return: List - Updated disassembled String word
    """
    i = 0
    new_dis_word = []

    while i < len(word) - 1:

        if pair == word[i] + word[i + 1]:
            new_dis_word.append(pair)
            i += 2
        else:
            new_dis_word.append(word[i])
            i += 1

    if i == len(word) - 1:  # append "last_char</w>" if not part of highest_frequent_pair
        new_dis_word.append(word[-1])

    return new_dis_word


def clear_file(file):
    """
    Function that clears the entire content of a file.
    :param file: String - File input
    :return: None
    """
    file.seek(0)
    file.truncate()


def make_bpe_model_readable(file_path_bpe_model, file_path_output):
    """
    Function that makes a BPE model human-readable.
    :param file_path_bpe_model - String, BPE model file path
    :param file_path_output - String, Output file path
    :return: None
    """
    file_input = open(file_path_bpe_model, "rb")
    model = pickle.load(file_input)
    file_input.close()

    file_output = open(file_path_output, "a", encoding="utf8")

    res = ""
    for m in model:
        res += m + "\n"

    clear_file(file_output)
    file_output.write(res)
    file_output.close()
