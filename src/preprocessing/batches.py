"""
This file creates Batches. A Batch ca be described as data in compact form.
Batches allow parallel processing, enabling efficient use of hardware.
Dimensions/Shapes:
    Source Windows: Batch Size x (window size * 2 + 1)
    Target Windows: Batch Size x window size
    Target Labels:  Batch size
"""

import sys
import numpy as np


class MiniBatch:
    """
    Class MiniBatch that acts as a helper for class Batch.
    """
    def __init__(self, source, target, source_dictionary, target_dictionary, window):
        """
        Minibatch class constructor.
        :param source: String - Source language file path
        :param target: String - Target language file path
        :param source_dictionary: Dictionary - Loaded dictionary of source language
        :param target_dictionary: Dictionary - Loaded dictionary of target language
        :param window: Integer - Window size
        """
        self.sequence_e, self.sequence_f = self.get_sequence(source,
                                                             target,
                                                             source_dictionary,
                                                             target_dictionary,
                                                             window)
        self.mini_big_s = self.get_mini_big_s(self.sequence_f, target, window)
        self.mini_big_t = self.get_mini_big_t(self.sequence_e, target, window)
        self.mini_big_l = self.get_mini_big_l(self.sequence_e, window)

    def get_sequence(self, source, target, target_dictionary, source_dictionary, window):
        """
        Function that returns one sequence of a Batch.
        :param source: List - Source sentence
        :param target: List - Target sentence
        :param source_dictionary: Dictionary - Loaded dictionary of source language
        :param target_dictionary: Dictionary - Loaded dictionary of target language
        :param window: Integer - Window size
        :return: Array | Array - Source window sequence, Target window sequence
        """
        # creates an array where length equals the length of target and fills it with 0
        sequence_e = np.zeros(len(target), dtype=int)
        # creates an array where length equals the length of source and fills it with 0
        sequence_f = np.zeros(len(source), dtype=int)

        # changes entries of the array to the right keys of the given word
        # if a word is not in the dictionary the entry will be 1 -> unknown word
        count = 0
        for target_word in target:
            sequence_e[count] = target_dictionary.word_to_index(target_word)
            count += 1

        # changes entries of the array to the right keys of the given word
        # if a word is not in the dictionary the entry will be 0 -> unknown word
        count = 0
        for source_word in source:
            sequence_f[count] = source_dictionary.word_to_index(source_word)
            count += 1

        # adds start of sequence and end of sequence symbols
        for i in range(window):
            sequence_e = np.insert(sequence_e, 0, target_dictionary.word_to_index("<s>"))
        sequence_e = np.append(sequence_e, target_dictionary.word_to_index("</s>"))

        # adds start of sequence and end of sequence symbols
        for i in range(window - 1):
            sequence_f = np.insert(sequence_f, 0, target_dictionary.word_to_index("<s>"))

        for i in range(window + 1):
            sequence_f = np.append(sequence_f, target_dictionary.word_to_index("</s>"))

        return sequence_e, sequence_f

    def get_mini_big_s(self, sequence_f, target, window):
        """
        Function that transforms the sequence to window size
        :param sequence_f: Array - Sequence of source window (indexed)
        :param target: List - Target sentence (readable)
        :param window: Integer - Window size
        :return: Array - Source window
        """
        mini_big_s = np.zeros(((len(target) + 1), 2 * window + 1), dtype=int)

        for i in range(len(target) + 1):
            part_sequence_f = np.array(sequence_f[i: 2 * window + i + 1])

            if len(part_sequence_f) == 2 * window + 1:
                mini_big_s[i] = part_sequence_f
            else:
                while len(part_sequence_f) < 2 * window + 1:
                    part_sequence_f = np.append(part_sequence_f, [2])
                mini_big_s[i] = part_sequence_f

        return mini_big_s

    def get_mini_big_t(self, sequence_e, target, window):
        """
        Function that returns Mini batch target window.
        :param sequence_e: Array - Sequence of target window (indexed)
        :param target: List - Target window sentence (readable)
        :param window: Integer - Window size
        :return: Array - Target window
        """
        mini_big_t = np.zeros(((len(target) + 1), window), dtype=int)

        for i in range(len(target) + 1):
            part_sequence_e = np.array(sequence_e[i: window + i])
            mini_big_t[i] = part_sequence_e

        return mini_big_t

    def get_mini_big_l(self, sequence_e, window):
        """
        Function that transforms Array axis of target labels
        :param sequence_e: Array - Target label entries
        :param window: Integer - Window size
        :return: Array - Target label
        """
        for i in range(window):
            sequence_e = np.delete(sequence_e, 0)
        sequence_e = np.expand_dims(sequence_e, axis=1)

        return sequence_e


class Batch:
    """
    Class Batch that acts as a helper for class Batches.
    """
    def __init__(self, mini_batch, batch_size):
        """
        Batch Class constructor. Adjusts batches with given parameters.
        :param mini_batch: MiniBatch - Mini Batch entries
        :param batch_size: Integer - Amount of sequences(total lines) in a Batch
        """
        big_s = np.zeros((batch_size, len(mini_batch.mini_big_s[0])),
                         dtype=int)  # creates matrix only containing 0s of the needed size
        big_t = np.zeros((batch_size, len(mini_batch.mini_big_t[0])),
                         dtype=int)  # creates matrix only containing 0s of the needed size
        big_l = np.zeros((batch_size, 1), dtype=int)  # creates matrix only containing 0s of the needed size
        self.entries = (big_s, big_t, big_l)  # puts all matrices in one triple

    # updates every single matrix by line
    def update_batch_line(self, big_s_line, big_t_line, big_l_line, index):
        """
        Function that updates the line in a batch
        :param big_s_line: Array - Source window line
        :param big_t_line: Array - Target window line
        :param big_l_line: Array - Target label line
        :param index: Integer - Index of line
        :return: None
        """
        self.entries[0][index] = big_s_line
        self.entries[1][index] = big_t_line
        self.entries[2][index] = big_l_line


class Batches:
    """
    Class Batches that realises the creation of Batches.
    """
    def __init__(self, start_line, end_line, source, target, source_dict, target_dict, batch_size, window):
        """
        Batches Class constructor. Adjusts batches with given parameters.
        :param start_line: Integer - File line starting position
        :param end_line: Integer - File line end position
        :param source: List - Source language sequence
        :param target: List - Target language sequence
        :param source_dict: Dictionary - Loaded dictionary of source language
        :param target_dict: Dictionary - Loaded dictionary of target language
        :param batch_size: Integer - Amount of sequences(total lines) in a Batch
        :param window: Integer - Window size
        """
        full_target = target
        full_source = source

        self.target_dictionary = target_dict  # loaded directory of target
        self.source_dictionary = source_dict  # loaded directory of source

        self.total_mini_batches = []

        # appends every mini_batch from start_line to end_line
        for x in range(start_line, end_line + 1):
            target = full_target[x].split()  # splits the target so Mini_Batch is creatable
            source = full_source[x].split()  # splits the source so Mini_Batch is creatable

            temp_mini_batch = MiniBatch(source, target, self.target_dictionary, self.source_dictionary, window)
            # appends Mini_Batch to our array
            self.total_mini_batches = np.append(self.total_mini_batches, temp_mini_batch)

        self.total_batches = []

        count = 0
        batch = Batch(self.total_mini_batches[0], batch_size)

        # iterating through every mini_batch in total_mini_batches, so we can create Batches
        for mini_batch in self.total_mini_batches:
            for x in range(len(mini_batch.mini_big_s)):
                # checking if count is less than batch_size since Batches are size of batch_size
                # if it's less than our current batch, get an update
                # if not we append our current batch to our array and overwrite it by a new one and reset count
                if not count < batch_size:
                    self.total_batches.append(batch)
                    batch = Batch(self.total_mini_batches[0], batch_size)
                    count = 0

                batch.update_batch_line(mini_batch.mini_big_s[x],
                                        mini_batch.mini_big_t[x],
                                        mini_batch.mini_big_l[x],
                                        count)
                count += 1

        self.total_batches.append(batch)


def create_batches(source_path, target_path, dict_source, dict_target, batch_size):
    """
    Function that creates batches based on the batch size, source and target dictionary.
    :param dict_source: Dictionary - Loaded dictionary of source language
    :param dict_target: Dictionary - Loaded dictionary of target language
    :param source_path: String - BPE encoded source input path
    :param target_path: String - BPE encoded target input path
    :param batch_size: Integer - defines batch size
    :return: List | List | List - Source windows, Target windows, Target labels
    """
    with open(target_path, 'r', encoding='utf-8') as f:
        full_target = f.readlines()
    with open(source_path, 'r', encoding='utf-8') as f:
        full_source = f.readlines()

    batches = Batches(0, len(full_source) - 1, full_source, full_target, dict_source, dict_target, batch_size, 1)
    batches = batches.total_batches

    source_window = []
    target_window = []
    target_labels = []

    for batch in batches:
        source_window.append(np.array(batch.entries[0]))
        target_window.append(np.array(batch.entries[1]))
        target_labels.append(np.array(batch.entries[2]))

    return source_window, target_window, target_labels


def batches_to_string(source_dict, target_dict, batch_entries):
    """
    Function that translates batches indices to words.
    :param source_dict: Dictionary - Loaded dictionary of source language
    :param target_dict: Dictionary - Loaded dictionary of target language
    :param batch_entries: Tuple(3) List - Batch entries
    :return: List - Readable Batch entries
    """
    source_windows = []
    target_windows = []
    target_labels = []
    for batch in batch_entries:

        big_s_matrix = []
        for big_s_line in batch[0]:
            string_entry_line = []

            for entry in big_s_line:
                string_entry_line.append(source_dict.index_to_word(entry))

            big_s_matrix.append(string_entry_line)
        source_windows.append(big_s_matrix)

        big_t_matrix = []
        for big_t_line in batch[1]:
            string_entry_line = []

            for entry in big_t_line:
                string_entry_line.append(target_dict.index_to_word(entry))

            big_t_matrix.append(string_entry_line)
        target_windows.append(big_t_matrix)

        big_l_matrix = []
        for big_l_line in batch[2]:
            string_entry_line = []

            for entry in big_l_line:
                string_entry_line.append(target_dict.index_to_word(entry))

            big_l_matrix.append(string_entry_line)
        target_labels.append(big_l_matrix)

    return source_windows, target_windows, target_labels


def save_batch_to_file(name, output_path, batch_entries):
    """
    Function that writes created batch into a text file.
    :param name: String - File name
    :param output_path: String - File path
    :param batch_entries: Tuple(3) List - Source, target and labels entries of a batch
    :return: None
    """
    sys.stdout = open(f"{output_path}/{name}.txt", "w")

    source_window = batch_entries[0]
    target_window = batch_entries[1]
    target_labels = batch_entries[2]

    res = "\n".join("{} {} {}".format(x, y, z) for x, y, z in zip(source_window, target_window, target_labels))

    print(res)
    sys.stdout.close()


def save_batch_to_readable_file(name, output_path, batch_entries, source_dict, target_dict):
    """
    Function that saves a readable version of created batches into a text file.
    :param name: String, File name
    :param output_path: String, File path
    :param batch_entries: Tuple(3) List, Batch entries
    :param source_dict: Dictionary, Loaded dictionary of source language
    :param target_dict: Dictionary, Loaded dictionary of target language
    :return: None
    """
    sys.stdout = open(f"{output_path}/{name}.txt", "w")

    entries_readable = batches_to_string(source_dict, target_dict, batch_entries)

    save_batch_to_file(name, output_path, entries_readable)
