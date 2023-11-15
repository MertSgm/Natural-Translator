"""
This file creates the dictionary for the neural network.
The dictionary enables the work with indices instead of words.
"""

import pickle


class Dictionary:
    """
    Class Dictionary that realizes a bidirectional mapping between Strings and indices of a vocabulary.
    """
    def __init__(self, name):
        """
        Dictionary class constructor.
        :param name: String - Name of the dictionary
        """
        self.name = name  # name to refer to our Vocabulary object.
        self.wordToIndex = {}  # to hold the words to their index values
        self.wordToCount = {}  # to hold individual word counts
        self.indexToWord = {}  # reverse wordToIndex
        self.num_words = 0  # count of the number of words
        self.add_word("<PAD>")  # adds padding symbol, index 0
        self.add_word("<UNK>")  # adds unknown word symbol, index 1
        self.add_word("<s>")  # adds start of sequence symbol, index 2
        self.add_word("</s>")  # adds end of sequence symbol, index 3

    # Adding words to the dictionary
    def add_word(self, word):
        # Word doesn't exist
        """
        Function that adds a String to the dictionary as an index.
        :param word: String - Word
        :return: None
        """
        if word not in self.wordToIndex:
            # First entry of word into vocabulary
            self.wordToIndex[word] = self.num_words
            self.wordToCount[word] = 1
            self.indexToWord[self.num_words] = word
            self.num_words += 1
        else:
            # Word exists; increase word count
            self.wordToCount[word] += 1

    def add_sentence(self, sentence):
        """
        Function that adds the words of a Sentence to the dictionary as indexes.
        :param sentence: List - String sentence
        :return: None
        """
        for word in sentence.split():
            self.add_word(word)

    # look up a word through its index
    def index_to_word(self, index):
        """
        Function that translates an index in the dictionary to its word.
        :param index: Integer - Dictionary index
        :return: String - Word of index
        """
        try:
            return self.indexToWord[index]
        except:
            print("Error: Some index doesn't exist in our vocabulary!")

    # look up an index through its word
    def word_to_index(self, word):
        """
        Function that translates a word in the dictionary to its index.
        :param word: String - Word
        :return: Integer - Index of word
        """
        try:
            return self.wordToIndex[word]
        except:
            print("Error: Some word doesn't exist in our vocabulary!")
            return self.wordToIndex["<UNK>"]

    def get_size(self):
        """
        Function that returns the dictionary size.
        :return: Dictionary length
        """
        return len(self.indexToWord)


def load_dictionary(file_path_dictionary):
    """
    Function that loads the dictionary.
    :param file_path_dictionary: String - Dictionary file path
    :return: Dictionary - pickle load dictionary
    """
    return pickle.load(open(file_path_dictionary, "rb"))


def create_dictionary(name, file_path_input):
    """
    Function that creates a dictionary from a BPE encoded input file.
    :param name: String - Dictionary name
    :param file_path_input: String - Input file path (BPE encoded)
    :return: None
    """
    dictionary = Dictionary(name)
    lines = (open(file_path_input, encoding="utf8").readlines())

    for line in lines:
        words = line.split()
        for word in words:
            dictionary.add_word(word)

    pickle.dump(dictionary, open(f"res/dictionaries/{name}.pkl", "wb"))
