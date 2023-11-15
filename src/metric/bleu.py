"""
This file calculates BLEU (Bilingual Evaluation Understudy),
a metric that scores machine-translations between two natural languages.
"""

import numpy as np


def text_to_list(text):
    """
    Function that transform a string sentence into a list.
    :param text: String - Sentence
    :return: List - Split sentence
    """
    total_sets = []

    for lines in text:
        total_sets.append(lines.split())

    return total_sets


def n_gram(ref, hyp, n):
    """
    Function that counts the amount of n_gram matches in a sentence between the reference and hypothesis.
    An n-gram is a sequence of n consecutive words.
    :param ref: List - Reference sentence
    :param hyp: List - Hypothesis sentence
    :param n: Integer - Sequence length necessary for a match
    :return: Integer - Amount of n-gram matches
    """
    matches = 0
    used_n_grams = {}

    for x in range(len(hyp)):
        temp_hyp = hyp[x:n + x]

        if len(temp_hyp) == n:
            for y in range(len(ref)):
                temp_ref = ref[y:n + y]

                if not np.array_equal(temp_ref, used_n_grams.get(y)):
                    if len(temp_ref) == n:

                        if np.array_equal(temp_hyp, temp_ref):
                            matches += 1

                            used_n_grams.update({y: temp_ref})

    return matches


def modified_n_gram(pairs, n):
    """
    Function that calculates the average of all n-gram matches by the total number of all n-grams.
    :param pairs: List - Pairs of (reference, hypothesis) sentences
    :param n: Integer - Sequence length necessary for a match
    :return: Float - Average of all n-gram matches by the total number of all n-grams
    """
    counter = 0
    denominator = 0

    for pair in pairs:
        ref = pair[0]
        hyp = pair[1]
        n_gram_hyp = 0

        for x in range(len(hyp)):
            temp_hyp = hyp[x:n + x]

            if (len(temp_hyp)) == n:
                n_gram_hyp += 1

        counter += min(
            n_gram(ref, hyp, n),
            n_gram_hyp
        )
        denominator += n_gram_hyp

    result = counter / denominator

    return result


def brevity_penalty(ref_len, hyp_len):
    """
    Function that adds a brevity penalty if the hypothesis is too short.
    :param ref_len: Integer - Length of the reference
    :param hyp_len: Integer - Length of the hypothesis
    :return: Float - Brevity Penalty
    """
    if hyp_len > ref_len:
        return 1
    else:
        return np.exp(1 - ref_len / hyp_len)


def bleu_n4(ref, hyp):
    """
    Function that calculates the Bleu metric with N=4.
    :param ref: List - Reference sentences
    :param hyp: List - Hypothesis sentences
    :return: Float - Bleu score in % (rounded to 2nd decimal)
    """
    pairs = []
    temp = 0
    hyp_len = 0
    ref_len = 0

    for x in range(len(ref)):
        pairs.append((ref[x], hyp[x]))

    for n in range(1, 5):
        temp += 1 / 4 * np.log(modified_n_gram(pairs, n))

    for line in hyp:
        hyp_len += len(line)

    for line in ref:
        ref_len += len(line)

    # result in %, rounded on the 2nd decimal
    result = np.round(100 * brevity_penalty(ref_len, hyp_len) * np.exp(temp), 2)
    return result


def calculate_bleu(file_path_ref, file_path_hyp):
    """
    Function that calculates the bleu score of a hypothesis and a reference.
    :param file_path_ref: String - Reference file path
    :param file_path_hyp: String - Hypothesis file path
    :return: Float - Bleu score in % (rounded to 2nd decimal)
    """
    print('Calculating Bleu...')

    ref = open(file_path_ref, 'r', encoding="utf-8")
    hyp = open(file_path_hyp, 'r', encoding="utf-8")

    total_ref_sets = text_to_list(ref)
    total_hyp_sets = text_to_list(hyp)

    bleu_score = bleu_n4(total_ref_sets, total_hyp_sets)

    print(f'The BLEU score is {bleu_score}%.')

    return bleu_score
