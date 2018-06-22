"""
Statistical Analysis of words and documents
-------------------------------------------

Mainly ChiSquare Stats
"""
import csv

from scipy.sparse import coo_matrix
from scipy.stats import chisquare
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from nlpkit.tokenizer import cached_or_tokenize_lazy_corpus


class CustomPreprocessor(object):
    def __call__(self, doc):
        return doc


class CustomTokenizer(object):
    def __call__(self, doc):
        return doc


def chisquare_words(data_samples, selected_words, metas_param):
    df = (5, 0.90)
    tf_vectorizer = CountVectorizer(
        min_df=df[0], max_df=df[1],
        tokenizer=CustomTokenizer(),
        preprocessor=CustomPreprocessor())

    tf_vectorizer.fit(data_samples)
    words = tf_vectorizer.get_feature_names()
    nbr_of_words = len(words)

    tf_trained = tf_vectorizer.transform(data_samples)
    N = tf_trained.shape[0]

    # THIS IS A CHEAT TO CALCULATE META WEIGHTS
    metas = list(set(metas_param))
    nbr_of_metas = len(metas)

    # Here, term occurences is the same as document occurences containing
    # word because a word is present or not in document
    t_counts = [0] * nbr_of_words
    count_t_m = [[0] * nbr_of_metas for _ in [0] * nbr_of_words]
    coo_tf_trained = coo_matrix(tf_trained)
    for i, j, df in zip(coo_tf_trained.row, coo_tf_trained.col, coo_tf_trained.data):
        # print("document = %d, term = %d, tf = %s" % (i, j, df))
        t_counts[j] += 1  # counting documents for every word
        meta_index = metas.index(metas_param[i])
        count_t_m[j][meta_index] += 1  # counting metas for every word

    freq_t_m = []
    for t_index, count_m_given_t in enumerate(count_t_m):
        # summing all the metas counts for each words
        total_count_m_given_t = sum(count_m_given_t)
        # Normalizing the ordered metas counts
        freq_m_given_t = []
        for occ in count_m_given_t:
            freq_m_given_t.append(occ / total_count_m_given_t)
        freq_t_m.append(freq_m_given_t)

    words_freq_means = np.mean(freq_t_m, axis=0)

    # words_count_means = np.mean(count_t_m, axis=1)

    # find index of selected words
    words_indexes = [words.index(selected_word) for selected_word in selected_words]
    # select distribution for these indexes
    selected_words_distributions = [freq_t_m[selected_word_index] for selected_word_index in words_indexes]
    # selected_words_means = [words_count_means[selected_word_index] for selected_word_index in words_indexes]

    # chisquare_results = [chisquare(word_distribution, f_exp=[1/nbr_of_metas]*nbr_of_metas)
    #                      for i, word_distribution in enumerate(selected_words_distributions)]

    chisquare_results = []
    distribution_len = len(selected_words_distributions[0])
    bigger_part_len = 2
    smaller_part_len = distribution_len-bigger_part_len
    for i, word_distribution in enumerate(selected_words_distributions):
        sorted_distribution = sorted(word_distribution)
        sorted_distribution.reverse()
        bigger_part = sorted_distribution[0:bigger_part_len]
        smaller_part = sorted_distribution[bigger_part_len:distribution_len]
        bigger_exp = sum(bigger_part)
        smaller_exp = sum(smaller_part)
        chisquare_result = chisquare([bigger_exp, smaller_exp],
                                     f_exp=[bigger_part_len/distribution_len, smaller_part_len/distribution_len])
        chisquare_results.append(chisquare_result)

    return chisquare_results