"""
Binary TF-IDF Vectorizer
------------------------

Implements a custom TF-IDF formula based on binary values and create a weighted vector for every document

Public Classes
~~~~~~~~~~~~~~
    BinaryTFIDFVectorizer

Public Methods
~~~~~~~~~~~~~~
    train_model

"""

from __future__ import print_function

import math
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np



class CustomPreprocessor(object):
    def __call__(self, doc):
        return doc


class CustomTokenizer(object):
    def __call__(self, doc):
        return doc


class BinaryTFIDFVectorizer(object):
    """Contains the logic of the Vectorizer using binary TFIDF

    Attributes
    ----------
        metas : list
            the list of metas, this should be passed as parameters
        nbr_of_metas : int
        words : list
            vocab
        nbr_of_words : list
    """
    def __init__(self, metas):
        df = (5, 0.90)
        self.tf_vectorizer = CountVectorizer(
            min_df=df[0], max_df=df[1],
            tokenizer=CustomTokenizer(),
            preprocessor=CustomPreprocessor())

        self.metas = list(set(metas))
        self.nbr_of_metas = len(self.metas)
        self.words = None
        self.nbr_of_words = None

    def fit(self, data_samples):
        self.tf_vectorizer.fit(data_samples)
        self.words = self.tf_vectorizer.get_feature_names()
        self.nbr_of_words = len(self.words)

    def transform(self, test_dataset, metas):
        tf_trained = self.tf_vectorizer.transform(test_dataset)
        N = tf_trained.shape[0]

        # THIS IS A CHEAT TO CALCULATE METAS WEIGHTS
        print("loading metas...")
        uniq_metas = list(set(metas))
        nbr_of_metas = len(uniq_metas)

        words_docs_occurences = [0] * self.nbr_of_words
        words_metas_occurences = [[0] * nbr_of_metas for _ in [0] * self.nbr_of_words]
        coo_tf_trained = coo_matrix(tf_trained)
        for i, j, df in zip(coo_tf_trained.row, coo_tf_trained.col, coo_tf_trained.data):
            # print("document = %d, term = %d, tf = %s" % (i, j, df))
            words_docs_occurences[j] += 1
            meta_index = uniq_metas.index(metas[i])
            words_metas_occurences[j][meta_index] += 1

        words_metas_occurences_sorted = []
        for word_metas_occurences in words_metas_occurences:
            word_metas_occurences_total = sum(word_metas_occurences)
            word_metas_occurences.sort(reverse=True)
            # The +1 is to forbid the division by 0 but also penalize the words with to low frequencies
            word_metas_occurences = [occ / (word_metas_occurences_total + 1) for occ in word_metas_occurences]
            words_metas_occurences_sorted.append(word_metas_occurences)

        idf = [- math.log(doc_occ / N) for k, doc_occ in enumerate(words_docs_occurences)]

        # helps see the weights of idf
        sorted_indexed_idf = np.argsort(idf)[::-1]
        sorted_word_idf = [[self.words[i], idf[i]] for i in sorted_indexed_idf.tolist()]

        # from pandas import DataFrame
        # sorted_word_idf_frame = DataFrame(sorted_word_idf)
        # with open('./output/tfidf_values.txt', 'w') as f:
        #     f.write(sorted_word_idf_frame.to_string())

        row = []
        col = []
        data = []
        for i, j, df in zip(coo_tf_trained.row, coo_tf_trained.col, coo_tf_trained.data):
            # print("row = %d, column = %d, value = %s" % (i, j, df))
            row.append(i)
            col.append(j)
            word_tfidf = idf[j]
            data.append(word_tfidf)

        return csr_matrix((np.array(data), (np.array(row), np.array(col))), shape=(N, self.nbr_of_words))

    def get_feature_names(self):
        return self.words


def train_model(data_samples, test_dataset, metas):
    vectorizer = BinaryTFIDFVectorizer()
    vectorizer.fit(data_samples)
    tfidf_trained = vectorizer.transform(test_dataset, metas)

    return {
        'features': vectorizer.get_feature_names(),
        'transformations': tfidf_trained,
        '_model': vectorizer,
        '_name': 'mutual_info'
    }
