"""
Custom Vectorizer based on MI
-----------------------------

Using mutual information between metas and words to weight words

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


class MIWeightsVectorizer(object):
    def __init__(self, metas):
        df = (5, 0.90)
        self.tf_vectorizer = CountVectorizer(
            min_df=df[0], max_df=df[1],
            tokenizer=CustomTokenizer(),
            preprocessor=CustomPreprocessor())

        # THIS IS A CHEAT TO CALCULATE meta WEIGHTS
        self.metas = metas
        self.metas = list(set(self.metas))
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

        # THIS IS A CHEAT TO CALCULATE meta WEIGHTS
        metas = list(set(metas))
        nbr_of_metas = len(metas)

        # Here, term occurences is the same as document occurences containing
        # word because a word is present or not in document
        t_counts = [0] * self.nbr_of_words
        count_t_m = [[0] * nbr_of_metas for _ in [0] * self.nbr_of_words]
        coo_tf_trained = coo_matrix(tf_trained)
        for i, j, df in zip(coo_tf_trained.row, coo_tf_trained.col, coo_tf_trained.data):
            # print("document = %d, term = %d, tf = %s" % (i, j, df))
            t_counts[j] += 1  # counting documents for every word
            meta_index = metas.index(metas[i])
            count_t_m[j][meta_index] += 1  # counting metas for every word

        # counts metas
        m_counts = [0] * nbr_of_metas
        for t_m in count_t_m:
            for m_index, m in enumerate(t_m):
                m_counts[m_index] += m

        total_count_m = sum(m_counts)
        m_frequencies = [m_count / total_count_m for m_count in m_counts]
        m_entropy = - sum([m_frequency * math.log(m_frequency) for m_frequency in m_frequencies])

        t_metas_centropy = []
        for t_index, count_m_given_t in enumerate(count_t_m):
            # summing all the metas counts for each words
            total_count_m_given_t = sum(count_m_given_t)
            # Normalizing the ordered metas counts
            # The +1 is to forbid the division by 0 but also penalize the words with to low frequencies
            freq_m_given_t = []
            for occ in count_m_given_t:
                freq_m_given_t.append(occ / total_count_m_given_t)

            info_m_given_t = []
            for freq_m in freq_m_given_t:
                if freq_m == 0:
                    info_m_given_t.append(0)
                else:
                    info_m_given_t.append(- freq_m * math.log(freq_m))
            m_given_t_entropy = sum(info_m_given_t)
            mutual_information = m_entropy - m_given_t_entropy

            if (mutual_information <0):
                # print(self.words[t_index])
                mutual_information = 0

            # Creating a new (word -> meta) normalized counts matrix
            t_metas_centropy.append(mutual_information)

        # helps see the weights of mutual_information
            # # helps see the weights of idf
            # sorted_indexed_idf = np.argsort(t_metas_centropy)[::-1]
            # sorted_word_idf = [[self.words[i], t_metas_centropy[i]] for i in sorted_indexed_idf.tolist()]
            # sorted_word_idf_frame = DataFrame(sorted_word_idf)
            # with open('./output/tfmut_values.txt', 'w') as f:
            #     f.write(sorted_word_idf_frame.to_string())

        row = []
        col = []
        data = []
        for i, j, df in zip(coo_tf_trained.row, coo_tf_trained.col, coo_tf_trained.data):
            # print("row = %d, column = %d, value = %s" % (i, j, df))
            row.append(i)
            col.append(j)
            word_mi = t_metas_centropy[j]
            data.append(word_mi)

        return csr_matrix((np.array(data), (np.array(row), np.array(col))), shape=(N, self.nbr_of_words))

    def get_feature_names(self):
        return self.words


def train_model(data_samples, test_dataset, metas):
    vectorizer = MIWeightsVectorizer()
    vectorizer.fit(data_samples)
    tfidf_trained = vectorizer.transform(test_dataset, metas)

    return {
        'features': vectorizer.get_feature_names(),
        'transformations': tfidf_trained,
        '_model': vectorizer,
        '_name': 'mutual_info'
    }
