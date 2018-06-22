"""
Gaussian Mixture Model
----------------------

Clustering method on BOW vectors

Public Methods
~~~~~~~~~~~~~~
    train_model

"""

from __future__ import print_function

from time import time

import numpy as np
from sklearn import mixture
from sklearn.feature_extraction.text import TfidfVectorizer


def print_occurences(counts_df):
    print(counts_df.sort_values(by='occurrences', ascending=False).head(20))


class CustomPreprocessor(object):
    def __call__(self, doc):
        return doc


class CustomTokenizer(object):
    def __call__(self, doc):
        return doc


def aggregate_by_topic_prediction(mixm_test, data_samples):
    predictions = np.argmax(mixm_test, axis=1)
    n_topics = 10
    aggregated_by_topic = [[] for _ in range(n_topics)]
    for i, topic_id in enumerate(predictions):
        aggregated_by_topic[topic_id].append(data_samples[i])
    return aggregated_by_topic


def train_tfidf(data_samples):
    # Use tf (raw term count) features for LDA.
    print("Extracting tf features ...")

    tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=5,  # found in 90% and found onnly 5 documents are ignored
                                       tokenizer=CustomTokenizer(),
                                       preprocessor=CustomPreprocessor())
    t0 = time()

    tfidf_vectorizer.fit(data_samples)
    tfidf_trained = tfidf_vectorizer.transform(data_samples)
    print("done in %0.3fs." % (time() - t0))
    return tfidf_vectorizer, tfidf_trained


def train_model(data_samples, test_dataset):
    n_topics = 10

    tfidf_vectorizer, tfidf_trained = train_tfidf(data_samples)

    print("Fitting Mixture model with tf features, "
          "n_samples=%d"
          % (len(data_samples)))

    mix_model = mixture.BayesianGaussianMixture(n_components=n_topics, covariance_type='full', init_params='kmeans',
                                                weight_concentration_prior=0.01)

    t0 = time()
    mix_model.fit(tfidf_trained.toarray())
    print("done in %0.3fs." % (time() - t0))

    tf_test = tfidf_vectorizer.transform(test_dataset)

    mixm_test = mix_model.predict_proba(tf_test.toarray())

    return {
        'transformations': mixm_test,
        'features': tfidf_vectorizer.get_feature_names(),
        '_model': mix_model,
        '_tfidf_vectorizer': tfidf_vectorizer,
        '_tfidf_transformations': tf_test
    }
