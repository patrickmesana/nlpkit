"""
Term Frequency Inverse Document Vectorizer
------------------------------------------

Wrapper around SciPy TFIDF Vectorizer

Public Methods
~~~~~~~~~~~~~~
    train_model

"""

from __future__ import print_function
from sklearn.feature_extraction.text import TfidfVectorizer


class CustomPreprocessor(object):
    def __call__(self, doc):
        return doc


class CustomTokenizer(object):
    def __call__(self, doc):
        return doc


def train_model(data_samples, test_dataset):
    df = (5, 0.90)
    tfidf_vectorizer = TfidfVectorizer(
        min_df=df[0], max_df=df[1],
                                       # found in 90% and found onnly 5 documents are ignored
                                       tokenizer=CustomTokenizer(),
                                       preprocessor=CustomPreprocessor())

    tfidf_vectorizer.fit(data_samples)
    tf_trained = tfidf_vectorizer.transform(test_dataset)

    return {
        'features': tfidf_vectorizer.get_feature_names(),
        'transformations': tf_trained,
        '_model': tfidf_vectorizer,
        '_name': 'tfidf'
    }
