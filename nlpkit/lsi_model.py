"""
Latent Semantic Indexing Model
------------------------------

BOW vectors dimension reduction with LSI

Public Methods
~~~~~~~~~~~~~~
    train_model

"""

from __future__ import print_function
from time import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer

class CustomPreprocessor(object):
    def __call__(self, doc):
        return doc


class CustomTokenizer(object):
    def __call__(self, doc):
        return doc


df = (5, 0.90)
default_tf_vectorizer = CountVectorizer(min_df=df[0], max_df=df[1],  # found in 90% and found onnly 5 documents are ignored
                                    tokenizer=CustomTokenizer(),
                                    preprocessor=CustomPreprocessor())


default_tfidf_vectorizer = TfidfVectorizer(min_df=df[0], max_df=df[1],
                                           # found in 90% and found onnly 5 documents are ignored
                                           tokenizer=CustomTokenizer(),
                                           preprocessor=CustomPreprocessor())


def train_model(data_samples, test_dataset, vectorizer=default_tfidf_vectorizer, n_components=50):

    n_samples = len(data_samples)

    # Use tfidf (raw term count) features for LSI.
    print("Extracting tfidf features for LSI...")

    t0 = time()
    vectorizer.fit(data_samples)
    tf = vectorizer.transform(data_samples)
    print("done in %0.3fs." % (time() - t0))
    print("Fitting LSI models with tf features, "
          "n_samples=%d"
          % n_samples)

    lsi_model = TruncatedSVD(n_components=n_components, n_iter=10, random_state=42)

    t0 = time()
    lsi_model.fit(tf)
    print("done in %0.3fs." % (time() - t0))

    # https://stats.stackexchange.com/questions/171539/percentage-of-variation-in-each-column-explained-by-each-svd-mode
    # https://stats.stackexchange.com/questions/184603/in-pca-what-is-the-connection-between-explained-variance-and-squared-error
    # Proportion of explained variance is 1 - Error^2 when we reconstruct
    print(lsi_model.explained_variance_ratio_)
    print(lsi_model.explained_variance_ratio_.sum())

    tf_test = vectorizer.transform(test_dataset)
    lsi_test = lsi_model.transform(tf_test)
    return {
        'transformations': lsi_test,
        'features': vectorizer.get_feature_names(),
        '_model': lsi_model,
        '_tfidf_vectorizer': vectorizer,
        '_tfidf_transformations': tf_test,
        '_name': vectorizer.__class__.__name__.lower() + '_lsi' + str(n_components)

    }