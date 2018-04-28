"""
Term Frequency Vectorizer
-------------------------

Uses SciPy text feature extraction to count words and create a vocabulary

Public Methods
~~~~~~~~~~~~~~
    train_model

"""

from __future__ import print_function

import numpy as np
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from typing import List
from functools import partial
from nlpkit.model_loader import cached_or_train
from nlpkit.tokenizer import cached_or_tokenize_lazy_corpus


class CustomPreprocessor(object):
    def __call__(self, doc):
        return doc


class CustomTokenizer(object):
    def __call__(self, doc):
        return doc


def train_model(data_samples, test_dataset):
    df = (5, 0.90)
    tf_vectorizer = CountVectorizer(min_df=df[0], max_df=df[1],  # found in 90% and found onnly 5 documents are ignored
                                    tokenizer=CustomTokenizer(),
                                    preprocessor=CustomPreprocessor())

    tf_vectorizer.fit(data_samples)
    tf_trained = tf_vectorizer.transform(test_dataset)

    return {
        'features': tf_vectorizer.get_feature_names(),
        'transformations': tf_trained,
        '_model': tf_vectorizer,
        '_name': 'tf'
    }


def summary(features, transformations, output_name='tf'):
    weights = np.asarray(transformations.sum(axis=0)).ravel().tolist()
    weights_df = DataFrame({'term': features, 'weight': weights})
    tfidf_summary = weights_df.sort_values(by='weight', ascending=False).head(100)
    with open('./output/' + output_name + '_summary.txt', 'w') as f:
        f.write(tfidf_summary.to_string())


def tf_dump_with_strategy(
        model_method,
        strategy_name="b",
        strategy="basic",
        doc_filter=None,
        model_name="tf",
        override_cached_model=True):
    return cached_or_train(
        model_method,
        partial(cached_or_tokenize_lazy_corpus,
                strategy_name=strategy_name,
                strategy=strategy,
                doc_filter=doc_filter),
        model_name=strategy_name + "_" + model_name + "_model",
        should_overwrite=override_cached_model
    )


def summarize_dump(strategy_name="b", strategy="basic",
                   model_name="tf"
                   ):
    tf_dump = tf_dump_with_strategy(train_model,
                                    strategy_name=strategy_name,
                                    strategy=strategy,
                                    model_name=model_name)
    summary(tf_dump['features'], tf_dump['transformations'], model_name)
