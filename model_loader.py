"""
Helpers to load cached model or train model
-------------------------------------------

"""

from nlpkit.utils import do_or_load_pickle
from functools import partial


def cached_or_train(training_method, lazy_training_corpus, model_name, should_overwrite=False):
    return cached_or_train_and_test(training_method,
                                    lazy_training_corpus,
                                    lazy_training_corpus,
                                    model_name,
                                    should_overwrite)


def cached_or_train_and_test(training_method,
                             lazy_training_corpus,
                             lazy_testing_corpus,
                             model_name, should_overwrite=False):
    return do_or_load_pickle(model_name,
                             lambda: training_method(lazy_training_corpus(), lazy_testing_corpus()),
                             should_overwrite)
