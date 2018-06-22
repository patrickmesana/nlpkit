"""
Latent Dirichlet Allocation Model
---------------------------------

Topic extraction with Latent Dirichlet Allocation

Public Methods
~~~~~~~~~~~~~~
    train_model

"""

from __future__ import print_function
from time import time

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
import lda
import pyLDAvis
import pyLDAvis.sklearn
import pyLDAvis.gensim

def hellinger_distance(dense1, dense2):
    return np.sqrt(0.5 * sum((np.sqrt(dense1) - np.sqrt(dense2)) ** 2))


def print_closest_documents(selected_document_idx, documents_distributions, original_dataset):
    selected_document_raw = original_dataset[selected_document_idx]
    selected_document_dense = documents_distributions[selected_document_idx]
    distances = []
    idx = 0
    nbr_of_top_documents = 5
    for dd in documents_distributions:
        dd_distance = hellinger_distance(selected_document_dense, dd)
        distances.append([dd_distance, idx])
        idx += 1
    sorted_selected_document_distances = sorted(distances, key=lambda x: x[0])
    top_documents = sorted_selected_document_distances[0:nbr_of_top_documents]
    selected_documents_indexes = [row[1] for row in top_documents]
    selected_documents_indexes = map(int, selected_documents_indexes)
    closest_documents_from_sample = [original_dataset[i] for i in selected_documents_indexes]
    print(selected_document_raw)
    print(closest_documents_from_sample)


def build_lda_model(lda_impl, n_iter, n_topics):
    if lda_impl == 'sklearn':
        return LatentDirichletAllocation(n_topics=n_topics, max_iter=n_iter,
                                         # alpha, low means a document contains just a few mixture of topics
                                         doc_topic_prior=0.0001,
                                         # beta, low means a topic contains just a few mixture of words
                                         topic_word_prior=0.0001)

    if lda_impl == 'lda':
        return lda.LDA(n_topics=n_topics, n_iter=n_iter, alpha=0.001)

    raise Exception('Undefined LDA Implementation')

class CustomPreprocessor(object):
    def __call__(self, doc):
        return doc


class CustomTokenizer(object):
    def __call__(self, doc):
        return doc


def train_model(data_samples, test_dataset, impl_name='sklearn'):
    n_topics = 100
    n_top_words = 10

    n_samples = len(data_samples)

    # Use tf (raw term count) features for LDA.
    print("Extracting tf features for LDA...")

    tf_vectorizer = CountVectorizer(tokenizer=CustomTokenizer(),
                                    preprocessor=CustomPreprocessor())
    t0 = time()
    tf = tf_vectorizer.fit_transform(data_samples)
    print("done in %0.3fs." % (time() - t0))

    tf_feature_names = tf_vectorizer.get_feature_names()

    print("Fitting LDA models with tf features, "
          "n_samples=%d"
          % (n_samples))

    n_iter = 2000

    lda_model = build_lda_model(impl_name, n_iter, n_topics)

    t0 = time()
    lda_model.fit(tf)
    print("done in %0.3fs." % (time() - t0))

    # print("\nTopics in LDA model:")

    # print_top_words(lda_model, tf_feature_names, n_top_words)

    tf_test = tf_vectorizer.transform(test_dataset)
    lda_test = lda_model.transform(tf_test)
    return {
        'transformations': lda_test,
        'features': tf_vectorizer.get_feature_names(),
        '_model': lda_model,
        '_tf_vectorizer': tf_vectorizer,
        '_tf_transformations': tf_test,
        '_name': 'lda'
    }



def lda_documents_as_summaries(documents_as_topics, topics_words, vocab):
    documents_as_major_topic_idx = documents_as_major_topic(documents_as_topics)
    summaries = lda_topics_summaries(topics_words, vocab, 2)
    documents_as_topics_words = [summaries[idx] for idx in documents_as_major_topic_idx]
    return documents_as_topics_words


def documents_as_major_topic(documents_as_topics):
    n_topics = documents_as_topics.shape[0]
    documents_as_major_topic_idx = []
    n_major_topics = 3
    for i in range(n_topics):
        major_topics = documents_as_topics[i].argsort()[::-1][:n_major_topics]
        documents_as_major_topic_idx.append(major_topics[0])
    return documents_as_major_topic_idx


def lda_topics_summaries(topics_words, vocab, n_top_words=1):
    # and get top words for each topic:
    summaries = []
    for i, topic_dist in enumerate(topics_words):
        vocab_np = np.array(vocab)
        sorted_vocab_np = vocab_np[np.argsort(topic_dist)]
        topic_words = sorted_vocab_np[:-(n_top_words + 1):-1]
        summaries.append(' '.join(topic_words))
    return summaries


def lda_fitting_metas(lda_dump):
    documents_as_topics = lda_dump['transformations']  # numpy: D documents x T topics
    topics_words = lda_dump['_model'].components_  # numpy: T topics x W words (the words are ordered for each topic)
    vocab = lda_dump['features']  # list: W words
    return lda_documents_as_summaries(documents_as_topics, topics_words, vocab)


def ldavis(lda_dump, name='ldavis'):
    vis_data = pyLDAvis.sklearn.prepare(lda_dump["_model"],
                                        lda_dump["_tf_transformations"],
                                        lda_dump["_tf_vectorizer"],
                                        mds='tsne')
    pyLDAvis.save_html(vis_data, './output/ldavis/%s.html' % name)