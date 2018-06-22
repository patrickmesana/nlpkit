"""
Gemsim implementation of LDA
----------------------------

This is another implementation of LDA than Scipy or LDA module
https://rstudio-pubs-static.s3.amazonaws.com/79360_850b2a69980c4488b1db95987a24867a.html

Public Methods
~~~~~~~~~~~~~~
    train_model


"""

from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from gensim import corpora
import gensim
import numpy as np

def prepare_corpus(doc_set):
    tokenizer = RegexpTokenizer(r'\w+')

    # create English stop words list
    en_stop = get_stop_words('en')

    # Create p_stemmer of class PorterStemmer
    p_stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    # list for tokenized documents in loop
    texts = []
    # loop through document list
    for i in doc_set:
        # clean and tokenize document string
        raw = i.lower()
        tokens = tokenizer.tokenize(raw)


        # stem tokens
        # definition of stemming: reducing inflected (or sometimes derived) words to their word stem, base or root form
        tokens = [p_stemmer.stem(i) for i in tokens]

        # remove stop words from tokens
        tokens = [i for i in tokens if not i in en_stop]

        # add tokens to list
        texts.append(tokens)

    # turn our tokenized documents into a id <-> term dictionary
    dictionary = corpora.Dictionary(texts)
    # convert tokenized documents into a document-term matrix
    corpus = [dictionary.doc2bow(text) for text in texts]
    return corpus, dictionary


def train_model(data_samples, test_dataset):
    nbr_of_topics = 50
    # turn our tokenized documents into a id <-> term dictionary
    tf_vectorizer = corpora.Dictionary(data_samples, prune_at=None)
    # convert tokenized documents into a document-term matrix
    tf_train = [tf_vectorizer.doc2bow(text) for text in data_samples]
    # generate LDA model
    ldamodel = gensim.models.ldamodel.LdaModel(tf_train,
                                               num_topics=nbr_of_topics,
                                               id2word=tf_vectorizer,
                                               passes=1,
                                               iterations=100)

    tf_test = [tf_vectorizer.doc2bow(text) for text in test_dataset]
    lda_transformations = [ldamodel.get_document_topics(bow) for bow in tf_test]

    tmp_transformations = [[0] * nbr_of_topics for _ in tf_test]

    for i, tmp_t in enumerate(tmp_transformations):
        for t in lda_transformations[i]:
            tmp_t[t[0]] = t[1]

    return {
        'transformations': np.array(tmp_transformations),
        'features': sorted(list(tf_vectorizer.id2token.values())),
        '_model': ldamodel,
        '_tf_vectorizer': tf_vectorizer,
        '_tf_transformations': tf_test
    }

