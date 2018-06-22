"""
Collocation Analysis
--------------------

Looking at bi-grams and tri-grams, ordering and printing by statistics

"""

from __future__ import print_function
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder, TrigramAssocMeasures, \
    TrigramCollocationFinder
from nlpkit.tokenizer import tokenize, bigram_prep, trigram_prep, break_list_by_window, allow_noun_adj_adp, tokens_to_texts, \
    lemmatizing, is_noun_or_adj
from nlpkit.utils import lines_to_file


# very slow compare to libs
def top_unigram(tokenized_docs, nbr_of_tops=1, should_filter=False):
    if should_filter:
        filtered_tokenized_docs = [bigram_prep(doc) for doc in tokenized_docs]
    else:
        filtered_tokenized_docs = tokenized_docs

    flatten_corpus = [t for d in filtered_tokenized_docs for t in d]
    vocab = [[el, flatten_corpus.count(el)] for el in set(flatten_corpus)]
    sorted_vocab = sorted(vocab, key=lambda w: w[1])
    sorted_vocab.reverse()
    return sorted_vocab[0:nbr_of_tops]


def top_trigram(tokenized_docs, as_str=True, should_filter=False, nbr_of_tops=1):
    trigrams_finder = create_trigram_finder(tokenized_docs, should_filter)
    most_common_as_tuple_list = [(most_common[0][0], most_common[0][1], most_common[0][2], most_common[1])
                                 for most_common in trigrams_finder.ngram_fd.most_common()]
    tops = most_common_as_tuple_list[0:nbr_of_tops]

    if as_str:
        results = [['%s_%s_%s' % (t[0], t[1], t[2]), t[3]] for t in tops]
    else:
        results = tops
    return results


def top_bigram(tokenized_docs, as_str=True, should_filter=False, nbr_of_tops=1):
    bigrams_finder = create_bigram_finder(tokenized_docs, should_filter)
    most_common_as_tuple_list = [(most_common[0][0], most_common[0][1], most_common[1])
                                 for most_common in bigrams_finder.ngram_fd.most_common()]
    tops = most_common_as_tuple_list[0:nbr_of_tops]
    if as_str:
        results = [['%s_%s' % (t[0], t[1]), t[2]] for t in tops]
    else:
        results = tops
    return results


def create_bigram_finder(tokenized_docs, should_filter=False):
    if should_filter:
        bigrams_data_samples = [bigram_prep(doc) for doc in tokenized_docs]
    else:
        bigrams_data_samples = tokenized_docs
    bigrams_finder = BigramCollocationFinder.from_documents(bigrams_data_samples)
    return bigrams_finder


def create_trigram_finder(tokenized_docs, should_filter=False):
    if should_filter:
        trigrams_data_samples = [trigram_prep(doc) for doc in tokenized_docs]
    else:
        trigrams_data_samples = tokenized_docs
    trigrams_finder = TrigramCollocationFinder.from_documents(trigrams_data_samples)
    return trigrams_finder


def save_trigrams(tokenized_docs, shouldWriteToFile=False):
    trigrams_finder = create_trigram_finder(tokenized_docs)
    trigram_measures = TrigramAssocMeasures()
    trigrams_scores = trigrams_finder.score_ngrams(trigram_measures.likelihood_ratio)

    trigrams_counts = ['%s_%s_%s,%d\n' % (most_common[0][0], most_common[0][1], most_common[0][2], most_common[1])
                       for most_common in trigrams_finder.ngram_fd.most_common()]

    trigrams_scores_as_str = [
        '%s_%s_%s,%d\n' % (most_common[0][0], most_common[0][1], most_common[0][2], most_common[1])
        for most_common in trigrams_scores]
    if shouldWriteToFile:
        with open('./output/trigrams_counts.csv', "w", encoding="utf8") as fout:
            lines_to_file(trigrams_counts, fout)
        with open('./output/trigrams_lr_scores.csv', "w", encoding="utf8") as fout:
            lines_to_file(trigrams_scores_as_str, fout)


def is_nouns_or_adjs(bigram):
    return is_noun_or_adj(bigram[0]) and is_noun_or_adj(bigram[1])


def key_from_bigram(bigram):
    return bigram[0] + '_' + bigram[1]


def save_bigrams(tokenized_docs, shouldWriteToFile=False):
    samples_pos = [allow_noun_adj_adp(doc) for doc in tokenized_docs]
    samples_rawlem = [lemmatizing(tokens_to_texts(doc)) for doc in samples_pos]
    bigrams_pos = [break_list_by_window(s) for s in samples_pos]
    bigrams_rawlem = [break_list_by_window(s) for s in samples_rawlem]

    bigrams_count = {}
    for i, doc in enumerate(bigrams_pos):
        for j, bigram in enumerate(doc):
            if is_nouns_or_adjs(bigram):
                lem_bigram = bigrams_rawlem[i][j]
                k = key_from_bigram(lem_bigram)
                if k in bigrams_count:
                    bigrams_count[k] += 1
                else:
                    bigrams_count[k] = 1

    bigrams_count_array = [[b, bigrams_count[b]] for b in bigrams_count]
    sorted_bigrams = sorted(bigrams_count_array, key=lambda b: b[1], reverse=True)
    bigrams_str = ['%s,%d\n' % (b[0], b[1]) for b in sorted_bigrams]
    if shouldWriteToFile:
        with open('./output/bigrams_counts.csv', "w", encoding="utf8") as fout:
            lines_to_file(bigrams_str, fout)


def DEPRECATED_save_bigrams(tokenized_docs, shouldWriteToFile=False):
    bigrams_data_samples = [bigram_prep(doc) for doc in tokenized_docs]

    bigram_measures = BigramAssocMeasures()
    bigrams_finder = BigramCollocationFinder.from_documents(bigrams_data_samples)
    bigrams_scores = bigrams_finder.score_ngrams(bigram_measures.likelihood_ratio)

    bigrams_counts = ['%s_%s,%d\n' % (most_common[0][0], most_common[0][1], most_common[1])
                      for most_common in bigrams_finder.ngram_fd.most_common()]

    bigrams_scores_as_str = ['%s_%s,%d\n' % (most_common[0][0], most_common[0][1], most_common[1])
                             for most_common in bigrams_scores]

    if shouldWriteToFile:
        with open('./output/bigrams_counts.csv', "w", encoding="utf8") as fout:
            lines_to_file(bigrams_counts, fout)
        with open('./output/bigrams_lr_scores.csv', "w", encoding="utf8") as fout:
            lines_to_file(bigrams_scores_as_str, fout)