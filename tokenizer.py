"""
Tranform a document into a vector
---------------------------------

Uses NLTK and Spacy to parse documents, filter, and produce syntactic tokens
"""
import multiprocessing

import spacy

from nlpkit.utils import do_or_load_json, file_to_lines
from functools import partial
from nltk.stem.porter import *
from nltk.stem.wordnet import WordNetLemmatizer

nlp = spacy.load('en')
stemmer = PorterStemmer()
lemmer = WordNetLemmatizer()

with open('./input/post_stopwords.txt', encoding="utf-8") as fin:
    poststop_words = file_to_lines(fin)


class POSToken:
    def __init__(self, text, pos, tag, isComposed=False):
        self.text = text
        self.pos_ = pos
        self.tag_ = tag
        self.isComposed = isComposed

    def __eq__(self, other):
        raise NotImplemented


def debug_without_noise_tokens(tokenized_doc):
    for x in tokenized_doc:
        if x.tag_ == u"NN":
            pass
    return []


def without_noise_tokens(tokenized_doc):
    return [x for x in tokenized_doc
            if not (
            x.pos_ == u'PUNCT'
            or x.pos_ == u"SYM"
            or x.pos_ == u"X"
            or x.pos_ == u"SPACE"
            or x.pos_ == u"NUM"
            or x.tag_ == u"NIL"
        )]


def with_composable_words(tokenized_doc):
    return [x for x in tokenized_doc
            if x.pos_ == u'NOUN'
            or x.pos_ == u"ADJ"
            or x.pos_ == u"VERB"
            or x.pos_ == u"CONJ"
            or x.pos_ == u"CCONJ"
            or x.pos_ == u"DET"
            or x.pos_ == u"ADP"
            or x.pos_ == u"PROPN"
            or x.pos_ == u"PRON"
            or x.pos_ == u"PART"
            or x.pos_ == u"ADV"]


# articles : the, a, an
# prepositions : above, before, except, from ...
# conjonctions : and, or, but, so
def without_art_prep_conj_words(tokenized_doc):
    return [x for x in tokenized_doc
            if x.tag_ != u"ART"
            and x.tag_ != u"APPRART"
            and x.tag_ != u"APPO"
            and x.tag_ != u"APPR"
            and x.pos_ != u"CONJ"
            and x.pos_ != u"CCONJ"
            and x.tag_ != u"KOKOM"
            and x.tag_ != u"KON"
            and x.tag_ != u"KOUI"
            and x.tag_ != u"KOUS"
            and x.tag_ != u"IN"
            and x.pos_ != u"DET"
            ]


# special cases for cconj (e.g and)
def without_conjj(tokenized_doc):
    return [x for x in tokenized_doc
            if x.pos_ != u"CCONJ"
            ]


def allow_noun_adj_adp(tokenized_doc):
    return [x for x in tokenized_doc
            if x.pos_ == u'NOUN'
            or x.pos_ == u"ADJ"
            or x.pos_ == u"ADP"]


def is_noun_or_adj(token):
    return token.pos_ == u'NOUN' \
           or (token.pos_ == u"ADJ"
               and (token.tag_ == u"JJ"
                    or token.tag_ == u"JJR"
                    or token.tag_ == u"AFX"
                    or token.tag_ == u"JJS"))


def allow_noun_adj(tokenized_doc):
    return [x for x in tokenized_doc if is_noun_or_adj(x)]


def tokens_to_texts(tokens):
    return [token.text for token in tokens]


def tokenize(doc):
    return [POSToken(x.text, x.pos_, x.tag_) for x in nlp(doc)]


def is_post_stop_words(word):
    predicates = [w for w in poststop_words if w == word]
    return len(predicates) > 0 or len(word) <= 1


def transform_with_bigrams(tokenized_doc, bigrams, is_full_mapping=False, iteration_nbr=25):
    iteration_size = 5
    transformed_doc = tokenized_doc
    bigrams_mappings = []
    for it in range(0, iteration_nbr + 1):
        start = it * iteration_size
        end = start + iteration_size
        interesting_bigrams = bigrams[start:end]
        transformed_doc, mappings = transform_with_gram_keyphrases(transformed_doc, interesting_bigrams)
        mappings = embedded_or_full_mapping(mappings, is_full_mapping)
        bigrams_mappings.append(mappings)
    return transformed_doc, embedded_or_full_mapping(bigrams_mappings, is_full_mapping)


def embedded_or_full_mapping(mappings, is_full_mapping):
    if is_full_mapping and len(mappings):
        returned_mapping = mappings
    else:
        returned_mapping = resolve_mappings(mappings)
    return returned_mapping


def transform_with_gram_keyphrases(tokenized_doc, grams):
    transformed_doc = tokenized_doc
    mappings = []
    for gram in grams:
        transformed_doc, mapping = replace_grams(transformed_doc, gram)
        mappings.append(mapping)
    return transformed_doc, mappings


def break_list_by_window(aList, window_size=2):
    return [aList[i:i + window_size] for i, el in enumerate(aList) if i <= len(aList) - window_size]


def replace_grams(aList, gram):
    resulted_list = []
    mapping = []
    window_size = len(gram)
    i = 0
    while i < len(aList):
        if aList[i:i + window_size] == gram:
            resulted_list.append('_'.join(gram))
            mapping.append(list(range(i, i + window_size)))
            i += window_size
        else:
            resulted_list.append(aList[i])
            mapping.append([i])
            i += 1
    return resulted_list, mapping


def bigrams_stats_as_str(csv):
    grams_as_str = csv['Gram'].tolist()
    return [gram.split('_') for gram in grams_as_str]


def transform_bigrams_with_stats(doc, csv, is_full_mapping=False, iteration_nbr=11):
    result, mappings = transform_with_bigrams(doc, csv, is_full_mapping, iteration_nbr)
    return result, mappings


def flatten_as_pos_token(doc_of_lists, doc_of_pos_tokens):
    assert len(doc_of_lists) == len(doc_of_pos_tokens)
    return [POSToken(val, doc_of_pos_tokens[i].pos_, doc_of_pos_tokens[i].tag_)
            for i, sublist in enumerate(doc_of_lists)
            for val in sublist]


def bigram_prep(doc):
    return lemmatizing(tokens_to_texts(allow_noun_adj_adp(doc)))


def trigram_prep(doc):
    return lemmatizing(tokens_to_texts(with_composable_words(doc)))


def is_mapping_el_a_word(mel):
    return len(mel) == 1 and isinstance(mel[0], int)


def resolve_mapping(current_mapping, previous_mapping, deep=False):
    assert not deep, 'Not Implemented'
    assert len(current_mapping) <= len(previous_mapping), 'Previous mapping should be lengthier'
    result_mapping = []
    for mel in current_mapping:
        if is_mapping_el_a_word(mel):
            result_mapping.append(previous_mapping[mel[0]])
        else:
            result_mapping.append(mel)
    return result_mapping


def resolve_mappings(input_mappings, deep=False):
    assert not deep, 'Not Implemented'
    i = len(input_mappings) - 1
    resolved_mapping = input_mappings[i]
    while i > 0:
        previous_mapping = input_mappings[i - 1]
        resolved_mapping = resolve_mapping(resolved_mapping, previous_mapping)
        i -= 1
    return resolved_mapping


def tokenify_words_with_mappings(words_with_grams, tokens, mappings):
    result_tokens = []
    for i, mapping in enumerate(mappings):
        if is_mapping_el_a_word(mapping):
            index = mapping[0]
            origin_token = tokens[index]
            word_token = POSToken(origin_token.text, origin_token.pos_, origin_token.tag_, False)
            result_tokens.append(word_token)
        else:
            gram_token = POSToken(words_with_grams[i], 'NOUN', None, True)
            result_tokens.append(gram_token)
    return result_tokens


def bigrams_filter(doc_text_tokens, bigrams_with_stats, iteration_nbr=11):
    # we do the same as for tri-grams but with a file of bi-grams with stats
    doc_as_strings, mapping = transform_bigrams_with_stats(
        doc_text_tokens,
        bigrams_with_stats,
        iteration_nbr=iteration_nbr)
    return doc_as_strings


def stemming(text_tokens):
    return [stemmer.stem(plural) for plural in text_tokens]


def lemmatizing(text_tokens):
    return [lemmer.lemmatize(plural) for plural in text_tokens]


def pos_token_with_replaced_text(pos_token: POSToken, new_text):
    return POSToken(new_text, pos_token.pos_, pos_token.tag_, pos_token.isComposed)


def doc_pos_tokens_with_texts(doc, texts):
    return [pos_token_with_replaced_text(pos_token, texts[i]) for i, pos_token in enumerate(doc)]


def dict_to_pos_token(token_as_dict):
    return POSToken(token_as_dict['text'],
                    token_as_dict['pos_'],
                    token_as_dict['tag_'],
                    token_as_dict['isComposed'])


def tokenize_corpus(original_dataset: list, strategy='basic'):
    if strategy == 'advanced':
        return advanced_tokenize_corpus(original_dataset)
    elif strategy == 'basic':
        return basic_tokenize_corpus(original_dataset)
    else:
        raise Exception('Unknown strategy')


# keep everything composable and apply a wordnet lemming for plurals handling
def basic_tokenize_doc(input_doc):
    doc = tokenize(input_doc)
    doc = without_noise_tokens(doc)
    tokens_text = lemmatizing(tokens_to_texts(doc))
    return doc_pos_tokens_with_texts(doc, tokens_text)


# basic : do not use collocation analysis
def basic_tokenize_corpus(original_dataset: list):
    tokenized_corpus = [basic_tokenize_doc(doc) for doc in original_dataset]
    post_stop_lines = [[word.__dict__ for word in doc if not is_post_stop_words(word.text)] for doc in
                       tokenized_corpus]

    return post_stop_lines


def advanced_tokenize_doc(input_doc, bigrams_with_stats):
    doc = tokenize(input_doc)
    pre_size = len(doc)
    # filtered_doc_as_tokens = tokens_to_texts(allow_noun_adj_verb_adp_part(doc_as_tokens))
    # doc = trigrams_filter(filtered_doc_as_tokens)
    # document as tokens filtered by their POS tagging for bigrams
    pos_tokens = without_noise_tokens(doc)
    after_size = len(pos_tokens)
    tokens_text = lemmatizing(tokens_to_texts(pos_tokens))
    lemmatized_pos_tokens = doc_pos_tokens_with_texts(pos_tokens, tokens_text)
    doc, mapping = transform_bigrams_with_stats(tokens_text, bigrams_with_stats)
    pos_tokens_with_grams = tokenify_words_with_mappings(doc, lemmatized_pos_tokens, mapping)

    return pos_tokens_with_grams, pre_size, after_size


# advanced : use collocation analysis
def advanced_tokenize_corpus(original_dataset: list, bigrams_with_stats):
    stats = bigrams_stats_as_str(bigrams_with_stats)

    tokenization_results = [advanced_tokenize_doc(doc, stats) for doc in original_dataset]

    transformed_with_bigrams = [r[0] for r in tokenization_results]

    before_filter = 0
    after_filter = 0
    for r in tokenization_results:
        before_filter += r[1]
        after_filter += r[2]

    print("Before: %d" % before_filter)
    print("After: %d" % after_filter)

    post_stop_lines = [[word.__dict__ for word in doc if not is_post_stop_words(word.text)] for doc in
                       transformed_with_bigrams]

    return post_stop_lines


def cached_or_tokenize_corpus(lazy_corpus, strategy_name="b",
                              should_overwrite=False,
                              base_path="./input/session/",
                              strategy='basic',
                              doc_filter=None,
                              raw=True):
    samples_pos_dict = do_or_load_json(strategy_name,
                                       lambda: tokenize_corpus(lazy_corpus(), strategy=strategy),
                                       should_overwrite=should_overwrite,
                                       base_path=base_path)

    corpus_as_pos_tokens = [[dict_to_pos_token(token) for token in doc] for doc in samples_pos_dict]

    if doc_filter is not None:
        corpus_as_pos_tokens = [doc_filter(doc) for doc in corpus_as_pos_tokens]

    if raw:
        return [tokens_to_texts(doc) for doc in corpus_as_pos_tokens]
    else:
        return corpus_as_pos_tokens


def cached_or_tokenize_lazy_corpus(base_path="./input/session/", strategy='basic',
                                   strategy_name="b",
                                   doc_filter=None,
                                   raw=True,
                                   cached_or_load_contents_fct=None,
                                   content_key="prepared_ind_contents"):
    return cached_or_tokenize_corpus(partial(cached_or_load_contents_fct, content_key),
                                     strategy_name=strategy_name,
                                     strategy=strategy,
                                     base_path=base_path,
                                     doc_filter=doc_filter,
                                     raw=raw)
