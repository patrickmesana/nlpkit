"""
Helpers to prepare the corpus for tokenization
----------------------------------------------

"""
import re
import string
from nlpkit.utils import space_split, space_join, flatten


def striphtml(data):
    p = re.compile(r'<.*?>')
    return p.sub('', data)


def filter_printable(s):
    printable = set(string.printable)
    return ''.join(filter(lambda x: x in printable, s))


def punctuation_cleaning(words):
    table = str.maketrans({key: ' ' for key in string.punctuation})
    cleaned_words = [w.translate(table) for w in words]
    resulted_doc = []
    for word in cleaned_words:
        spliced_word = word.split(' ')
        spliced_word = [w for w in spliced_word if len(w) > 0]
        resulted_doc.append(spliced_word)
    return resulted_doc


def prepare_line(line):
    line = striphtml(line)
    line = filter_printable(line)
    line = line.lower()
    line = re.sub(r'^https?:\/\/.*[\r\n]*', '', line)
    words = space_split(line)
    words = flatten(punctuation_cleaning(words))
    line = space_join(words)
    return line
