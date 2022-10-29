# utils/preprocessing.py

import re
from collections import Counter, defaultdict


def get_word_frequencies(corpus: str):
    """Compute the frequency of each word.

    Parameters
    ----------
    corpus: ``str``
        List of texts in natural language.

    Returns
    -------
    word_freqs: ``dict``
        Dictionnary of words and their frequency.

    Examples
    --------
    >>> corpus = ["low low low low low lower lower newest newest newest newest
    >>>    newest newest widest widest widest happier happier"]
    >>> word_freqs = get_word_freqs(corpus)
    >>> word_freqs
    {"low": 5,
     "lower": 2,
     "newest": 6,
     "hidest": 6,
     "widest": 3,
     "happier": 2}
    """
    space_pattern = re.compile("([^A-Za-zÀ-ÖØ-öø-ÿ0-9\s]+)") # Match punctuation
    tokens = []
    for text in corpus:
        spaced_text = re.sub(space_pattern, r" \1 ", text)   # Add space around punctuation
        tokens.extend([word for word in spaced_text.split()])
    word_freqs = Counter(tokens)
    return word_freqs

def get_alphabet(word_freqs: dict):
    """Get the list of unique characters in the corpus.

    Parameters
    ----------
    word_freqs: ``dict``
        Dictionnary of words and their frequency.

    Returns
    -------
    alphabet: ``list``
        List of unique characters.

    Examples
    --------
    >>> alphabet = get_alphabet(word_freqs)
    >>> alphabet
    ['a', 'd', 'e', 'h', 'i', 'l', 'n', 'o', 'p', 'r', 's', 't', 'w']
    """
    alphabet = []
    for word in word_freqs.keys():
        alphabet.extend([*word])
    alphabet = list(set(alphabet))
    alphabet.sort()
    return alphabet

def get_pair_freqs(word_freqs: dict, splits: dict):
    """Compute the frequency of each symbol pairs.

    Parameters
    ----------
    word_freqs: ``dict``
        Dictionnary of words and their frequency.
    splits: ``dict``
        List of characters for each word in the vocabulary.

    Returns
    -------
    pair_freqs: ``dict``
        Dictionary binding each pair of symbols to its frequency.

    Examples
    --------
    >>> splits = {word: [c for c in word] for word in word_freqs.keys()}
    >>> pair_freqs = get_pair_freqs(splits)
    >>> pair_freqs
    {("l", "o"): 7,
     ("o", "w"): 7,
     ("w", "e"): 8,
     ("e", "r"): 4,
     ("n", "e"): 6,
     ("e", "w"): 6,
     ("e", "s"): 9,
     ("s", "t"): 9,
     ("w", "i"): 3,
     ("i", "d"): 3,
     ("d", "e"): 3,
     ("h", "a"): 2,
     ("a", "p"): 2,
     ("p", "p"): 2,
     ("p", "i"): 2,
     ("i", "e"): 2}
    """
    pair_freqs = defaultdict(int)
    for word, freq in word_freqs.items():
        split = splits[word]
        if len(split) == 1:
            continue
        for i in range(len(split) - 1):
            pair = (split[i], split[i + 1])
            pair_freqs[pair] += freq
    return pair_freqs

def merge_pair(a: str, b: str, splits: dict, word_freqs: dict):
    """Merge the pair passed into the vocabulary and add it in the dictionary
    of rules for merging.

    Parameters
    ----------
    a: ``str``
        First symbol.
    b: ``str``
        Second symbol.
    splits: ``dict``
        Dictionary of rules for merging.
    word_freqs: ``dict``
        Dictionary of words and their frequency.

    Returns
    -------
    splits: ``dict``
        Updated dictionary of tules for merging.
    """
    for word in word_freqs:
        split = splits[word]
        if len(split) == 1:
            continue
        i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i + 1] == b:
                split = split[:i] + [a + b] + split[i + 2 :]
            else:
                i += 1
        splits[word] = split
    return splits

def windowizer(doc: list, wsize=3):
    """Converts sequences to sliding-window pairs [1]_ .

    Parameters
    ----------
    doc: ``list``
        Sequence of symbols in the form of integers.
    wsize: ``int``
        Window size.

    Returns
    -------
    out: ``list``
        List of symbol pairs as integers.

    References
    ----------
    ..  [1] Implementing Word2Vec in PyTorch. (2021, October 21). Full Stack
        Political Science.
        https://muhark.github.io/python/ml/nlp/2021/10/21/word2vec-from-scratch.html
    """
    out = []
    for i, symbol_int in enumerate(doc):
        target = symbol_int
        window = [
            i + j for j in range(-wsize, wsize + 1)\
            if (i + j >= 0) & (i + j < len(doc)) & (j != 0)
        ]
        out += [(target, doc[idx]) for idx in window]
    return out

