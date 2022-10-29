# utils/dataset.py

import re

from torch.utils.data import Dataset

from bmo.utils import preprocessing


class LMDataset(Dataset):
    """Custom dataset for language modeling.

    Parameters
    ----------
    corpus: ``str``
        Raw corpus in natural language.
    window_size: ``int``
        Window size.
    device: ``str``
        Device where the tensor are stored.
    max_size: ``int``
        Maximum size of the vocabulary.
    max_iter: ``int``
        Maximum number of iteration.

    Attributes
    ----------
    corpus: ``str``
        Raw corpus in natural language.
    window_size: ``int``
        Window size.
    splitted_corpus: ``list``
        Training corpus splitted and merged according to the merging rules.
    encoded_corpus: ``list``
        ``splitted_corpus`` as integers.
    data: ``list``
        List of data points as tuple of word and target word within a window
        size.
    vocab_frequencies: ``dict``
        Tokens vocabulary and their frequency.
    vocab: ``dict``
        Unique symbols and their integer value.
    device: ``str``
        Device where the tensor are stored.
    max_size: ``int``
        Maximum size of the vocabulary.
    max_iter: ``int``
        Maximum number of iteration.
    pair_stats: ``dict``
        Symbol pairs with their frequency.
    bpe_codes: ``dict``
        Unique identifier of each pairs.
    """

    word_freqs: dict = {}
    alphabet: list = []
    splits: dict = {}
    pair_freqs: dict = {}
    vocab: list = []
    merges: dict = {}
    vocab2ix: dict = {}
    splitted_corpus: list = []
    encoded_corpus: list = []
    data: list = []

    def __init__(
            self, corpus: list=[], window_size: int=3, device: str="cpu",
            vocab_size: int=None, merging_size: int=None, *args, **kwargs
    ):
        self.corpus = corpus
        self.window_size = window_size
        self.vocab_size = vocab_size
        self.merging_size = merging_size
        self.device = device

    def build_vocab(self):
        """Uses the corpus to build merging rules and a vocabulary. Use the
        defined rules to tokenize and encode the corpus in order to build the
        training data.

        Warnings
        --------
        This method shall only be used when no pre-trained paramaters have been
        loaded into the class.
        """
        self.word_freqs = preprocessing.get_word_frequencies(self.corpus)
        self.alphabet = preprocessing.get_alphabet(self.word_freqs)
        self.vocab = ["</unk>", "</pad>", "</eos>"] + self.alphabet.copy()
        self.splits = {word: [*word] for word in self.word_freqs.keys()}
        i = 0
        while True:
            self.pair_freqs = preprocessing.get_pair_freqs(
                self.word_freqs, self.splits
            )
            best_pair = ""
            max_freq = None
            for pair, freq in self.pair_freqs.items():
                if max_freq is None or max_freq < freq:
                    best_pair = pair
                    max_freq = freq
            self.splits = preprocessing.merge_pair(
                *best_pair, self.splits, self.word_freqs
            )
            self.merges[best_pair] = best_pair[0] + best_pair[1]
            self.vocab.append(best_pair[0] + best_pair[1])
            if self.vocab_size is not None:
                if len(self.vocab) >= self.vocab_size:
                    break
            i += 1
            self.vocab2ix = {i: word for word, i in enumerate(self.vocab)}
            if self.merging_size is not None:
                if i == self.merging_size:
                    break
        for doc in self.corpus:
            splitted_doc = self.tokenize(doc)
            encoded_doc = self.encode(splitted_doc)
            self.splitted_corpus.append(splitted_doc)
            self.encoded_corpus.append(encoded_doc)
            self.data.extend(
                preprocessing.windowizer(encoded_doc, wsize=self.window_size)
            )

    def tokenize(self, text: str):
        """Apply BPE algorithm [1]_ to an input string.

        Parameters
        ----------
        text: ``str``
            Input string.

        Returns
        -------
        sum(splits, []): ``list``
            List of symbols.

        References
        ----------
        ..  [1] Shibata Yusuke, Kida Takuya, Fukamachi Shuichi, Takeda Masayuki,
            Shinohara Ayumi, Shinohara Takeshi. (1999). Byte Pair Encoding: A
            Text Compression Scheme That Accelerates Pattern Matching.
        """
        space_pattern = re.compile("([^A-Za-zÀ-ÖØ-öø-ÿ0-9\s]+)") # Match punctuation
        spaced_text = re.sub(space_pattern, r" \1 ", text)       # Add space around punctuation
        tokens = spaced_text.split()
        splits = [[*word] for word in tokens]
        for pair, merge in self.merges.items():
            for idx, split in enumerate(splits):
                i = 0
                while i < len(split) - 1:
                    if split[i] == pair[0] and split[i + 1] == pair[1]:
                        split = split[:i] + [merge] + split[i + 2 :]
                    else:
                        i += 1
                splits[idx] = split
        return sum(splits, [])

    def encode(self, tokenized_text: list):
        """Express a list of symbols as integers.

        Parameters
        ----------
        tokenized_text: ``list``
            List of symbols from the ``_tokenized`` method.

        Returns
        -------
        encoded: ``list``
            Tokenized text expressed as integers.
        """
        encoded = [self.vocab2ix.get(token, 0) for token in tokenized_text]
        return encoded

    def __call__(self, text: str):
        return self.tokenize(text)

    def __len__(self):
        return len(data)

    def __getitem__(self, idx):
        return self.data[idx]

