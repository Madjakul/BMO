# modules/word_embedding.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class CBOW(nn.Module):
    """Language model. Using the Continuous Bag of Words algorithm [1]_ to
    compute natural language symbols representaion in a vector space.

    Parameters
    ----------
    vocab_size: ``int``
        Number of unique symbols in the vocabulary.
    embedding_size: ``int``
        Size of the embedding vectors.
    context_size: ``int``
        Self explanatory.

    Attributes
    ----------
    embedding: ``torch.nn.Embedding``
        Embedding layer (vocab_size, embedding_size).
    linear_1: ``torch.nn.Linear``
        Fully connected linear layer (context_size * 2 * embedding_size, 512)
    linear_2: ``torch.nn.Linear``
        Fully connected linear layer (512, vocab_size)

    References
    ----------
    ..  [1] Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. 2013.
        Efficient estimation of word representations in vector space.
        (September 2013). Retrieved November 1, 2022 from
        https://arxiv.org/abs/1301.3781
    """

    def __init__(
            self, vocab_size: int, embedding_size: int=300, context_size: int=3
    ):
        super(CBOW, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.linear_1 = nn.Linear(context_size * 2 * embedding_size, 512)
        self.linear_2 = nn.Linear(512, vocab_size)

    def forward(self, inp: torch.LongTensor):
        """For a given input symbol, returns the probabilies of each symbol
        in the vocabulary to be the next one within the **context_size**.

        Parameters
        ----------
        inp: ``torch.LongTensor``
            Input symbol.

        Returns
        -------
        logits: ``torch.Tensor``
            For each symbol in the vocabulary, its probability of being the
            next word within the **context_size**.
        """
        hidden = self.embedding(inp).view(1, -1)
        hidden = hidden.view(1, -1)
        logits = F.relu(self.linear_1(hidden))
        logits = F.log_softmax(logits, dim=1)
        return logits

    def get_embedding(self, word_idx: int):
        """Returns the embeddin vector associated to a symbol in the
        vocabulary.

        Parameters
        ----------
        word_idx: ``int``
            Index associated to a symbol.

        Returns
        -------
        embedding: ``torch.Tensor``
            Embedding vector representing the input symbol.
        """
        word = Variable(torch.LongTensor([word_idx]))
        embedding = self.embedding(word).view(1, -1)
        return embedding

