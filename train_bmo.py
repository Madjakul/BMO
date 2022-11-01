# train_bmo.py

import torch
import torch.nn as nn
from torch.optim import SGD

from bmo.utils.dataset import LMDataset
from bmo.modules.word_embedding import CBOW


EPOCH = 10
VERVOSE = 2


def train_cbow(model: CBOW, train_data: LMDataset, valid_data: LMDataset=None):
    """Trains the language model.

    Parameters
    ----------
    model: ``bmo.modules.word_embedding.CBOW``
        Language model.
    train_data: ``bmo.utils.dataset.LMDataset``
        Custom dataset to train the model.
    valid_data: ``bmo.utils.dataset.LMDataset``
        Custom dataset to validate the training.

    Returns
    -------
    model: ``bmo.modules.word_embedding.CBOW``
        Trained language model.
    """
    nll_loss = nn.NLLLoss() # Negative log-likelihood loss function
    optimizer = SGD(model.parameters(), lr=1e-3)
    for epoch in range(EPOCH):
        total_loss = 0
        for context, target in train_data:
            x = torch.LongTensor(context)
            y = torch.LongTensor([target])
            model.zero_grad()
            log_prob = model(x)
            loss = nll_loss(log_prob, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.data
        if epoch % VERVOSE == 0:
            loss_avg = float(total_loss / len(train_data))
            print(f"{epoch}/{EPOCH} loss{loss_avg:.2f}")
    return model

def main():
    corpus = [
        "Hello nice to meet you.",
        "Hi Francis, how is the meeting going?",
        "How are you going to go to France?",
        "Where is he from?",
        "I don't know but he has gone mad.",
        "Why is he going to meet the boss?"
    ]
    dataset = LMDataset(corpus=corpus, window_size=3, merging_size=20)
    dataset.build_vocab()
    vocab_size = len(dataset.vocab)
    model = CBOW(vocab_size=vocab_size, embedding_size=128, context_size=3)
    model = train_cbow(model, train_data=dataset)
    print(model.get_embedding(dataset.vocab2ix["meet"]))


if __name__=="__main__":
    main()

