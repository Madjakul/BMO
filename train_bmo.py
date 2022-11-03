# train_bmo.py

import logging
import argparse

import torch
import torch.nn as nn
from torch.optim import SGD

from bmo.utils.dataset import LMDataset
from bmo.modules.word_embedding import CBOW


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus_path", type=str, default="./data/CDR_TrainingSet.txt"
    )
    parser.add_argument("--window_size", type=int, default=3)
    parser.add_argument("--vocab_size", type=int, default=None)
    parser.add_argument("--merging_size", type=int, default=None)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--embedding_size", type=int, default=300)
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--update_rate", type=int, default=10)
    parser.add_argument("--model_path", type=str, default="./tmp/bmo.torch")
    args = parser.parse_args()

    with open(args.corpus_path, "r", encoding="utf-8") as f:
        corpus = f.read().splitlines()
    dataset = LMDataset(
        corpus=corpus, window_size=args.window_size, device=args.device,
        vocab_size=args.vocab_size, merging_size=args.merging_size
    )
    dataset.build_vocab()

    model = CBOW(
        vocab_size=len(dataset.vocab), embedding_size=args.embedding_size,
        context_size=args.window_size
    )
    nll_loss = nn.NLLLoss() # Negative log-likelihood loss function
    optimizer = SGD(model.parameters(), lr=1e-3)
    for epoch in range(args.epoch):
        total_loss = 0
        for context, target in dataset:
            x = torch.LongTensor(context)
            y = torch.LongTensor([target])
            model.zero_grad()
            log_prob = model(x)
            loss = nll_loss(log_prob, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.data
        if epoch % args.update_rate == 0:
            loss_avg = float(total_loss / len(train_data))
            print(f"{epoch}/{EPOCH} loss{loss_avg:.2f}")

