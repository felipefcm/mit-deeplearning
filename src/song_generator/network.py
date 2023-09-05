import torch
from torch import nn
import torch.nn.functional as F


class SongNetwork(nn.Module):
    def __init__(self, rnn_units: int, vocab_size: int, embedding_dim: int):
        super().__init__()

        self.rnn_units = rnn_units
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.h = None
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=rnn_units,
            batch_first=True,
        )

        self.l1 = nn.Linear(rnn_units, vocab_size)

    def forward(self, x):
        embeddings = self.embedding(x)

        y, _ = self.lstm(embeddings)
        y = self.l1(y)

        y = F.log_softmax(y, dim=2)

        return y
