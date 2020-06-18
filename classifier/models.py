from torch import nn
import torch

"""
This module contains 2 examples of RNN: GRU, LSTM
"""


class GRU(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.GRU = nn.GRU(emb_dim, hidden_dim, num_layers=1, batch_first=True)  # , dropout=dropout)
        self.linear = nn.Linear(hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim
        self.logprob = nn.LogSoftmax(dim=2)
        self.drop = nn.Dropout(dropout)

    def forward(self, seq, hidden=None):
        seq = self.embedding(seq)
        if hidden is None:
            output, hidden = self.GRU(seq)
        else:
            output, hidden = self.GRU(seq, hidden)
        output = self.linear(output)

        return self.logprob(self.drop(output)), hidden

    def init_hidden(self, batch_size):
        return torch.zeros([1, batch_size, self.hidden_dim])


class LSTM(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.RNN = nn.LSTM(emb_dim, hidden_dim, num_layers=1, batch_first=True)
        self.linear = nn.Linear(hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim
        self.logprob = nn.LogSoftmax(dim=2)
        self.drop = nn.Dropout(dropout)

    def forward(self, seq, hidden=None, cell=None):
        seq = self.embedding(seq)
        if hidden is None:
            output, (h, c) = self.RNN(seq)
        else:
            output, (h, c) = self.GRU(seq, hidden, cell)
        output = self.linear(self.drop(output))

        return self.logprob(output), (h, c)

    def init_hidden(self, batch_size):
        return torch.zeros([1, batch_size, self.hidden_dim])
