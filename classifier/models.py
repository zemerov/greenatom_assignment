from torch import nn
import torch

"""
This module contains 2 examples of RNN: GRU, LSTM
"""


class GRU(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.GRU = nn.GRU(emb_dim, hidden_dim, num_layers=1, batch_first=True)  # , dropout=dropout)
        self.linear = nn.Linear(hidden_dim, 10)
        self.hidden_dim = hidden_dim
        self.logprob = nn.LogSoftmax(dim=1)
        self.drop = nn.Dropout(dropout)

    def forward(self, seq):
        seq = self.embedding(seq)
        output, hidden = self.GRU(seq)
        output = self.linear(hidden[0])

        return self.logprob(self.drop(output)), hidden

    def init_hidden(self, batch_size):
        return torch.zeros([1, batch_size, self.hidden_dim])


class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, kernel_sizes, out_channels, dropout=0.5):
        super(CNN, self).__init__()

        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.convolutions = nn.ModuleList()
        self.linear = nn.Linear(out_channels * len(kernel_sizes), 10)
        self.logprob = nn.LogSoftmax(dim=1)

        for k_size in kernel_sizes:
            self.convolutions.append(
                nn.Sequential(
                    nn.Conv1d(embedding_dim, out_channels, kernel_size=k_size, padding=k_size // 2),
                    nn.LeakyReLU(0.2)
                )
            )

    def forward(self, x):  # [batch_size, seq_len]
        x = self.embed(x)  # [batch_size, seq_len, D]
        x = x.permute((0, 2, 1))  # [batch_size, D, seq_len]

        res = []

        for layer in self.convolutions:
            # print('x shape', x.shape)
            current = torch.max(layer(x), dim=2).values
            # print(current.shape, current.shape)
            res.append(current)

        x = torch.cat(res, dim=1)
        logit = self.logprob(self.linear(x))

        return logit, 0


class LargeCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, kernel_sizes, out_channels, dropout=0.5):
        super(LargeCNN, self).__init__()

        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.convolutions = nn.ModuleList()
        self.lin1 = nn.Linear(out_channels * len(kernel_sizes) * 2, 128)
        self.lin2 = nn.Linear(128, 10)
        self.logprob = nn.LogSoftmax(dim=1)
        self.dropout = nn.Dropout(dropout)

        for k_size in kernel_sizes:
            self.convolutions.append(
                nn.Sequential(
                    nn.Conv1d(embedding_dim, out_channels, kernel_size=k_size, padding=k_size // 2),
                    nn.LeakyReLU(0.2)
                )
            )

    def forward(self, x):  # [batch_size, seq_len]
        x = self.embed(x)  # [batch_size, seq_len, D]
        x = x.permute((0, 2, 1))  # [batch_size, D, seq_len]

        res = []

        for layer in self.convolutions:
            # print('x shape', x.shape)
            max_pool = torch.max(layer(x), dim=2).values
            mean_pool = torch.mean(layer(x), dim=2)
            res.append(max_pool)
            res.append(mean_pool)

        x = self.lin1(torch.cat(res, dim=1))
        x = self.lin2(self.dropout(x))
        logit = self.logprob(x)

        return logit, 0

