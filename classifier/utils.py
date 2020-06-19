import numpy as np
import torch

from torch.utils.data import Dataset


class Dataset(Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data, vocab, overfit_size=1000):
        """
        :param data: list of pairs (tokens, label)
        :param vocab: tokens mapping {'token': number, ...}
        """
        self.data = data
        self.vocab = vocab
        self.overfit = False
        self.overfit_size = overfit_size

    def __getitem__(self, index):
        """
        :param index: position of data element in self.data
        :return: list of  [word, label] for data element. word is a LongTensor of mapped tokens
        """
        label = torch.FloatTensor([int(self.data[index][1])])
        word = torch.LongTensor(self.vocab([self.data[index][0]]))[0]

        return [word, label]

    def __len__(self):
        if self.overfit:
            return self.overfit_size
        else:
            return len(self.data)


def pad_tensor(vec, length, pad_symbol):
    x = torch.stack(
        tuple(
            map(
                lambda x: torch.cat([x[0], torch.LongTensor([pad_symbol] * max(length - x[0].shape[0], 0))]),
                vec
            )
        ),
        0
    )

    return x, vec[:, 1]


class Padder:
    def __init__(self, pad_symbol=0):
        self.pad_symbol = pad_symbol

    def __call__(self, batch):
        max_size = max(map(lambda x: len(x[1]), batch))
        return pad_tensor(batch, max_size, self.pad_symbol)


def perplexity_criterion(model_, eval_iter, criter, device):
    """
    :param model_: torch.nn.Module
    :param eval_iter: evaluation itarator
    :param criter: torch.criterion
    :param device: 'cpu' or 'cuda'
    :return:
    """
    cnt_sample = 0
    perplexity = 0

    for batch, real in eval_iter:
        res, _ = model_(batch.to(device))
        perplexity += torch.exp(criter(
            res.view(np.prod(real.shape), -1),
            real.to(device).view(np.prod(real.shape))
        )).item()
        cnt_sample += 1

    perplexity /= cnt_sample

    return perplexity
