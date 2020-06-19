import collections
import itertools
import io
import string
from os import listdir

"""
This file contains 2 classes:

1. ManualTokenizer
2. Vocabulary
"""


class ManualTokenizer:
    """
    Tokenizer read raw txt file and make simple preprocessing.
    """

    def __init__(self):
        self.cnt = 0

    def get_tokens_and_score(self, dir_path):
        """
        :param dir_path: path to txt file

        :return (yield) text, score
        """

        for filename in listdir(dir_path):
            with io.open(dir_path + filename, encoding='utf-8') as file:
                self.cnt += 1
                table = str.maketrans('', '', string.punctuation)

                score = filename.split('.')[0].split('_')[1]  # Get score from the name
                tokens = ''.join(file.readlines()).replace('<br />', ' ')
                
                yield [tok.lower().translate(table) for tok in tokens.split()], score


class Vocabulary:
    def __init__(self, special_tokens=['UNK']):
        self.i2t = {}
        self.t2i = {}
        self.special_tokens = special_tokens
        self.count = 0
        self.counter = collections.Counter()

    def fit(self, tokens, min_count=0):
        for token in self.special_tokens:
            self.t2i[token] = self.count
            self.i2t[self.count] = token
            self.count += 1

        for token in itertools.chain(*tokens):
            self.counter[token] += 1

        for token in itertools.chain(*tokens):
            if token not in self.t2i.keys() and self.counter[token] >= min_count:
                self.t2i[token] = self.count
                self.i2t[self.count] = token
                self.count += 1

    def __len__(self):
        return self.count

    def __call__(self, batch):
        indices_batch = []

        for sample in batch:
            current = []
            for token in sample:
                if token in self.t2i.keys():
                    current.append(self.t2i[token])
                else:
                    current.append(self.t2i['UNK'])
            indices_batch.append(current)

        return indices_batch

    def get_word(self, idx):
        return list(map(lambda x: self.i2t[x], idx))
