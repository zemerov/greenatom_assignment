import collections
import itertools
import io
import string

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
        self.text = ''

    def read_file(self, file_path):
        """
        :param file_path: path to txt file
        """

        with io.open(file_path, encoding='utf-8') as file:
            self.text = file.readlines()

    def get_tokens(self):
        """
        Note: you should call read_file method before to store raw txt file

        :return python list  of words
        """
        if self.text == '':
            raise Exception('Nothing to tokenize')
        else:
            self.text = ''.join([x.replace('\n', ' ') for x in self.text[5:]])
            self.text = list(filter(lambda x: len(x) > 3, self.text.split('.')))
            self.text = [" ".join(x.split()) for x in self.text]

            table = str.maketrans('', '', string.punctuation)

            return [tok.lower().translate(table) for tok in self.text]  # Get rid of punctuation


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
