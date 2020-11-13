import os

import torch
from torchtext.data import Field, LabelField, TabularDataset, Iterator
from torchtext.vocab import Vectors
from torchtext import datasets

torch.manual_seed(1234)
torch.cuda.manual_seed(1234)

import torch
from torchtext import data

torch.backends.cudnn.deterministic = True



class MyDataset(object):

    def __init__(self, root_dir='data', batch_size=64, use_vector=True):

        self.TEXT = Field(tokenize='revtok')
        self.LABEL = LabelField(dtype=torch.float)

        train_data, test_data = datasets.IMDB.splits(self.TEXT, self.LABEL)

        train_data, valid_data = train_data.split(random_state=torch.random.seed(1234))

        MAX_VOCAB_SIZE = 25_000
        self.TEXT.build_vocab(train_data, max_size=MAX_VOCAB_SIZE)
        self.LABEL.build_vocab(train_data)

        BATCH_SIZE = 128

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.dataloader=data.BucketIterator.splits(
            (train_data, valid_data, test_data),
            batch_size=BATCH_SIZE,
            device=device)


if __name__ == '__main__':
    dataset = MyDataset()
