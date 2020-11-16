import math
import numpy as np
import torch
import torch.utils.data
from typing import Sized, Iterator
from torch.utils.data import Dataset, Sampler


class FirstLastSampler(Sampler):
    """
    A sampler that returns elements in a first-last order.
    """

    def __init__(self, data_source: Sized):
        """
        :param data_source: Source of data, can be anything that has a len(),
        since we only care about its number of elements.
        """
        super().__init__(data_source)
        self.data_source = data_source
        self.indices = list(range(0, len(data_source)))
    def __iter__(self) -> Iterator[int]:

        # Implement the logic required for this sampler.
        # If the length of the data source is N, you should return indices in a
        # first-last ordering, i.e. [0, N-1, 1, N-2, ...].
        # ====== YOUR CODE: ======
        self.cur_idx = None
        self.added_last = None
        self.last_sample = None

        return self
        # ========================

    def __next__(self):

        if self.last_sample == round(len(self.data_source) / 2):
            # stop when reaching middle
            raise StopIteration

        if self.last_sample is None:
            self.cur_idx = 0
            self.added_last = 0
            self.last_sample = 0
        else:
            if self.added_last == 0:
                self.added_last = 1
                self.cur_idx += 1
                self.last_sample = len(self.data_source) - round((self.cur_idx + 1) / 2)
            else:
                self.added_last = 0
                self.cur_idx += 1
                self.last_sample = round(self.cur_idx / 2)  # for even indexes

        return self.last_sample

    def __len__(self):
        return len(self.data_source)


def create_train_validation_loaders(
    dataset: Dataset, validation_ratio, batch_size=100, num_workers=2
):
    """
    Splits a dataset into a train and validation set, returning a
    DataLoader for each.
    :param dataset: The dataset to split.
    :param validation_ratio: Ratio (in range 0,1) of the validation set size to
        total dataset size.
    :param batch_size: Batch size the loaders will return from each set.
    :param num_workers: Number of workers to pass to dataloader init.
    :return: A tuple of train and validation DataLoader instances.
    """
    if not (0.0 < validation_ratio < 1.0):
        raise ValueError(validation_ratio)

    # TODO:
    #  Create two DataLoader instances, dl_train and dl_valid.
    #  They should together represent a train/validation split of the given
    #  dataset. Make sure that:
    #  1. Validation set size is validation_ratio * total number of samples.
    #  2. No sample is in both datasets. You can select samples at random
    #     from the dataset.
    #  Hint: you can specify a Sampler class for the `DataLoader` instance
    #  you create.
    # ====== YOUR CODE: ======
    val_size = round(validation_ratio * len(dataset))

    # split the dataset from the previously created function
    # list for train and test
    train_list = list(range(0,round(len(dataset) - val_size)))  # first indices
    valid_list = list(range(round(len(dataset) - val_size), round(len(dataset))))

    train_dataset = torch.utils.data.Subset(dataset, train_list)
    valid_dataset = torch.utils.data.Subset(dataset, valid_list)

    dl_train = torch.utils.data.DataLoader(dataset=train_dataset,sampler=torch.utils.data.SubsetRandomSampler(train_list))
    dl_valid = torch.utils.data.DataLoader(dataset=valid_dataset,sampler=torch.utils.data.SubsetRandomSampler(valid_list))

    train_idx = (dl_train.sampler.indices)
    valid_idx = (dl_valid.sampler.indices)

    # ========================

    return dl_train, dl_valid
