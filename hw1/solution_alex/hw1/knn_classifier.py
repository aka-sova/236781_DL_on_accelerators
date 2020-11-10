import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

import cs236781.dataloader_utils as dataloader_utils

from . import dataloaders


class KNNClassifier(object):
    def __init__(self, k):
        self.k = k
        self.x_train = None
        self.y_train = None
        self.n_classes = None

    def train(self, dl_train: DataLoader):
        """
        Trains the KNN model. KNN training is memorizing the training data.
        Or, equivalently, the model parameters are the training data itself.
        :param dl_train: A DataLoader with labeled training sample (should
            return tuples).
        :return: self
        """

        # TODO:
        #  Convert the input dataloader into x_train, y_train and n_classes.
        #  1. You should join all the samples returned from the dataloader into
        #     the (N,D) matrix x_train and all the labels into the (N,) vector
        #     y_train.
        #  2. Save the number of classes as n_classes.
        # ====== YOUR CODE: ======

        x_train_l = []
        y_train_l = []


        for idx, batch in enumerate(dl_train):
            x_train_l.append(batch[0])
            y_train_l.append(batch[1])

        x_train = torch.cat(x_train_l, dim=0)
        y_train = torch.cat(y_train_l, dim=0)
        n_classes = torch.unique(y_train).shape[0]

        # ========================

        self.x_train = x_train
        self.y_train = y_train
        self.n_classes = n_classes
        return self

    def predict(self, x_test: Tensor):
        """
        Predict the most likely class for each sample in a given tensor.
        :param x_test: Tensor of shape (N,D) where N is the number of samples.
        :return: A tensor of shape (N,) containing the predicted classes.
        """

        # Calculate distances between training and test samples
        dist_matrix = l2_dist(self.x_train, x_test)

        # TODO:
        #  Implement k-NN class prediction based on distance matrix.
        #  For each training sample we'll look for it's k-nearest neighbors.
        #  Then we'll predict the label of that sample to be the majority
        #  label of it's nearest neighbors.

        n_test = x_test.shape[0]
        y_pred = torch.zeros(n_test, dtype=torch.int64)

        for i in range(n_test):
            # TODO:
            #  - Find indices of k-nearest neighbors of test sample i
            #  - Set y_pred[i] to the most common class among them
            #  - Don't use an explicit loop.
            # ====== YOUR CODE: ======

            # taking relevant row from the dist matrix
            distances = dist_matrix[:, i]

            # sort
            sorted, indices = torch.sort(distances)

            # cut all beyond K
            sorted_k = sorted[0:self.k-1]
            indices_k = indices[0:self.k-1]

            # count most common label. take first if equal number of cases
            labels = self.y_train[indices_k]
            counts = torch.bincount(labels)

            y_pred[i] =torch.argmax(counts)

            # ========================

        return y_pred


def l2_dist(x1: Tensor, x2: Tensor):
    """
    Calculates the L2 (euclidean) distance between each sample in x1 to each
    sample in x2.
    :param x1: First samples matrix, a tensor of shape (N1, D).
    :param x2: Second samples matrix, a tensor of shape (N2, D).
    :return: A distance matrix of shape (N1, N2) where the entry i, j
    represents the distance between x1 sample i and x2 sample j.
    """

    # TODO:
    #  Implement L2-distance calculation efficiently as possible.
    #  Notes:
    #  - Use only basic pytorch tensor operations, no external code.
    #  - Solution must be a fully vectorized implementation, i.e. use NO
    #    explicit loops (yes, list comprehensions are also explicit loops).
    #    Hint: Open the expression (a-b)^2. Use broadcasting semantics to
    #    combine the three terms efficiently.
    #  - Don't use torch.cdist


    # ====== YOUR CODE: ======

    # l2 = sqrt ( sum  ( x_i^2 - 2*x_i*y_i + y_i^2 ) for all i,j in N1,N2 )

    # to implement it we expand one 2D tensor to a 3rd dimension, duplicating the 2D tensor
    # length of 3rd dim is N2.  Bringing N1,D -> N1, D, N2
    # then we use broadcasting to multiply it by rotated 2nd tensor
    # this gives us x_i * y*i matrix of size N1, D, N2, later multiply by -2

    # we make same matrix for x_i^2 of size N1, D, N2
    # and matrix for y_i^2 of size N2, D, N1  (which we rotate)

    # then summing over matrixes along D and taking sqrt we get the result of size N1 by N2

    N1, D = x1.shape
    N2, _ = x2.shape

    # bring x1 to N1, D, N2
    extended_x1 = torch.zeros(N1, D, 1)
    extended_x1[:, :, 0] = x1
    extended_x1 = extended_x1.repeat(1, 1, N2)

    # bring x2 to N1, D, N2
    extended_x2 = torch.zeros(N2, D, 1)
    extended_x2[:, :, 0] = x2
    extended_x2 = extended_x2.repeat(1, 1, N1)
    extended_x2 = torch.rot90(extended_x2, 1, [0, 2])

    # calculate x_i ^ 2, y_i ^ 2, -2 * x_i * y_i
    x_i_sqrd = extended_x1 * extended_x1
    y_i_sqrd = extended_x2 * extended_x2
    x_i_y_i = extended_x1 * extended_x2

    # sum over the dim = 1, which is D
    total_tensor = torch.sum((x_i_sqrd + y_i_sqrd - 2*x_i_y_i),1)
    dists = torch.sqrt(total_tensor)


    # ========================

    return dists


def accuracy(y: Tensor, y_pred: Tensor):
    """
    Calculate prediction accuracy: the fraction of predictions in that are
    equal to the ground truth.
    :param y: Ground truth tensor of shape (N,)
    :param y_pred: Predictions vector of shape (N,)
    :return: The prediction accuracy as a fraction.
    """
    assert y.shape == y_pred.shape
    assert y.dim() == 1

    # TODO: Calculate prediction accuracy. Don't use an explicit loop.
    # ====== YOUR CODE: ======

    y_diff = (y - y_pred)
    bool_list = list(map(bool,y_diff))  # hope it's not 'explicit loop'
    false_preds = sum(bool_list)

    accuracy = 1 - (false_preds / y.shape[0])

    # ========================

    return accuracy


def find_best_k(ds_train: Dataset, k_choices, num_folds):
    """
    Use cross validation to find the best K for the kNN model.

    :param ds_train: Training dataset.
    :param k_choices: A sequence of possible value of k for the kNN model.
    :param num_folds: Number of folds for cross-validation.
    :return: tuple (best_k, accuracies) where:
        best_k: the value of k with the highest mean accuracy across folds
        accuracies: The accuracies per fold for each k (list of lists).
    """

    accuracies = []

    for i, k in enumerate(k_choices):
        model = KNNClassifier(k)

        # TODO:
        #  Train model num_folds times with different train/val data.
        #  Don't use any third-party libraries.
        #  You can use your train/validation splitter from part 1 (note that
        #  then it won't be exactly k-fold CV since it will be a
        #  random split each iteration), or implement something else.

        # ====== YOUR CODE: ======
        raise NotImplementedError()
        # ========================

    best_k_idx = np.argmax([np.mean(acc) for acc in accuracies])
    best_k = k_choices[best_k_idx]

    return best_k, accuracies
