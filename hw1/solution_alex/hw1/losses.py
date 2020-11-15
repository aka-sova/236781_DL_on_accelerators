import abc
import torch


class ClassifierLoss(abc.ABC):
    """
    Represents a loss function of a classifier.
    """

    def __call__(self, *args, **kwargs):
        return self.loss(*args, **kwargs)

    @abc.abstractmethod
    def loss(self, *args, **kw):
        pass

    @abc.abstractmethod
    def grad(self):
        """
        :return: Gradient of the last calculated loss w.r.t. model
            parameters, as a Tensor of shape (D, C).
        """
        pass


class SVMHingeLoss(ClassifierLoss):
    def __init__(self, delta=1.0):
        self.delta = delta
        self.grad_ctx = {}

    def loss(self, x, y, x_scores, y_predicted):
        """
        Calculates the Hinge-loss for a batch of samples.

        :param x: Batch of samples in a Tensor of shape (N, D).
        :param y: Ground-truth labels for these samples: (N,)
        :param x_scores: The predicted class score for each sample: (N, C).
        :param y_predicted: The predicted class label for each sample: (N,).
        :return: The classification loss as a Tensor of shape (1,).
        """

        assert x_scores.shape[0] == y.shape[0]
        assert y.dim() == 1

        # TODO: Implement SVM loss calculation based on the hinge-loss formula.
        #  Notes:
        #  - Use only basic pytorch tensor operations, no external code.
        #  - Full credit will be given only for a fully vectorized
        #    implementation (zero explicit loops).
        #    Hint: Create a matrix M where M[i,j] is the margin-loss
        #    for sample i and class j (i.e. s_j - s_{y_i} + delta).

        loss = None
        # ====== YOUR CODE: ======

        # w_j * x_i  -> given in x_scores
        # w_y_i * x_i -> take the column for the correct class in x_scores, and extend it

        # 1. take all the correct classes scores
        cor_classes_scores = x_scores[torch.arange(x_scores.shape[0]), y]
        cor_classes_scores = cor_classes_scores.unsqueeze(0).T

        # 2. create the M matrix from the hint
        M = x_scores - cor_classes_scores + self.delta  # N X D

        # 3. set the true labels items to 0
        M[torch.arange(M.shape[0]), y] = 0


        # use the max operation
        M_max = torch.max(M, torch.zeros(M.shape)) # N X D

        M_sum = torch.sum(M_max, dim=1) # N X 1

        loss = torch.sum(M_sum) / M.shape[0]

        # ========================

        # TODO: Save what you need for gradient calculation in self.grad_ctx
        # ====== YOUR CODE: ======
        self.grad_ctx =  (x,y,M)

        # ========================

        return loss

    def grad(self):
        """
        Calculates the gradient of the Hinge-loss w.r.t. parameters.
        :return: The gradient, of shape (D, C).

        """
        # TODO:
        #  Implement SVM loss gradient calculation
        #  Same notes as above. Hint: Use the matrix M from above, based on
        #  it create a matrix G such that X^T * G is the gradient.

        grad = None
        x, y, M = self.grad_ctx
        # ====== YOUR CODE: ======
        # x is [NXD]
        # M is [NXC] , where N = num features, C = num classes
        # G is [NXC]
        # grad is [DXC], where D = num features

        G_ones = torch.ones(M.shape)
        G_zeros = torch.zeros(M.shape)

        G = torch.where(M > 0, G_ones, G_zeros)

        M_new = torch.where(M > 0, G_ones, G_zeros)

        G[torch.arange(M.shape[0]), y] = -torch.sum(M_new, dim=1)  # for j = y_i

        G = G / x.shape[0] # G/N
        grad = torch.matmul(x.T, G)
        # ========================

        return grad
