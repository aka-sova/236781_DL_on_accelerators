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
        #       that's best way i found to do it...
        cor_classes_scores = [(x_score[y_true]) for x_score, y_true in zip(x_scores,y)]
        cor_classes_scores = torch.Tensor(cor_classes_scores).unsqueeze(0).T # N X 1

        # 2. create the M matrix from the hint - without delta yet
        M = x_scores - cor_classes_scores + self.delta  # N X D

        # 3. use a stupid loop, later remove it
        for N in range(M.shape[0]):
            for D in range(M.shape[1]):
                if D == (y[N].item()):
                    # j == y_i
                    M[N][D] -= self.delta


        # use the max operation
        M_max = torch.max(M, torch.zeros(M.shape)) # N X D

        M_sum = torch.sum(M_max, 1) # N X 1
        M_sum = M_sum # subtract delta one time for the correct class

        num_samples = x.shape[0]
        loss = torch.sum(M_sum) / num_samples

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

        # M is [NXC] , where N = num features, C = num classes
        # G is [NXC]
        # grad is [DXC], where D = num features

        G = torch.ones(M.shape)

        # use simple loop to check that the calculation is correct
        for N in range(G.shape[0]):
            # every sample

            # sum_falses = 0
            for D in range(G.shape[1]):
                # class prediction
                if D != (y[N].item()):
                    # j != y_i
                    if M[N][D] <= 0:
                        G[N][D] = 0
            ## for some reason this implementation fails

            #         else:
            #             sum_falses += 1
            # G[N][y[N].item()] = -sum_falses

                else:
                    # j == y_i
                    if M[N][D] > 0:
                        G[N][D] = -1
                    else:
                        G[N][D] = 0

        # it works!  Now need to change to broadcasting

        G = G / x.shape[0]
        grad = torch.matmul(x.T, G)
        # ========================

        return grad
