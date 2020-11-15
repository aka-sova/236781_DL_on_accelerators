import torch
from torch import Tensor
from collections import namedtuple
from torch.utils.data import DataLoader

from .losses import ClassifierLoss


class LinearClassifier(object):
    def __init__(self, n_features, n_classes, weight_std=0.001):
        """
        Initializes the linear classifier.
        :param n_features: Number or features in each sample.
        :param n_classes: Number of classes samples can belong to.
        :param weight_std: Standard deviation of initial weights.
        """
        self.n_features = n_features
        self.n_classes = n_classes

        # TODO:
        #  Create weights tensor of appropriate dimensions
        #  Initialize it from a normal dist with zero mean and the given std.

        self.weights = None
        # ====== YOUR CODE: ======
        self.weights = torch.normal(size = (n_features, n_classes), mean=0, std = weight_std)
        # ========================

    def predict(self, x: Tensor):
        """
        Predict the class of a batch of samples based on the current weights.
        :param x: A tensor of shape (N,n_features) where N is the batch size.
        :return:
            y_pred: Tensor of shape (N,) where each entry is the predicted
                class of the corresponding sample. Predictions are integers in
                range [0, n_classes-1].
            class_scores: Tensor of shape (N,n_classes) with the class score
                per sample.
        """

        # TODO:
        #  Implement linear prediction.
        #  Calculate the score for each class using the weights and
        #  return the class y_pred with the highest score.

        y_pred, class_scores = None, None
        # ====== YOUR CODE: ======
        class_scores = torch.matmul(x, self.weights)  # is NXC
        y_pred = torch.max(class_scores, 1).indices  # max pred class value for each sample

        # ========================

        return y_pred, class_scores

    @staticmethod
    def evaluate_accuracy(y: Tensor, y_pred: Tensor):
        """
        Calculates the prediction accuracy based on predicted and ground-truth
        labels.
        :param y: A tensor of shape (N,) containing ground truth class labels.
        :param y_pred: A tensor of shape (N,) containing predicted labels.
        :return: The accuracy in percent.
        """

        # TODO:
        #  calculate accuracy of prediction.
        #  Use the predict function above and compare the predicted class
        #  labels to the ground truth labels to obtain the accuracy (in %).
        #  Do not use an explicit loop.

        acc = None
        # ====== YOUR CODE: ======
        y_diff = (y - y_pred)
        bool_list = list(map(bool, y_diff))
        false_preds = sum(bool_list)

        acc = 1 - (false_preds / y.shape[0])
        # ========================

        return acc * 100

    def train(
        self,
        dl_train: DataLoader,
        dl_valid: DataLoader,
        loss_fn: ClassifierLoss,
        learn_rate=0.1,
        weight_decay=0.001,
        max_epochs=100,
    ):

        Result = namedtuple("Result", "accuracy loss")
        train_res = Result(accuracy=[], loss=[])
        valid_res = Result(accuracy=[], loss=[])

        print("Training", end="\n")
        for epoch_idx in range(max_epochs):
            total_correct = 0
            average_loss = 0

            # TODO:
            #  Implement model training loop.
            #  1. At each epoch, evaluate the model on the entire training set
            #     (batch by batch) and update the weights.
            #  2. Each epoch, also evaluate on the validation set.
            #  3. Accumulate average loss and total accuracy for both sets.
            #     The train/valid_res variables should hold the average loss
            #     and accuracy per epoch.
            #  4. Don't forget to add a regularization term to the loss,
            #     using the weight_decay parameter.

            # ====== YOUR CODE: ======

            total_grad = torch.zeros(self.weights.shape)
            total_train_loss = 0
            total_train_accuracy = 0
            batches_num_train = 0

            for batch in dl_train:
                # 1 batch = batc_size images and labels
                X, y = batch

                # 1. evaluate the model
                y_pred, class_scores = self.predict(X)
                accuracy = self.evaluate_accuracy(y, y_pred)
                total_train_accuracy += accuracy

                # 2. Calculate the loss
                loss = loss_fn.loss(X, y, class_scores, y_pred)
                total_train_loss += loss

                # 3. calculate the gradients for the weights
                gradients = loss_fn.grad()

                # 4. sum to the total gradients
                total_grad += gradients
                batches_num_train += 1

            train_res.loss.append(total_train_loss / batches_num_train)
            train_res.accuracy.append(total_train_accuracy / batches_num_train)

            # 5. make iteration of the weights update.
            # TODO add weight_decay
            self.weights -= learn_rate * (total_grad / batches_num_train)  + (self.weights * weight_decay)

            # 6. test on the validation set
            total_val_loss = 0
            total_val_accuracy = 0
            batches_num_val = 0

            for batch in dl_valid:
                # 1 batch = batch_size images and labels
                X, y = batch

                # 1. evaluate the model
                y_pred, class_scores = self.predict(X)
                accuracy = self.evaluate_accuracy(y, y_pred)
                total_val_accuracy += accuracy

                # 2. Calculate the loss
                loss = loss_fn.loss(X, y, class_scores, y_pred)
                total_val_loss += loss

                batches_num_val += 1


            valid_res.loss.append(total_val_loss / batches_num_val)
            valid_res.accuracy.append(total_val_accuracy / batches_num_val)

            print(f"Epoch : {epoch_idx}, "
                  f"training loss: {round(total_train_loss.item() / batches_num_train, 2)}, "
                  f"training accuracy: {round(total_train_accuracy / batches_num_train,2)} "
                  f"validation loss: {round(total_val_loss.item() / batches_num_val, 2)} "
                  f"validation accuracy: {round(total_val_accuracy / batches_num_val, 2)} ")






            # ========================
            # print(".", end="")

        print("")
        return train_res, valid_res

    def weights_as_images(self, img_shape, has_bias=True):
        """
        Create tensor images from the weights, for visualization.
        :param img_shape: Shape of each tensor image to create, i.e. (C,H,W).
        :param has_bias: Whether the weights include a bias component
            (assumed to be the first feature).
        :return: Tensor of shape (n_classes, C, H, W).
        """

        # TODO:
        #  Convert the weights matrix into a tensor of images.
        #  The output shape should be (n_classes, C, H, W).

        # ====== YOUR CODE: ======
        C, H, W = img_shape
        if has_bias:
            weights = self.weights[1:, :]
        else:
            weights = self.weights

        n_classes = weights.shape[1]
        weights = weights.T
        new_shape = (n_classes, C, H, W)
        weights_reshaped = weights.view(new_shape)

        # ========================

        return weights_reshaped


def hyperparams():
    hp = dict(weight_std=0.0, learn_rate=0.0, weight_decay=0.0)

    # TODO:
    #  Manually tune the hyperparameters to get the training accuracy test
    #  to pass.
    # ====== YOUR CODE: ======
    hp['learn_rate'] = 0.01 # 0.008
    hp['weight_std'] = 0.001
    hp['weight_decay'] = 0.001
    # ========================

    return hp
