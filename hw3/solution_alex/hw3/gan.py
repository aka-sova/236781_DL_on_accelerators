from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader


class Discriminator(nn.Module):
    def __init__(self, in_size):
        """
        :param in_size: The size of on input image (without batch dimension).
        """
        super().__init__()
        self.in_size = in_size
        # TODO: Create the discriminator model layers.
        #  To extract image features you can use the EncoderCNN from the VAE
        #  section or implement something new.
        #  You can then use either an affine layer or another conv layer to
        #  flatten the features.
        # ====== YOUR CODE: ======
        modules = []
        in_channels = in_size[0]
        n = 64
        # first layer
        modules.append(nn.Conv2d(in_channels,out_channels=n,kernel_size=3,stride=1, padding=1))
        modules.append(nn.BatchNorm2d(n))
        modules.append(nn.MaxPool2d(2))
        modules.append(nn.LeakyReLU())
        # second layer
        modules.append(nn.Conv2d(in_channels=n,out_channels=n*2,kernel_size=3,stride=1, padding=1))
        modules.append(nn.BatchNorm2d(n*2))
        modules.append(nn.MaxPool2d(4))
        modules.append(nn.LeakyReLU())
        # third layer
        modules.append(nn.Conv2d(in_channels=n*2,out_channels=n*4,kernel_size=3,stride=1, padding=1))
        modules.append(nn.BatchNorm2d(n*4))
        modules.append(nn.MaxPool2d(8))
        modules.append(nn.LeakyReLU())
        # fourth layer
        modules.append(nn.Conv2d(in_channels=n*4,out_channels=n*8,kernel_size=1))
        modules.append(nn.LeakyReLU())
        # output layer
        modules.append(nn.Conv2d(in_channels=n*8,out_channels=1,kernel_size=1))
        modules.append(nn.Sigmoid())

        self.cnn = nn.Sequential(*modules)

        # ========================

    def forward(self, x):
        """
        :param x: Input of shape (N,C,H,W) matching the given in_size.
        :return: Discriminator class score (not probability) of
        shape (N,).
        """
        # TODO: Implement discriminator forward pass.
        #  No need to apply sigmoid to obtain probability - we'll combine it
        #  with the loss due to improved numerical stability.
        # ====== YOUR CODE: ======

        y = self.cnn(x)
        y = y.reshape(-1,1)

        # ========================
        return y



class Generator(nn.Module):
    def __init__(self, z_dim, featuremap_size=4, out_channels=3):
        """
        :param z_dim: Dimension of latent space.
        :featuremap_size: Spatial size of first feature map to create
        (determines output size). For example set to 4 for a 4x4 feature map.
        :out_channels: Number of channels in the generated image.
        """
        super().__init__()
        self.z_dim = z_dim

        # TODO: Create the generator model layers.b
        #  To combine image features you can use the DecoderCNN from the VAE
        #  section or implement something new.
        #  You can assume a fixed image size.
        # ====== YOUR CODE: ======
        modules = []
        pDrop = 0.0
        biases = True

        # first layer
        modules.append(nn.ConvTranspose2d(z_dim,256,4,stride = 2, padding = 1, output_padding=0, bias = biases))
        modules.append(nn.BatchNorm2d(256))
        modules.append(nn.Dropout2d(pDrop))
        modules.append(nn.LeakyReLU())
        # second layer
        modules.append(nn.ConvTranspose2d(256,128,4,stride = 2, padding = 1, output_padding=0))
        modules.append(nn.BatchNorm2d(128))
        modules.append(nn.Dropout2d(pDrop))
        modules.append(nn.LeakyReLU())
        # third layer
        modules.append(nn.ConvTranspose2d(128,64,4,stride = 2, padding = 1, output_padding=0, bias = biases))
        modules.append(nn.BatchNorm2d(64))
        modules.append(nn.LeakyReLU())
        #
        modules.append(nn.ConvTranspose2d(64,64,4,stride = 2, padding = 1, output_padding=0, bias = biases))
        modules.append(nn.BatchNorm2d(64))
        modules.append(nn.LeakyReLU())
        #
        modules.append(nn.ConvTranspose2d(64,64,4,stride = 2, padding = 1, output_padding=0, bias = biases))
        modules.append(nn.BatchNorm2d(64))
        modules.append(nn.LeakyReLU())
        #
        modules.append(nn.ConvTranspose2d(64,out_channels,4,stride = 2, padding = 1, output_padding=0, bias = False))
        modules.append(nn.Tanh())


        self.cnn = nn.Sequential(*modules)

        # ========================

    def sample(self, n, with_grad = False):
        """
        Samples from the Generator.
        :param n: Number of instance-space samples to generate.
        :param with_grad: Whether the returned samples should be part of the
        generator's computation graph or standalone tensors (i.e. should be
        be able to backprop into them and compute their gradients).
        :return: A batch of samples, shape (N,C,H,W).
        """
        device = next(self.parameters()).device
        # TODO: Sample from the model.
        #  Generate n latent space samples and return their reconstructions.
        #  Don't use a loop.
        # ====== YOUR CODE: ======
        
        if with_grad:
            z = torch.randn(n, self.z_dim, requires_grad = True).to(device)
            samples = self.forward(z)
        else:
            with torch.no_grad():
                z = torch.randn(n, self.z_dim).to(device)
                samples = self.forward(z).cpu()

        # ========================
        return samples

    def forward(self, z):
        """
        :param z: A batch of latent space samples of shape (N, latent_dim).
        :return: A batch of generated images of shape (N,C,H,W) which should be
        the shape which the Discriminator accepts.
        """
        # TODO: Implement the Generator forward pass.
        #  Don't forget to make sure the output instances have the same
        #  dynamic range as the original (real) images.
        # ====== YOUR CODE: ======
        z = z.reshape(z.shape[0], z.shape[1], 1, 1)
        x = self.cnn(z)

        # ========================
        return x

def discriminator_loss_fn(y_data, y_generated, data_label=0, label_noise=0.0):
    """
    Computes the combined loss of the discriminator given real and generated
    data using a binary cross-entropy metric.
    This is the loss used to update the Discriminator parameters.
    :param y_data: Discriminator class-scores of instances of data sampled
    from the dataset, shape (N,).
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :param label_noise: The range of the noise to add. For example, if
    data_label=0 and label_noise=0.2 then the labels of the real data will be
    uniformly sampled from the range [-0.1,+0.1].
    :return: The combined loss of both.
    """
    assert data_label == 1 or data_label == 0
    # TODO:
    #  Implement the discriminator loss.
    #  See pytorch's BCEWithLogitsLoss for a numerically stable implementation.
    # ====== YOUR CODE: ======


    crit = nn.BCEWithLogitsLoss()
    rand_ = torch.rand_like(y_data)

    data_label_rnd = rand_*label_noise-label_noise/2+data_label
    
    genr_label = 1-data_label
    genr_label_rnd = torch.rand_like(y_data)*label_noise-label_noise/2+genr_label

    loss_data = crit(y_data, data_label_rnd)
    loss_generated = crit(y_generated, genr_label_rnd)

    real_acc = abs(y_data).mean().item()-0.3
    fake_acc = abs(y_generated).mean().item()
    if real_acc>0.5:
        if real_acc>fake_acc:
            loss_data = loss_data*0
            loss_generated = loss_generated*0


    # ========================
    return loss_data + loss_generated


def generator_loss_fn(y_generated, data_label=0):
    """
    Computes the loss of the generator given generated data using a
    binary cross-entropy metric.
    This is the loss used to update the Generator parameters.
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :return: The generator loss.
    """
    assert data_label == 1 or data_label == 0
    # TODO:
    #  Implement the Generator loss.
    #  Think about what you need to compare the input to, in order to
    #  formulate the loss in terms of Binary Cross Entropy.
    # ====== YOUR CODE: ======
    data_label_ = data_label*torch.ones_like(y_generated)
    crit = nn.BCEWithLogitsLoss()
    loss = crit(y_generated, data_label_)

    # ========================
    return loss


def train_batch(dsc_model: Discriminator, gen_model: Generator,
                dsc_loss_fn: Callable, gen_loss_fn: Callable,
                dsc_optimizer: Optimizer, gen_optimizer: Optimizer,
                x_data: DataLoader):
    """
    Trains a GAN for over one batch, updating both the discriminator and
    generator.
    :return: The discriminator and generator losses.
    """

    # TODO: Discriminator update
    #  1. Show the discriminator real and generated data
    #  2. Calculate discriminator loss
    #  3. Update discriminator parameters
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================

    # TODO: Generator update
    #  1. Show the discriminator generated data
    #  2. Calculate generator loss
    #  3. Update generator parameters
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================

    return dsc_loss.item(), gen_loss.item()


def save_checkpoint(gen_model, dsc_losses, gen_losses, checkpoint_file):
    """
    Saves a checkpoint of the generator, if necessary.
    :param gen_model: The Generator model to save.
    :param dsc_losses: Avg. discriminator loss per epoch.
    :param gen_losses: Avg. generator loss per epoch.
    :param checkpoint_file: Path without extension to save generator to.
    """

    saved = False
    checkpoint_file = f'{checkpoint_file}.pt'

    # TODO:
    #  Save a checkpoint of the generator model. You can use torch.save().
    #  You should decide what logic to use for deciding when to save.
    #  If you save, set saved to True.
    # ====== YOUR CODE: ======
    raise NotImplementedError()

    # ===========================
    return saved