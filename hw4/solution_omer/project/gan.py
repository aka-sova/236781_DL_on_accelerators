from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.autograd import Variable


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
        #modules.append(nn.BatchNorm2d(n))
        modules.append(nn.MaxPool2d(2))
        modules.append(nn.ReLU())
        # second layer
        modules.append(nn.Conv2d(in_channels=n,out_channels=n*2,kernel_size=3,stride=1, padding=1))
        modules.append(nn.BatchNorm2d(n*2))
        modules.append(nn.MaxPool2d(4))
        modules.append(nn.ReLU())
        # third layer
        modules.append(nn.Conv2d(in_channels=n*2,out_channels=n*4,kernel_size=3,stride=1, padding=1))
        modules.append(nn.BatchNorm2d(n*4))
        modules.append(nn.MaxPool2d(8))
        modules.append(nn.ReLU())
        # fourth layer
        modules.append(nn.Conv2d(in_channels=n*4,out_channels=n*8,kernel_size=1))
        modules.append(nn.ReLU())
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
        modules.append(nn.LeakyReLU(0.2))
        # second layer
        modules.append(nn.ConvTranspose2d(256,128,4,stride = 2, padding = 1, output_padding=0))
        modules.append(nn.BatchNorm2d(128))
        modules.append(nn.Dropout2d(pDrop))
        modules.append(nn.LeakyReLU(0.2))
        # third layer
        modules.append(nn.ConvTranspose2d(128,64,4,stride = 2, padding = 1, output_padding=0, bias = biases))
        modules.append(nn.BatchNorm2d(64))
        modules.append(nn.LeakyReLU(0.2))
        #
        modules.append(nn.ConvTranspose2d(64,64,4,stride = 2, padding = 1, output_padding=0, bias = biases))
        modules.append(nn.BatchNorm2d(64))
        modules.append(nn.LeakyReLU(0.2))
        #
        modules.append(nn.ConvTranspose2d(64,64,4,stride = 2, padding = 1, output_padding=0, bias = biases))
        modules.append(nn.BatchNorm2d(64))
        modules.append(nn.LeakyReLU(0.2))
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
    real_pred = dsc_model(x_data)

    fake_data = gen_model.sample(x_data.shape[0], with_grad=True).to(x_data.device)
    fake_pred = dsc_model(fake_data.detach())
    fake_acc = abs(1 - fake_pred).sum().item() / len(fake_pred)

    dsc_loss = dsc_loss_fn(real_pred, fake_pred)
    real_acc = abs(real_pred).sum().item() / len(real_pred)

    dsc_optimizer.zero_grad()
    dsc_loss.backward()
    dsc_optimizer.step()
    # raise NotImplementedError()
    # ========================

    # TODO: Generator update
    #  1. Show the discriminator generated data
    #  2. Calculate generator loss
    #  3. Update generator parameters
    # ====== YOUR CODE: ======
    reps = max(1, int(10 * (real_acc - fake_acc)))
    for i in range(reps):
        fake_data = gen_model.sample(x_data.shape[0], with_grad=True).to(x_data.device)
        fake_pred = dsc_model(fake_data)

        gen_loss = gen_loss_fn(fake_pred)
        fake_acc = abs(fake_pred).sum().item() / len(fake_pred)

        gen_optimizer.zero_grad()
        gen_loss.backward(retain_graph=True)
        gen_optimizer.step()

    return fake_acc, real_acc
    # raise NotImplementedError()
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
    import os
    dirname = os.path.dirname("checkpoints/gan.pt")
    os.makedirs(dirname, exist_ok=True)

    if len(gen_losses) > 1:
        temp1 = gen_losses[-1] - dsc_losses[-1]

        gen_losses = torch.Tensor(gen_losses)
        dsc_losses = torch.Tensor(dsc_losses)

        temp2 = torch.mean(gen_losses - dsc_losses).item()

        if (temp1 < temp2):
            torch.save(gen_model, checkpoint_file)
            saved = True
    #raise NotImplementedError()

    # ===========================
    return saved

class Spectral_norm_Discriminator(nn.Module):
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
        "In each 2D convolution we preform spectral normalization, and no complimentary regularization techniques" \
        "e.g. batch normalization, weight decay and reature matching"
        # first layer
        modules.append(spectral_norm(nn.Conv2d(in_channels,out_channels=n,kernel_size=3,stride=1, padding=1)))
        #modules.append(nn.BatchNorm2d(n))
        modules.append(nn.MaxPool2d(2))
        modules.append(nn.ReLU())
        # second layer
        modules.append(spectral_norm(nn.Conv2d(in_channels=n,out_channels=n*2,kernel_size=3,stride=1, padding=1)))
        #modules.append(nn.BatchNorm2d(n*2))
        modules.append(nn.MaxPool2d(4))
        modules.append(nn.ReLU())
        # third layer
        modules.append(spectral_norm(nn.Conv2d(in_channels=n*2,out_channels=n*4,kernel_size=3,stride=1, padding=1)))
        #modules.append(nn.BatchNorm2d(n*4))
        modules.append(nn.MaxPool2d(8))
        modules.append(nn.ReLU())
        # fourth layer
        modules.append(spectral_norm(nn.Conv2d(in_channels=n*4,out_channels=n*8,kernel_size=1)))
        modules.append(nn.ReLU())
        # output layer
        modules.append(spectral_norm(nn.Conv2d(in_channels=n*8,out_channels=1,kernel_size=1)))
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

def Wasserstein_train_batch(dsc_model: Discriminator, gen_model: Generator,
                dsc_optimizer: Optimizer, gen_optimizer: Optimizer,
                x_data: DataLoader,n_critic):
    """
        Trains a Wasserstein GAN for over one batch, updating the discriminator more times than the generator.
        n_critic: the ratio training between the discriminator and the generator
        dsc_optimizer, gen_optimizer : torch.optim.RMSprop(gen.parameters(), lr=lr)
        :return: The discriminator and generator losses.
        """
    batches_done = 0
    for i, (imgs, _) in enumerate(x_data):

        # Configure input
        real_imgs = Variable(imgs.type(torch.FloatTensor))

        "Train Discriminator"

        dsc_optimizer.zero_grad()

        # Sample noise as generator input
        z = Variable(torch.FloatTensor(np.random.normal(0, 1, (imgs.shape[0], 100))))

        # Generate a batch of images
        fake_imgs = gen_model(z).detach()
        # Adversarial loss
        loss_D = -torch.mean(dsc_model(real_imgs)) + torch.mean(dsc_model(fake_imgs))

        loss_D.backward()
        dsc_optimizer.step()

        # Clip weights of discriminator
        for p in dsc_model.parameters():
            p.data.clamp_(-0.01, 0.01)

        # Train the generator every n_critic iterations
        if i % n_critic == 0:

            "Train Generator"

            gen_optimizer.zero_grad()
            # Generate a batch of images
            gen_imgs = gen_model(z)
            # Adversarial loss
            loss_G = -torch.mean(dsc_model(gen_imgs))
            loss_G.backward()
            gen_optimizer.step()
        batches_done += 1

    return loss_D.item(),loss_G.item()
