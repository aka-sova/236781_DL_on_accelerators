r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
**Your answer:**

Input tensor X is 128 X 1024
Hidden layer tensor size is 1024 X 2048
Output tensor size is 128 X 2048. 

The Jacobian of the output tensor w.r.t. a single input is 1024 X 2048. 
 The Jacobian of the output tensor w.r.t. a batch of 128 inputs is 1024 X 2048 X 128 

The amount of memory required to store 1 int is 4 bytes, thus is it required 1024 X 2048 X 128 X 4 [bytes] ~= 1.073 [GB]


"""

# ==============
# Part 2 answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    wstd, lr, reg = 0.01,0.01,0.0001
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = 0, 0, 0, 0, 0

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    # hyper parameters for vanilla SGD
    #wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = 0.1, 0.1, 0, 0, 0.0003
    # hyper parameters for momentum SGD
    #wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = 0.1, 0.1, 0.01, 0, 0.001
    # hyper parameters for RMSprop
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = 0.1, 0.1, 0.01, 0.0001, 0
    # ========================
    return dict(
        wstd=wstd,
        lr_vanilla=lr_vanilla,
        lr_momentum=lr_momentum,
        lr_rmsprop=lr_rmsprop,
        reg=reg,
    )


def part2_dropout_hp():
    wstd, lr, = 0, 0
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    wstd = 0.1
    lr = 0.001
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
**Your answer:**
1. We expected test accuracy of the network with dropout to be better than the network without,
as we can see in the graphs. Dropout has the effect of making the training process noisy, 
forcing nodes within a layer to probabilistically take on more or less responsibility for the inputs,
which suggests that perhaps dropout breaks-up situations where network layers co-adapt to correct mistakes 
from prior layers, in turn making the model more robust and thus performs better on test set

2.A network with low dropout settings  performed better than a network with high dropout setting
because when the dropout rate is high, 0.8-1, the effect of forcing nodes within a layer to probabilistically 
take on more or less responsibility for the input is deminished.

"""

part2_q2 = r"""
**Your answer:**

The situation during training when , a model with the cross-entropy loss function,
test loss increases for a few epochs while the test accuracy also increases is possible.
Loss function is being calculated using actual predicted probabilities while accuracy is being 
calculated using one hot vectors thus,when predicted probability mass is distributed both test loss
and test accuracy increase.



"""
# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**

The bottleneck block emploies  a 1x1 convolution before the 3x3 convolutional layer and after
in order reduce dimension which decrease the number of feature maps whilst retaining their salient features.

# of weights for residual block: 3 * 3* 64 * 64
# of weights for bottleneck block: 1* 1 * 64 * 256



"""

part3_q2 = r"""
**Your answer:**

1.Depth can reduce number of learned units required to represent a desired function by a NN
and can reduce the amount of generalization error which will improve accuracy on test set.
L=2 gave the best 
2. A model with L =8,16 did not learned because they were numerical instble which led to exploding and vanishing gradients
change regularization term, add dropout.

"""

part3_q3 = r"""
**Your answer:**

The increase in number of filter in models with L=2,4 did improve  performance in compare to 
experiment 1.1 because it extracts more features.
When L=8 we still have the probelm of vanishing gradients

"""

part3_q4 = r"""
**Your answer:**

Differnet number of convolution filter per layer did not improve performance as the depth of layer increases,
further more from L=3 the model was unable to learn because vanishing gradient

"""

part3_q5 = r"""
**Your answer:**

When adding residual block the models with increase depth were able to learn, in comparison to experiment 
1.3 and 1.1 because residual block deals with vanished gradients which enables us to train deeper networks. 

"""

part3_q6 = r"""
**Your answer:**

In our YourCodeNet class with added batch normalization and dropout to the residual block which improve performance
in comparison to experiment 1 where we did not.

We used Batch Normalization because it has the effect of stabilizing the learning process and dramatically reducing
the number of training epochs required to train deep networks, and Dropout because it has the effect of making 
the training process noisy, forcing nodes within a layer to probabilistically 
take on more or less responsibility for the inputs,which suggests that perhaps dropout breaks-up situations 
where network layers co-adapt to correct mistakes  from prior layers, in turn making the model more robust and
thus performs better on test set.






"""
# ==============
