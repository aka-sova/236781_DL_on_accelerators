r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers


def part1_rnn_hyperparams():
    hypers = dict(
        batch_size=0,
        seq_len=0,
        h_dim=0,
        n_layers=0,
        dropout=0,
        learn_rate=0.0,
        lr_sched_factor=0.0,
        lr_sched_patience=0,
    )
    # TODO: Set the hyperparameters to train the model.
    # ====== YOUR CODE: ======
    
    hypers["batch_size"] = 128
    hypers["seq_len"] = 50
    hypers["h_dim"] = 256
    hypers["n_layers"] = 3
    hypers["dropout"] = 0.2
    hypers["learn_rate"] = 0.001
    hypers["lr_sched_factor"] = 0.8
    hypers["lr_sched_patience"] = 10
    
    
    
    # ========================
    return hypers


def part1_generation_params():
    start_seq = ""
    temperature = 0.0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    start_seq = "ACT I. The rise of the machine:"
    temperature = 0.5
    # ========================
    return start_seq, temperature


part1_q1 = r"""
**Your answer:**
Several reasons:

1. Training on the whole set physically doesn't let us to divice the training set into batches to parallelize the learning.
2. The longer the sequence, the harder it is to train, since there are longer 'chains' of gradients. 
    By saving the hidden state, and passing is on, we cut the 'gradient chains' (by using .detach operation)
3. There is no need for such long chains because of the problem of vanishing and exploding gradients.
4. Also thinking logically - the character at the beginning of a corpus will likely not have an effect on the character in the end of the corpus


"""

part1_q2 = r"""
**Your answer:**

When generating text, we generate a single character each time. But we save the hidden state and pass it on.
This allows us to generate sequences of any length, and it has nothing to do with the sequence length. 
The sequence length parameter only influenced the 'learning' part.

"""

part1_q3 = r"""
**Your answer:**

Because we deliberately ordered them in a specific way such that the batches sequences will continue the previous batches sequences.
If we shuffle, there will be no connections between sequences in a specifix batch idx. 

"""

part1_q4 = r"""
**Your answer:**

1. At learning, we want the model to much more likely give the correct prediction, so we want to maximize the 'original' softmax value.
    At generating sequence, we want want less randomness so that the generated context will be consisted of real words, and not some
    unconnected characters which got sampled by chance. 
2. when the temperature is high, the probabilities to sample the letters get "flattened", thus making our choice of the next character more random
3. when the temperature is low, the probabilities get increased accoridng to their value, meaning the higher the probability, the more it will be increased. 
    This means that the choices for the next letter will be less random, and it will tend to choose the character with highest probability even more.

"""
# ==============


# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0, h_dim=0, z_dim=0, x_sigma2=0, learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======

    # those were quiet good
    
    hypers['batch_size'] = 5
    hypers['h_dim'] = 1000
    hypers['z_dim'] = 10
    hypers['x_sigma2'] = 0.002
    hypers['learn_rate'] = 0.0002
    hypers['betas'] = (0.99, 0.99)    

    
    # ========================
    return hypers


part2_q1 = r"""
**Your answer:**

The sigma2 parameter is a regularization parameter between the 2 main loss components:
1. The similarity component (reconstruction term) between the original image and the image after passing through the encoder-decoder network
2. The DL component - or the similarity of the latent space probability distribution to the Normal distribution with mean of 0 and variance of 1 - makes the latent space regular


If it's high - the (1) term becomes less dominant, and the mappings become more random, and less resembling the actual original image. But it guarantees the latent space to be continuous (two close points in latent space give approximately same decoded image) and complete (it does give meaningful content).
If it's low - less randomness is allowed in the decoder mapping from Z to X, the (1) is more dominant, and the latent space distibutions will be 
    spreaded further away from each other, allowing less overlap. Then, the latent space will be less continuous and complete, not allowing us to generate new images easily. Sampling from latent space will create images with undesirable artifacts


"""

part2_q2 = r"""
**Your answer:**


1. Explanation:
- reconstruction term - how similar is the original image to the image after being processed through the VAE
- KL divergence loss - The similarity of the latent space probability distribution to the Standard Normal distribution (mean = 0, std = 1)


2. If the latent space distibution is not similar to the Standard Normal distribution, this loss will be high, and it will penalize those layers which create those distributions from the last output layer of the encoder network. 


3. As explained in q.1, it guarantees completelness (it does give meaningful content for any point sampled in the latent space) and continuity (two close points in latent space give approximately same decoded image). By saying simply, the similar original images in the instance space should also be similar in the latent space. And there should be no 'gaps' between the distributions in the latent space.

"""

part2_q3 = r"""
**Your answer:**

Our objective is first of all to maximize the probability of the training set.
But the direct computation is infeasible. 

But in the encoder part, 
By constraining the posterior distribution $q_{\alpha} (Z|X)$ to be similar to the true posterios $P(Z|X)$, we reach the
expression which allows us to express the lower bound for $P(X)$. By minimizing it, we minimize the evidence.

"""

part2_q4 = r"""
**Your answer:**

The variance squared has to be a positive number. But simply connecting to the MLP will also output positive and negative numbers. So this is the way to make it positive while keeping it differentiable. 

"""

# ==============

# ==============
# Part 3 answers

PART3_CUSTOM_DATA_URL = None


def part3_gan_hyperparams():
    hypers = dict(
        batch_size=0,
        z_dim=0,
        data_label=0,
        label_noise=0.0,
        discriminator_optimizer=dict(
            type="",  # Any name in nn.optim like SGD, Adam
            lr=0.0,
            # You an add extra args for the optimizer here
        ),
        generator_optimizer=dict(
            type="",  # Any name in nn.optim like SGD, Adam
            lr=0.0,
            # You an add extra args for the optimizer here
        ),
    )
    # TODO: Tweak the hyperparameters to train your GAN.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return hypers


part3_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============
