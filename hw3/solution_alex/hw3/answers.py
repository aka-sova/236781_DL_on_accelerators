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


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

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
