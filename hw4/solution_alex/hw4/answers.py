r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""


# ==============
# Part 1 answers


def part1_pg_hyperparams():
    hp = dict(
        batch_size=32, gamma=0.99, beta=0.5, learn_rate=1e-3, eps=1e-8, num_workers=2,
    )
    # TODO: Tweak the hyperparameters if needed.
    #  You can also add new ones if you need them for your model's __init__.
    # ====== YOUR CODE: ======
 
    # 1. 
    #     hp['batch_size'] = 32    
    #     hp['gamma'] = 0.9
    #     hp['beta'] = 0.1
    #     hp['learn_rate'] = 1e-3    
    #     hp['eps'] = 1e-8
    #     hp['num_workers'] = 0    
    
    hp['batch_size'] = 32    
    hp['gamma'] = 0.9
    hp['beta'] = 0.05
    hp['learn_rate'] = 1e-3    
    hp['eps'] = 1e-8
    hp['num_workers'] = 0    
    
    # ========================
    return hp


def part1_aac_hyperparams():
    hp = dict(
        batch_size=32,
        gamma=0.99,
        beta=1.0,
        delta=1.0,
        learn_rate=1e-3,
        eps=1e-8,
        num_workers=2,
    )
    # TODO: Tweak the hyperparameters. You can also add new ones if you need
    #   them for your model implementation.
    # ====== YOUR CODE: ======
    
    
#     hp['batch_size'] = 1   
#     hp['gamma'] = 0.999
#     hp['beta'] = 0.05    # multiply the loss_e
#     hp['delta'] = 0.02   # multiply the loss_v
#     hp['learn_rate'] = 0.00003
#     hp['eps'] = 1e-8
#     hp['num_workers'] = 0    
    
    hp['batch_size'] = 1   
    hp['gamma'] = 0.999
    hp['beta'] = 0.1    # multiply the loss_e
    hp['delta'] = 0.04   # multiply the loss_v
    hp['learn_rate'] = 0.00004
    hp['eps'] = 1e-8
    hp['num_workers'] = 0      
    
    # ========================
    return hp


part1_q1 = r"""
**Your answer:**

As explained in the lecture, the absolute value of the reward is of little meaning in the decision
whether to increase or decrease the probability of certain action. We need to know whether the certain action
increases the expected reward or is the best action to choose in a given state.

Policy weight, by which we multiply the log probabilities, can be huge, if we take the absolute value of the
calculated q_values, as in the case with the Vanilla PG. Those are the discounted future rewards. Their
value depends on the horizon (depth that we look forward), the reward function structure and variance, 
and all the future states.

By introducing the baseline, we still detect which actions perform good (better than baseline), but the loss is
not so big. We clearly see in the graphs.

in the case with AAC, we subtract the Value function of the state, i.e. the expected return. It includes all the
variance of the future states in it. So the $\hat{q}_{i,t}-v_\pi(s_t)$ only depends on the current state, and has
lower variance.


"""


part1_q2 = r"""
**Your answer:**


Following the hint, and remembering the definion of Value function and Q-function:


$$
\begin{align}
v_{\pi}(s) &= \E{g(\tau)|s_0 = s,\pi} \\
q_{\pi}(s,a) &= \E{g(\tau)|s_0 = s,a_0=a,\pi}.
\end{align}
$$

The only difference is that the Q-function fixes the first action. Both follow the chosen policy.
But in our approximation, the action was already chosen by the policy when this experience was created. 
So we use this Q-value as target for the Value of the state. 

"""


part1_q3 = r"""
**Your answer:**


1. In first part graphs we see:

- The loss_p starts at negative for vpg, epg and goes up. This is explainable - the q_vals in the beginning are negative.
The log_softmax values are always negative. And the loss_p has minus sign. As q_values in the experiences begin to be less 
negative, so does the loss_p.

- We can clearly see the effect of the baseline, the losses aren't big, and the variance of the policy network is much smaller

- From our experience, the CPG had a better performance over many times and had almost succeeded to reach the surface of the moon.

2. The AAC training was much more complex. Some of the important things we did to calibrate it:

- Adjust the delta, so that the loss of the Critic will be of the same scale as the loss of the policy.
- Adjust the beta, so that the "randomality" of the actions will be controlled
- Finding the appropriating network for Actor and Critic. In the end it was decided to use separate networks for each, not the 2-head network as proposed.

By looking at the graphs, we can see that:
- We were able to reach bigger rewards earlier
- The policy loss was able to reach the 0 benchmark much earlier
- On average, we received higher max award.
- In general, the learning is more stable. 

"""
