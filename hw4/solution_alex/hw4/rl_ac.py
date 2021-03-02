import gym
import torch
import torch.nn as nn
import torch.nn.functional

from .rl_pg import TrainBatch, PolicyAgent, VanillaPolicyGradientLoss


class AACPolicyNet(nn.Module):
    def __init__(self, in_features: int, out_actions: int, **kw):
        """
        Create a model which represents the agent's policy.
        :param in_features: Number of input features (in one observation).
        :param out_actions: Number of output actions.
        :param kw: Any extra args needed to construct the model.
        """
        super().__init__()
        # TODO:
        #  Implement a dual-head neural net to approximate both the
        #  policy and value. You can have a common base part, or not.
        # ====== YOUR CODE: ======
        
        self.in_features = in_features
        self.out_actions = out_actions
        
        # BACKBONE NET
        backbone_features_list = [in_features, 100, 100, 100]
        
        backbone_layers = []
        
        for idx, _ in enumerate(backbone_features_list[:-1]):
            backbone_layers.append(nn.Linear(backbone_features_list[idx], backbone_features_list[idx+1]))
            
            if idx < len(backbone_features_list)-1:
                backbone_layers.append(nn.ReLU())
                
        backbone_net = nn.Sequential(*backbone_layers)
        
        # ACTIONS HEAD
        action_layers = []         
        action_layers.append(nn.Linear(backbone_features_list[-1], out_actions))        
        actions_head = nn.Sequential(*action_layers)
        actions_net = [backbone_net, actions_head]
        
        self.actions_net = nn.Sequential(*actions_net)
        
        # VALUE HEAD
        
        value_layers = []         
        value_layers.append(nn.Linear(backbone_features_list[-1], 1))        
        value_head = nn.Sequential(*value_layers)
        value_net = [backbone_net, value_head]
        
        self.value_net = nn.Sequential(*value_net)
        
        # ========================

    def forward(self, x):
        """
        :param x: Batch of states, shape (N,O) where N is batch size and O
        is the observation dimension (features).
        :return: A tuple of action values (N,A) and state values (N,1) where
        A is is the number of possible actions.
        """
        # TODO:
        #  Implement the forward pass.
        #  calculate both the action scores (policy) and the value of the
        #  given state.
        # ====== YOUR CODE: ======
        action_scores = self.actions_net(x)  
        state_values = self.value_net(x)
        # ========================

        return action_scores, state_values

    @staticmethod
    def build_for_env(env: gym.Env, device="cpu", **kw):
        """
        Creates a A2cNet instance suitable for the given environment.
        :param env: The environment.
        :param kw: Extra hyperparameters.
        :return: An A2CPolicyNet instance.
        """
        # TODO: Implement according to docstring.
        # ====== YOUR CODE: ======
        
        ENV_N_ACTIONS = env.action_space.n
        ENV_N_OBSERVATIONS = env.observation_space.shape[0]
        
        net = AACPolicyNet(in_features = ENV_N_OBSERVATIONS, out_actions = ENV_N_ACTIONS, **kw)
        
        # ========================
        return net.to(device)


class AACPolicyAgent(PolicyAgent):
    def current_action_distribution(self) -> torch.Tensor:
        # TODO: Generate the distribution as described above.
        # ====== YOUR CODE: ======
        
        action_scores, _ = self.p_net(self.curr_state)
        
        # turn scores into probabilities        
        actions_proba = torch.softmax(action_scores, dim=0)
        # ========================
        return actions_proba


class AACPolicyGradientLoss(VanillaPolicyGradientLoss):
    def __init__(self, delta: float):
        """
        Initializes an AAC loss function.
        :param delta: Scalar factor to apply to state-value loss.
        """
        super().__init__()
        self.delta = delta

    def forward(self, batch: TrainBatch, model_output, **kw):

        # Get both outputs of the AAC model
        action_scores, state_values = model_output

        # TODO: Calculate the policy loss loss_p, state-value loss loss_v and
        #  advantage vector per state.
        #  Use the helper functions in this class and its base.
        # ====== YOUR CODE: ======
        
        actions_log_probs = torch.log_softmax(action_scores, dim=1)
        
        # state_value loss for CRITIC
        loss_v = self._value_loss(batch, state_values)
        
        # policy loss for ACTOR
     
        advantage = self._policy_weight(batch, state_values)  # NxN
        # print(advantage.shape)        
        
        
        policy_weight = advantage
        selected_action_logprobs = actions_log_probs[torch.arange(action_scores.shape[0]), batch.actions]
        selected_action_logprobs = selected_action_logprobs.unsqueeze(dim=1)
        # print(selected_action_logprobs.shape)      
        losses_mult = policy_weight * selected_action_logprobs
        loss_p = -torch.mean(losses_mult)
        
        
        # ========================

        loss_v *= self.delta
        loss_t = loss_p + loss_v
        return (
            loss_t,
            dict(
                loss_p=loss_p.item(),
                loss_v=loss_v.item(),
                adv_m=advantage.mean().item(),
            ),
        )

    def _policy_weight(self, batch: TrainBatch, state_values: torch.Tensor):
        # TODO:
        #  Calculate the weight term of the AAC policy gradient (advantage).
        #  Notice that we don't want to backprop errors from the policy
        #  loss into the state-value network.
        # ====== YOUR CODE: ======
        
        # batch.q_vals (N,)
        # state_values (N, N)
        
        q_vals_unsqueezed = batch.q_vals.unsqueeze(dim=1)
        #         print(q_vals_unsqueezed.shape)
        #         print(state_values.shape)
        advantage = q_vals_unsqueezed - state_values        
              
        # ========================
        return advantage

    def _value_loss(self, batch: TrainBatch, state_values: torch.Tensor):
        # TODO: Calculate the state-value loss.
        # ====== YOUR CODE: ======
        
        SE = (state_values - batch.q_vals)**2
        MSE = torch.mean(SE)
        
        loss_v = MSE
        
        # ========================
        return loss_v
