# Modified from https://github.com/lmzintgraf/varibad/tree/master
"""
Based on https://github.com/ikostrikov/pytorch-a2c-ppo-acktr
"""
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Beta, TanhTransform, TransformedDistribution

from diva.components.popart import PopArt
from diva.environments import utils as utl
from diva.environments.utils import EnvLatents
from diva.utils.torch import DeviceConfig


class TanhNormal(TransformedDistribution):
    def __init__(self, base_distribution, transforms, validate_args=None):
        super().__init__(base_distribution, transforms, validate_args=None)

@property
def mean(self):
    x = self.base_dist.mean
    for transform in self.transforms:
        x = transform(x)
    return x



class Policy(nn.Module):
    def __init__(self,
                 args,
                 # input
                 pass_state_to_policy,
                 pass_latent_to_policy,
                 pass_belief_to_policy,
                 pass_task_to_policy,
                 dim_state,
                 dim_latent,
                 dim_belief,
                 dim_task,
                 # hidden
                 hidden_layers,
                 activation_function,  # tanh, relu, leaky-relu
                 policy_initialisation,  # orthogonal / normc
                 # output
                 action_space,
                 init_std,
                 state_feature_extractor=utl.FeatureExtractor,
                 state_is_dict=False,
                 use_popart=False,
                 use_beta=False,
                 logger=None
                 ):
        """
        The policy can get any of these as input:
        - state (given by environment)
        - task (in the (belief) oracle setting)
        - latent variable (from VAE)
        """
        super(Policy, self).__init__()
        assert isinstance(dim_state, dict) == state_is_dict  # TODO: redundant, choose

        self.args = args
        self.logger = logger

        if activation_function == 'tanh':
            self.activation_function = nn.Tanh()
        elif activation_function == 'relu':
            self.activation_function = nn.ReLU()
        elif activation_function == 'leaky-relu':
            self.activation_function = nn.LeakyReLU()
        else:
            raise ValueError

        if policy_initialisation == 'normc':
            init_ = lambda m: init(m, init_normc_, lambda x: nn.init.constant_(x, 0), nn.init.calculate_gain(activation_function))  # noqa: E731
        elif policy_initialisation == 'orthogonal':
            init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), nn.init.calculate_gain(activation_function))  # noqa: E731

        self.pass_state_to_policy = pass_state_to_policy
        self.pass_latent_to_policy = pass_latent_to_policy
        self.pass_task_to_policy = pass_task_to_policy
        self.pass_belief_to_policy = pass_belief_to_policy
        self.state_is_dict = state_is_dict

        # set normalisation parameters for the inputs
        # (will be updated from outside using the RL batches)
        self.norm_state = self.args.policy.norm_state and (dim_state is not None)
        if self.pass_state_to_policy and self.norm_state:
            if self.state_is_dict:
                raise NotImplementedError('Deprecated!')
                # self.state_rms = mze.DictRunningMeanStd(shapes=dim_state)
            else:
                self.state_rms = utl.RunningMeanStd(shape=(dim_state))
        self.norm_latent = self.args.policy.norm_latent and (dim_latent is not None)
        if self.pass_latent_to_policy and self.norm_latent:
            self.latent_rms = utl.RunningMeanStd(shape=(dim_latent))
        self.norm_belief = self.args.policy.norm_belief and (dim_belief is not None)
        if self.pass_belief_to_policy and self.norm_belief:
            self.belief_rms = utl.RunningMeanStd(shape=(dim_belief))
        self.norm_task = self.args.policy.norm_task and (dim_task is not None)
        if self.pass_task_to_policy and self.norm_task:
            self.task_rms = utl.RunningMeanStd(shape=(dim_task))

        # We do this confusing logic so we don't need to use else's for three 
        # out of the four if's
        curr_input_dim = dim_latent * int(self.pass_latent_to_policy) + \
                         dim_belief * int(self.pass_belief_to_policy) + \
                         dim_task * int(self.pass_task_to_policy)
        # initialise encoders for separate inputs
        self.use_state_encoder = self.args.policy.state_embedding_dim is not None
        if self.pass_state_to_policy and self.use_state_encoder:
            self.state_encoder = state_feature_extractor(dim_state, self.args.policy.state_embedding_dim, self.activation_function)
            curr_input_dim += self.args.policy.state_embedding_dim
        else:
            if self.pass_state_to_policy:
                curr_input_dim += dim_state
                # curr_input_dim += dim_state * int(self.pass_state_to_policy)
        self.use_latent_encoder = self.args.policy.latent_embedding_dim is not None
        if self.pass_latent_to_policy and self.use_latent_encoder:
            self.latent_encoder = utl.FeatureExtractor(dim_latent, self.args.policy.latent_embedding_dim, self.activation_function)
            curr_input_dim = curr_input_dim - dim_latent + self.args.policy.latent_embedding_dim
        self.use_belief_encoder = self.args.policy.belief_embedding_dim is not None
        if self.pass_belief_to_policy and self.use_belief_encoder:
            self.belief_encoder = utl.FeatureExtractor(dim_belief, self.args.policy.belief_embedding_dim, self.activation_function)
            curr_input_dim = curr_input_dim - dim_belief + self.args.policy.belief_embedding_dim
        self.use_task_encoder = self.args.policy.task_embedding_dim is not None
        if self.pass_task_to_policy and self.use_task_encoder:
            self.task_encoder = utl.FeatureExtractor(dim_task, self.args.policy.task_embedding_dim, self.activation_function)
            curr_input_dim = curr_input_dim - dim_task + self.args.policy.task_embedding_dim

        # initialise actor and critic
        hidden_layers = [int(h) for h in hidden_layers]
        self.actor_layers = nn.ModuleList()
        self.critic_layers = nn.ModuleList()
        for i in range(len(hidden_layers)):
            fc = init_(nn.Linear(curr_input_dim, hidden_layers[i]))
            self.actor_layers.append(fc)
            fc = init_(nn.Linear(curr_input_dim, hidden_layers[i]))
            self.critic_layers.append(fc)
            curr_input_dim = hidden_layers[i]
        # self.critic_linear = init_(nn.Linear(hidden_layers[-1], 1))

        # Popart on value head 
        if use_popart:
            self.critic_linear = PopArt(hidden_layers[-1], 1)
            self.popart = self.critic_linear
        else:
            self.critic_linear = nn.Linear(hidden_layers[-1], 1)
            self.popart = None

        # output distributions of the policy
        self.use_beta = use_beta
        self.action_space = action_space
        if self.use_beta:
            self.action_low = torch.tensor(self.action_space.low, dtype=torch.float32, device=DeviceConfig.DEVICE)
            self.action_high = torch.tensor(self.action_space.high, dtype=torch.float32, device=DeviceConfig.DEVICE)

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(hidden_layers[-1], num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            if self.use_beta:
                self.fc_alpha = nn.Linear(hidden_layers[-1], num_outputs)
                self.fc_beta = nn.Linear(hidden_layers[-1], num_outputs)
                self.softplus = nn.Softplus()
            else:
                self.dist = DiagGaussian(hidden_layers[-1], num_outputs, init_std, self.args.policy.norm_actions_pre_sampling)
        else:
            raise NotImplementedError

    def get_actor_params(self):
        return [*self.actor.parameters(), *self.dist.parameters()]

    def get_critic_params(self):
        return [*self.critic.parameters(), *self.critic_linear.parameters()]

    def forward_actor(self, inputs):
        h = inputs
        for i in range(len(self.actor_layers)):
            h = self.actor_layers[i](h)
            h = self.activation_function(h)
        return h

    def forward_critic(self, inputs):
        h = inputs
        for i in range(len(self.critic_layers)):
            h = self.critic_layers[i](h)
            h = self.activation_function(h)
        return h

    def forward(self, state, latent, belief, task):
        # Handle inputs (normalise + embed)

        if self.pass_state_to_policy:
            if self.norm_state:
                if self.state_is_dict:
                    # Image should either be shape (H, W, C) or (..., BS, H, W, C)
                    # and broadcasting will be done automatically in batch case
                    # just as below
                    for k in state.keys():
                        state[k] = ((state[k] - self.state_rms[k].mean) / 
                                    torch.sqrt(self.state_rms[k].var + 1e-8))
                else:
                    # State is either shape (N) or (..., BS, N)
                    # self.state_rms.mean/var are shape (N)
                    # Broadcasting will be done automatically in batch case
                    state = (state - self.state_rms.mean) / torch.sqrt(self.state_rms.var + 1e-8)

            if self.use_state_encoder:
                state = self.state_encoder(state)
            else:
                # If state is dict, we need to encode
                assert not self.state_is_dict
        else:
            state = torch.zeros(0, device=DeviceConfig.DEVICE)
        
        if self.pass_latent_to_policy:
            if self.norm_latent:
                latent = (latent - self.latent_rms.mean) / torch.sqrt(self.latent_rms.var + 1e-8)
            if self.use_latent_encoder:
                latent = self.latent_encoder(latent)
        else:
            latent = torch.zeros(0, device=DeviceConfig.DEVICE)
        
        if self.pass_belief_to_policy:
            if self.norm_belief:
                belief = (belief - self.belief_rms.mean) / torch.sqrt(self.belief_rms.var + 1e-8)
            if self.use_belief_encoder:
                belief = self.belief_encoder(belief.float())
        else:
            belief = torch.zeros(0, device=DeviceConfig.DEVICE)
        
        if self.pass_task_to_policy:
            if self.norm_task:
                task = (task - self.task_rms.mean) / torch.sqrt(self.task_rms.var + 1e-8)
            if self.use_task_encoder:
                task = self.task_encoder(task.float())
        else:
            task = torch.zeros(0, device=DeviceConfig.DEVICE)

        # Handle case where there is only one process
        if len(state.shape) != 1 and max(len(latent.shape), len(belief.shape), len(task.shape)) == 1:
            # Remove extra dimension from state
            # TODO: Fix this case in a more elegant way
            print('WARNING: shape mismatch in Policy.forward()')
            state = state.squeeze(0)
        # Concatenate inputs (NOTE: state is always already encoded here if dict)
        inputs = torch.cat((state, latent, belief, task), dim=-1)

        # Forward through critic/actor part
        hidden_critic = self.forward_critic(inputs)
        hidden_actor = self.forward_actor(inputs)
        return self.critic_linear(hidden_critic), hidden_actor

    def act(self, state, latent, belief, task, deterministic=False, return_stats=False):
        """
        Returns the (raw) actions and their value.
        """
        value, actor_features = self.forward(state=state, latent=latent, belief=belief, task=task)
        
        # Initialize the distribution based on the type (assuming self.use_beta determines if we use Beta)
        if self.use_beta:
            # +1 is for numerical stability
            alpha = 1 + self.softplus(self.fc_alpha(actor_features))
            beta = 1 + self.softplus(self.fc_beta(actor_features))
            dist = Beta(alpha, beta)
        else:
            dist = self.dist(actor_features)
        
        # Determine action based on whether deterministic is required
        if deterministic:
            if isinstance(dist, FixedCategorical):
                action = dist.mode()  # For categorical actions
            else:
                action = dist.mean  # For continuous actions like Beta or Gaussian
        else:
            action = dist.sample()

        # If using Beta distribution, scale the action to the desired range
        if self.use_beta:
            action = self.action_low + (self.action_high - self.action_low) * action

        # Optionally return additional statistics about the actor's output
        if return_stats:
            stats = {
                'actor_features_mean': actor_features.mean(),
                'actor_features_std': actor_features.std(),
                'actor_features_min': actor_features.min(),
                'actor_features_max': actor_features.max(),
                'actor_features_shape': actor_features.shape
            }
            return value, action, stats
        else:
            return value, action

    def get_value(self, state, latent, belief, task):
        value, _ = self.forward(state, latent, belief, task)
        return value

    def update_rms(self, args, policy_storage):
        """ Update normalisation parameters for inputs with current data """
        if self.pass_state_to_policy and self.norm_state:
            if self.state_is_dict:
                for k in policy_storage.prev_state.keys():
                    self.state_rms[k].update(policy_storage.prev_state[k][:-1])
            else:
                self.state_rms.update(policy_storage.prev_state[:-1])
        if self.pass_latent_to_policy and self.norm_latent:
            env_latents = EnvLatents(
                z_samples=torch.cat(policy_storage.latent_samples[:-1]),
                z_means=torch.cat(policy_storage.latent_mean[:-1]),
                z_logvars=torch.cat(policy_storage.latent_logvar[:-1]),
                hs=None)
            latent = utl.get_zs_for_policy(args, env_latents)
            self.latent_rms.update(latent)
        if self.pass_belief_to_policy and self.norm_belief:
            self.belief_rms.update(policy_storage.beliefs[:-1])
        if self.pass_task_to_policy and self.norm_task:
            self.task_rms.update(policy_storage.tasks[:-1])

    def evaluate_actions(self, state, latent, belief, task, action):

        value, actor_features = self.forward(state, latent, belief, task)

        # Initialize the distribution based on the type (assuming self.use_beta determines if we use Beta)
        if self.use_beta:
            # +1 is for numerical stability
            alpha = 1 + self.softplus(self.fc_alpha(actor_features))
            beta = 1 + self.softplus(self.fc_beta(actor_features))
            dist = Beta(alpha, beta)
            # Scale action from [low, high] to [0, 1] for Beta distribution evaluation
            scaled_action = (action - self.action_low) / (self.action_high - self.action_low)
        else:
            dist = self.dist(actor_features)

        if isinstance(dist, FixedNormal):  # Assuming DiagGaussian returns FixedNormal distribution
            action_log_probs = dist.log_prob(action).sum(-1, keepdim=True)
            
            # For Gaussian distribution, we don't have logits. But if you need to return something similar,
            # you can return the mean or some other representation. 
            # Here, I'm returning the mean just as an example.
            action_log_dist = dist.mean

            if self.args.policy.norm_actions_post_sampling:
                transformation = TanhTransform(cache_size=1)
                dist = TanhNormal(dist, transformation)
                action = transformation(action)
                
                # Safe log probs
                action_log_probs = dist.log_prob(action)
                clamped_log_probs = torch.clamp(action_log_probs, min=-20)
                action_log_probs = clamped_log_probs.sum(-1, keepdim=True)

                # As for logits, again you can return the mean or some representation.
                action_log_dist = dist.mean
                # entropy of underlying dist (isn't correct but works well in practice)
                dist_entropy = dist.base_dist.entropy().mean()
            else:
                dist_entropy = dist.entropy().mean()

        elif isinstance(dist, FixedCategorical):  # Assuming Categorical returns FixedCategorical distribution
            action_log_probs = dist.log_probs(action)
            action_log_dist = dist.logits  # For categorical, logits are appropriate
            dist_entropy = dist.entropy().mean()

        elif isinstance(dist, Beta):
            # Evaluate log probability for scaled actions
            action_log_probs = dist.log_prob(scaled_action)
            clamped_log_probs = torch.clamp(action_log_probs, min=-20)
            action_log_probs = clamped_log_probs.sum(-1, keepdim=True)

            action_log_dist = dist.mean  # This is already in [0, 1], representing the unscaled mean
            dist_entropy = dist.entropy().mean()

        else:
            raise NotImplementedError

        return value, action_log_probs, action_log_dist, dist_entropy

FixedCategorical = torch.distributions.Categorical

old_sample = FixedCategorical.sample
FixedCategorical.sample = lambda self: old_sample(self).unsqueeze(-1)

log_prob_cat = FixedCategorical.log_prob
FixedCategorical.log_probs = lambda self, actions: log_prob_cat(self, actions.squeeze(-1)).unsqueeze(-1)

FixedCategorical.mode = lambda self: self.probs.argmax(dim=-1, keepdim=True)

FixedNormal = torch.distributions.Normal
log_prob_normal = FixedNormal.log_prob
FixedNormal.log_probs = lambda self, actions: log_prob_normal(self, actions).sum(-1, keepdim=True)

entropy = FixedNormal.entropy
FixedNormal.entropy = lambda self: entropy(self).sum(-1)

FixedNormal.mode = lambda self: self.mean


def init(module, weight_init, bias_init, gain=1.0):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


# https://github.com/openai/baselines/blob/master/baselines/common/tf_util.py#L87
def init_normc_(weight, gain=1):
    weight.normal_(0, 1)
    weight *= gain / torch.sqrt(weight.pow(2).sum(1, keepdim=True))


class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()

        init_ = lambda m: init(m,  # noqa: E731
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0),
                               gain=0.01)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedCategorical(logits=x)


class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs, init_std, norm_actions_pre_sampling):
        super(DiagGaussian, self).__init__()

        init_ = lambda m: init(m,  # noqa: E731
                               init_normc_,
                               lambda x: nn.init.constant_(x, 0))

        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.logstd = nn.Parameter(np.log(torch.zeros(num_outputs) + init_std))
        self.norm_actions_pre_sampling = norm_actions_pre_sampling
        self.min_std = torch.tensor([1e-6], device=DeviceConfig.DEVICE)

    def forward(self, x):

        action_mean = self.fc_mean(x)
        if self.norm_actions_pre_sampling:
            action_mean = torch.tanh(action_mean)
        std = torch.max(self.min_std, self.logstd.exp())
        dist = FixedNormal(action_mean, std)

        return dist


class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().reshape(1, -1)
        else:
            bias = self._bias.t().reshape(1, -1, 1, 1)

        return x + bias
