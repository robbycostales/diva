# Modified from https://github.com/lmzintgraf/varibad/tree/master
"""
Based on https://github.com/ikostrikov/pytorch-a2c-ppo-acktr

Used for on-policy rollout storages.
"""
import argparse
from typing import Any, Dict, Tuple

import gym
import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from diva.environments import utils as utl
from diva.environments.utils import EnvLatents
from diva.utils.torch import DeviceConfig


def _flatten_helper(T, N, _tensor):
    return _tensor.reshape(T * N, *_tensor.size()[2:])


class OnlineStorage(object):
    def __init__(
            self,
            args: argparse.Namespace, 
            model: torch.nn.Module,
            num_steps: int, 
            num_processes: int,
            state_dim: Any,
            belief_dim: int, 
            task_dim: int,
            action_space: gym.Space,
            hidden_size: int, 
            latent_dim: int, 
            normalise_rewards: bool,
            use_gae: bool, 
            gamma: float, 
            tau: float, 
            use_proper_time_limits=True,
            use_popart: bool = False,
            add_exploration_bonus: bool = False,
            intrinsic_reward: Any = None,
    ):
        """
        Stores data collected during a rollout.

        Args:
            args (argparse.Namespace): Arguments.
            num_steps (int): Number of steps to store.
            num_processes (int): Number of parallel processes.
            state_dim (int or tuple): Dimensionality of state space.
            belief_dim (int): Dimensionality of belief space.
            task_dim (int): Dimensionality of task space.
            action_space (gym.Space): Action space.
            hidden_size (int): Size of hidden RNN states.
            latent_dim (int): Dimensionality of latent space (of VAE).
            normalise_rewards (bool): Whether to normalise rewards or not.
            use_gae (bool): Whether to use Generalised Advantage Estimation.
            gamma: (float): The discount factor.
            tau: (float): The GAE parameter.
            use_proper_time_limits: (bool): Whether to use proper time limits.
            use_popart (bool): Whether to use pop-art for reward normalisation.
            add_exploration_bonus (bool): Whether to add exploration bonus to
                the reward.
            intrinsic_reward (Any): Intrinsic reward object.
                        use_gae: (bool): Whether to use Generalised Advantage Estimation.

        > What is the format of these buffers?

        prev_state (N+1) :    [  s_0* |  s_1  |  ...  |  s_{N-1}  |  s_N  ]
        beliefs    (N+1) :    [  b_0^ |  b_1  |  ...  |  b_{N-1}  |  b_N  ]
        rewards    (N)   :    [  r_0  |  r_1  |  ...  |  r_{N-1}  ]   
        actions    (N)   :    [  a_0  |  a_1  |  ...  |  a_{N-1}  ]  
        masks      (N+1) :    [   0 ^ |   0   |  ...  |     0     |  0/1  ]  
        next_state (N)   :    [  s_1  |  s_2  |  ...  |    s_N    ] 
        latents    (N+1) :    [  l_0  |  l_1  |  ...  |  l_{N-1}  |  l_N  ]
            * NOTE: s_0 is added in MetaLearner>kickstart_training()
            ^ NOTE: b_0 is never filled in, nor is mask_0

        > On 'insert' we do:     
        -                       (step)      (step + 1)
        prev_state     ...  |     ...      |   state    |  ...
        beliefs        ...  |     ...      |   belief   |  ...
        actions        ...  |    action    |    ...     |  ...
        rewards        ...  |    reward    |    ...     |  ...
        masks          ...  |     ...      |    mask    |  ...
        next_state     ...  |    state*    |    ...     |  ...
        latents        ...  |     ...      |  latent^   |  ... 
            * NOTE: metalearner adds this afterwards; not added in insert
            ^ NOTE: latent values are appended differently, but this is how the indices work out
        """

        self.args = args
        # Support tuple state dimensions
        if isinstance(state_dim, int):
            # We set int state_dim 
            self.state_dim = (state_dim,)
        elif isinstance(state_dim, tuple) or isinstance(state_dim, list):
            # We set tuple state_dim
            self.state_dim = tuple(state_dim)
        else: 
            raise ValueError("state_dim must be int, tuple or list, got {}"
                             .format(type(state_dim)))
        self.belief_dim = belief_dim
        self.task_dim = task_dim
        self.model = model

        self.num_steps = num_steps  # how many steps to do per update (= size of online buffer)
        self.num_processes = num_processes  # number of parallel processes
        self.step = 0  # keep track of current environment step

        # normalisation of the rewards
        self.normalise_rewards = normalise_rewards
        self.use_popart = use_popart 

        # computing returns
        self.use_gae = use_gae
        self.gamma = gamma
        self.tau = tau
        self.use_proper_time_limits = use_proper_time_limits

        # inputs to the policy
        # this will include s_0 when state was reset (hence num_steps+1)
        self.prev_state = torch.zeros(num_steps + 1, num_processes, *self.state_dim)
        if self.args.policy.pass_latent:
            # latent variables (of VAE)
            self.latent_dim = latent_dim
            self.latent_samples = []
            self.latent_mean = []
            self.latent_logvar = []
            # hidden states of RNN (necessary if we want to re-compute embeddings)
            self.hidden_size = hidden_size
            self.hidden_states = torch.zeros(num_steps + 1, num_processes, hidden_size)
        else:
            self.latent_mean = None
            self.latent_logvar = None
            self.latent_samples = None

        # next_state will include s_N when state was reset, skipping s_0
        # (only used if we need to re-compute embeddings after backpropagating RL loss through encoder)
        self.next_state = torch.zeros(num_steps, num_processes, *self.state_dim)

        if self.args.policy.pass_belief:
            self.beliefs = torch.zeros(num_steps + 1, num_processes, belief_dim)
        else:
            self.beliefs = None
        
        self.tasks = torch.zeros(num_steps + 1, num_processes, task_dim)

        # rewards and end of trials
        self.rewards_raw = torch.zeros(num_steps, num_processes, 1)
        self.rewards_normalised = torch.zeros(num_steps, num_processes, 1)
        self.done = torch.zeros(num_steps + 1, num_processes, 1)
        self.masks = torch.ones(num_steps + 1, num_processes, 1)
        # masks that indicate whether it's a true terminal state (false) or time limit end state (true)
        self.bad_masks = torch.ones(num_steps + 1, num_processes, 1)

        # Keep track of cliffhanger timesteps (from DCD)
        self.cliffhanger_masks = torch.ones(num_steps + 1, num_processes, 1)
        
        self.level_seeds = torch.zeros(num_steps, num_processes, 1, dtype=torch.int)

        # actions
        if action_space.__class__.__name__ == 'Discrete':
            action_shape = 1
        else:
            action_shape = action_space.shape[0]
        self.actions = torch.zeros(num_steps, num_processes, action_shape)
        if action_space.__class__.__name__ == 'Discrete':
            self.actions = self.actions.long()
        self.action_log_probs = None
        self.action_log_dist = None

        # values and returns
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.denorm_value_preds = None

        # exploration bonus
        self.add_reward_bonus = add_exploration_bonus
        if self.add_reward_bonus:
            self.intrinsic_reward = intrinsic_reward

        self.to_device()

    @property
    def rewards(self):
        if self.normalise_rewards:
            return self.rewards_normalised
        else:
            return self.rewards_raw

    def to_device(self):
        """
        Send all fields to GPU device.
        """
        if self.args.policy.pass_state:
            self.prev_state = self.prev_state.to(DeviceConfig.DEVICE)
        if self.args.policy.pass_latent:
            self.latent_samples = [t.to(DeviceConfig.DEVICE) for t in self.latent_samples]
            self.latent_mean = [t.to(DeviceConfig.DEVICE) for t in self.latent_mean]
            self.latent_logvar = [t.to(DeviceConfig.DEVICE) for t in self.latent_logvar]
            self.hidden_states = self.hidden_states.to(DeviceConfig.DEVICE)
        self.next_state = self.next_state.to(DeviceConfig.DEVICE)
        if self.args.policy.pass_belief:
            self.beliefs = self.beliefs.to(DeviceConfig.DEVICE)
        if self.args.policy.pass_task:
            self.tasks = self.tasks.to(DeviceConfig.DEVICE)
        self.rewards_raw = self.rewards_raw.to(DeviceConfig.DEVICE)
        self.rewards_normalised = self.rewards_normalised.to(DeviceConfig.DEVICE)
        self.done = self.done.to(DeviceConfig.DEVICE)
        self.masks = self.masks.to(DeviceConfig.DEVICE)
        self.bad_masks = self.bad_masks.to(DeviceConfig.DEVICE)
        self.cliffhanger_masks = self.cliffhanger_masks.to(DeviceConfig.DEVICE)
        self.value_preds = self.value_preds.to(DeviceConfig.DEVICE)
        self.returns = self.returns.to(DeviceConfig.DEVICE)
        self.actions = self.actions.to(DeviceConfig.DEVICE)
        if self.level_seeds is not None:
            self.level_seeds = self.level_seeds.to(DeviceConfig.DEVICE)

    def insert_initial_latents(
            self, 
            env_latents: EnvLatents):
        """
        Insert the initial latent states into the buffer before rollout starts.
        """
        self.latent_samples.append(env_latents.z_samples.clone())
        self.latent_mean.append(env_latents.z_means.clone())
        self.latent_logvar.append(env_latents.z_logvars.clone())
        self.hidden_states[0].copy_(env_latents.hs)

    def add_initial_state(self, state):
        if isinstance(state, dict):
            for k in state.keys():
                self.prev_state[k][0].copy_(state[k])
        else:
            self.prev_state[0].copy_(state)

    def add_next_state_before_reset(self, step_idx, state):
        """We assume state is already cloned!"""
        if isinstance(state, dict):
            for k in state.keys():
                self.next_state[k][step_idx] = state[k]
        else:
            self.next_state[step_idx] = state

    def insert(self,
               state: torch.Tensor,
               belief: torch.Tensor,
               task: torch.Tensor,
               actions: torch.Tensor,
               rewards_raw: torch.Tensor,
               rewards_normalised: torch.Tensor,
               value_preds: torch.Tensor,
               masks: torch.Tensor,
               bad_masks: torch.Tensor,
               done: torch.Tensor,
               hidden_states: torch.Tensor = None,
               latent_sample: torch.Tensor = None,
               latent_mean: torch.Tensor = None,
               latent_logvar: torch.Tensor = None,
               level_seeds: torch.Tensor = None,
               cliffhanger_masks: torch.Tensor = None):
        """
        Insert a transition into the buffer.
        
        Args:
            state (torch.Tensor)
            belief (torch.Tensor)
            task (torch.Tensor)
            actions (torch.Tensor)
            rewards_raw (torch.Tensor)
            rewards_normalised (torch.Tensor)
            value_preds (torch.Tensor)
            masks (torch.Tensor): masks that indicate whether it's a true
                terminal state (false) or time limit end state (true)
            bad_masks (torch.Tensor): masks that indicate whether it's a time
                limit end state because of failure
            done (torch.Tensor): 
            hidden_states (torch.Tensor): 
            latent_sample (torch.Tensor)
            latent_mean (torch.Tensor)
            latent_logvar (torch.Tensor)
            level_seeds (torch.Tensor)
        """
        self.prev_state[self.step + 1].copy_(state)
        if self.args.policy.pass_belief:
            self.beliefs[self.step + 1].copy_(belief)
        if self.args.policy.pass_task:
            self.tasks[self.step + 1].copy_(task)
        if self.args.policy.pass_latent:
            self.latent_samples.append(latent_sample.detach().clone())
            self.latent_mean.append(latent_mean.detach().clone())
            self.latent_logvar.append(latent_logvar.detach().clone())
            self.hidden_states[self.step + 1].copy_(hidden_states.detach())
        self.actions[self.step] = actions.detach().clone()
        self.rewards_raw[self.step].copy_(rewards_raw)
        self.rewards_normalised[self.step].copy_(rewards_normalised)
        if isinstance(value_preds, list):
            self.value_preds[self.step].copy_(value_preds[0].detach())
        else:
            self.value_preds[self.step].copy_(value_preds.detach())
        self.masks[self.step + 1].copy_(masks)
        self.bad_masks[self.step + 1].copy_(bad_masks)
        self.done[self.step + 1].copy_(done)

        if cliffhanger_masks is not None:
            self.cliffhanger_masks[self.step + 1].copy_(cliffhanger_masks)
        
        if level_seeds is not None:
            self.level_seeds[self.step].copy_(level_seeds)

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        """
        Copy the last state of the buffer to the first state.
        """
        self.prev_state[0].copy_(self.prev_state[-1])
        if self.args.policy.pass_belief:
            self.beliefs[0].copy_(self.beliefs[-1])
        if self.args.policy.pass_task:
            self.tasks[0].copy_(self.tasks[-1])
        if self.args.policy.pass_latent:
            self.latent_samples = []
            self.latent_mean = []
            self.latent_logvar = []
            self.hidden_states[0].copy_(self.hidden_states[-1])
        self.done[0].copy_(self.done[-1])
        self.masks[0].copy_(self.masks[-1])
        self.bad_masks[0].copy_(self.bad_masks[-1])
        self.cliffhanger_masks[0].copy_(self.cliffhanger_masks[-1])
        self.action_log_probs = None
        self.action_log_dist = None

    def compute_returns(
            self, 
            next_value: torch.Tensor, 
            vae=None):
        """
        Compute the returns for each step in the buffer.

        Args:
            next_value: (torch.Tensor): The value of the next state.
            vae: (VAE): The VAE used for latent space.
        """
        
        if not self.add_reward_bonus:
            rewards = (self.rewards_normalised.clone() if self.normalise_rewards 
                       else self.rewards_raw.clone())
        else:
            # compute reward bonus (we do it in here because then we can batch 
            # the computation)
            with torch.no_grad():
                if self.args.policy.pass_latent:
                    # compute the rew bonus for s^+_{t+1}, i.e., skipping first 
                    # observation and prior
                    belief = torch.cat((torch.stack(self.latent_mean[1:]), 
                                        torch.stack(self.latent_logvar[1:])), 
                                        dim=-1)
                    rew_bonus = self.intrinsic_reward.reward(
                        state=self.next_state,
                        belief=belief,
                        done=self.done,
                        update_normalisation=True,
                        vae=vae,
                        latent_mean=self.latent_mean,
                        latent_logvar=self.latent_logvar,
                        batch_prev_obs=self.prev_state[:-1],
                        batch_next_obs=self.prev_state[1:],
                        batch_actions=self.actions,
                        batch_rewards=self.rewards_raw,
                        batch_tasks=self.tasks[1:])
                else:
                    rew_bonus = self.intrinsic_reward.reward(
                        state=self.next_state,
                        belief=self.beliefs[1:],
                        done=self.done[1:],
                        update_normalisation=True)

            if self.normalise_rewards:
                rewards = self.rewards_normalised.clone() + rew_bonus.clone()

            else:
                rewards = self.rewards_raw.clone() + rew_bonus.clone()
        
        
        self._compute_returns(next_value=next_value, rewards=rewards, 
                              value_preds=self.value_preds,
                              returns=self.returns,
                              gamma=self.gamma, tau=self.tau, use_gae=self.use_gae, 
                              use_proper_time_limits=self.use_proper_time_limits)

    def _compute_returns(self, 
                         next_value, 
                         rewards, 
                         value_preds, 
                         returns, 
                         gamma, 
                         tau, 
                         use_gae, 
                         use_proper_time_limits):

        if self.use_popart:
            self.denorm_value_preds = self.model.popart.denormalize(value_preds) # denormalize all value predictions
            value_preds = self.denorm_value_preds
        
        if use_proper_time_limits:
            if use_gae:
                value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(rewards.size(0))):
                    delta = rewards[step] + gamma * value_preds[step + 1] * self.masks[step + 1] - value_preds[step]
                    gae = delta + gamma * tau * self.masks[step + 1] * gae
                    gae = gae * self.bad_masks[step + 1]
                    returns[step] = gae + value_preds[step]
            else:
                returns[-1] = next_value
                for step in reversed(range(rewards.size(0))):
                    returns[step] = (returns[step + 1] * gamma * self.masks[step + 1] + rewards[step]) * self.bad_masks[
                        step + 1] + (1 - self.bad_masks[step + 1]) * value_preds[step]
        else:
            if use_gae:
                value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(rewards.size(0))):
                    delta = rewards[step] + gamma * value_preds[step + 1] * self.masks[step + 1] - value_preds[step]
                    gae = delta + gamma * tau * self.masks[step + 1] * gae
                    returns[step] = gae + value_preds[step]
            else:
                returns[-1] = next_value
                for step in reversed(range(rewards.size(0))):
                    returns[step] = returns[step + 1] * gamma * self.masks[step + 1] + rewards[step]

    def num_transitions(self):
        """
        Get the total number of transitions in the buffer.

        Returns:
            (int) The total number of transitions in the buffer.
        """
        return len(self.prev_state) * self.num_processes

    def before_update(self, 
                      policy: torch.nn.Module):
        """
        Compute the action log probabilities before the update step.

        Args:
            policy: (torch.nn.Module): The policy network.
        """
        env_latents = EnvLatents(
            z_samples=torch.stack(self.latent_samples[:-1]) 
                if self.latent_samples is not None else None,
            z_means=torch.stack(self.latent_mean[:-1])
                if self.latent_mean is not None else None,
            z_logvars=torch.stack(self.latent_logvar[:-1])
                if self.latent_logvar is not None else None,
            hs=None)

        latent = utl.get_zs_for_policy(
            self.args,                               
            env_latents=env_latents)
        _, action_log_probs, action_log_dist, _ = policy.evaluate_actions(
            self.prev_state[:-1],
            latent,
            self.beliefs[:-1] if self.beliefs is not None else None,
            self.tasks[:-1] if self.tasks is not None else None,
            self.actions)
        self.action_log_probs = action_log_probs.detach()
        self.action_log_dist = action_log_dist.detach()

    def feed_forward_generator(self,
                               advantages: torch.Tensor,
                               num_mini_batch: int = None,
                               mini_batch_size: int = None):
        """
        Generator that yields training data for PPO. The advantages are computed
        over num_steps steps and stored in the rollouts object.

        Args:
            advantages: (torch.Tensor)
            num_mini_batch: (int) number of batches for ppo
            mini_batch_size: (int) number of samples for each ppo batch
        """
        num_steps, num_processes = self.rewards_raw.size()[0:2]
        batch_size = num_processes * num_steps

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(num_processes, num_steps, num_processes * num_steps,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch
        
        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)),
            mini_batch_size,
            drop_last=True)
        for indices in sampler:

            if self.args.policy.pass_state:
                state_batch = self.prev_state[:-1].reshape(-1, *self.prev_state.size()[2:])[indices]
            else:
                state_batch = None
            if self.args.policy.pass_latent:
                latent_sample_batch = torch.cat(self.latent_samples[:-1])[indices]
                latent_mean_batch = torch.cat(self.latent_mean[:-1])[indices]
                latent_logvar_batch = torch.cat(self.latent_logvar[:-1])[indices]
            else:
                latent_sample_batch = latent_mean_batch = latent_logvar_batch = None
            if self.args.policy.pass_belief:
                belief_batch = self.beliefs[:-1].reshape(-1, *self.beliefs.size()[2:])[indices]
            else:
                belief_batch = None
            if self.args.policy.pass_task:
                task_batch = self.tasks[:-1].reshape(-1, *self.tasks.size()[2:])[indices]
            else:
                task_batch = None

            actions_batch = self.actions.reshape(-1, self.actions.size(-1))[indices]

            value_preds_batch = self.value_preds[:-1].reshape(-1, 1)[indices]
            return_batch = self.returns[:-1].reshape(-1, 1)[indices]

            old_action_log_probs_batch = self.action_log_probs.reshape(-1, 1)[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages.reshape(-1, 1)[indices]

            yield state_batch, belief_batch, task_batch, \
                  actions_batch, \
                  latent_sample_batch, latent_mean_batch, latent_logvar_batch, \
                  value_preds_batch, return_batch, old_action_log_probs_batch, adv_targ

    def recurrent_generator(self,
                            advantages: torch.Tensor,
                            num_mini_batch: int = None,
                            mini_batch_size: int = None):
        del advantages, num_mini_batch, mini_batch_size
        raise NotImplementedError


###############################################################################


class DictOnlineStorage(OnlineStorage):
    def __init__(self,
                 args: argparse.Namespace, 
                 model: torch.nn.Module,
                 num_steps: int, 
                 num_processes: int,
                 state_dim: Dict[str, Tuple[int, ...]],  # TODO: Is this right?
                 belief_dim: int, 
                 task_dim: int,
                 action_space: gym.spaces.Dict,
                 hidden_size: int, 
                 latent_dim: int, 
                 normalise_rewards: bool,
                 use_popart: bool = False,
                 add_exploration_bonus: bool = False,
                 intrinsic_reward: Any = None):
        # TODO: Implement OnlineStorage as special case of DictOnlineStorage,
        # so we only have to make modifications to the Dict one. I tried other
        # way around, but too much redundancy in storage (i.e. only obs are
        # different, rest is the same).
        """
        Stores data collected during a rollout (with Dict observation space)

        Args:
            args (argparse.Namespace): parsed command line arguments
            num_steps (int): number of steps to collect per rollout
            num_processes (int): number of parallel processes
            state_dim (Dict[str, Tuple[int, ...]]): dimensionality of the state space
            belief_dim (int): dimensionality of the belief space
            task_dim (int): dimensionality of the task space
            action_space (gym.spaces.Dict): action space of the environment
            hidden_size (int): Size of hidden RNN states.
            latent_dim (int): Dimensionality of latent space (of VAE).
            normalise_rewards (bool): Whether to normalise rewards or not.
            use_popart (bool): Whether to use pop-art for reward normalisation.
            add_exploration_bonus (bool): Whether to add exploration bonus to
                the reward.
            intrinsic_reward (Any): Intrinsic reward object.
        """

        self.args = args
        self.state_dim = state_dim
        self.belief_dim = belief_dim
        self.task_dim = task_dim
        self.model = model

        self.num_steps = num_steps  # how many steps to do per update (= size of online buffer)
        self.num_processes = num_processes  # number of parallel processes
        self.step = 0  # keep track of current environment step

        # normalisation of the rewards
        self.normalise_rewards = normalise_rewards
        self.use_popart = use_popart

        # inputs to the policy
        # this will include s_0 when state was reset (hence num_steps+1)
        self.prev_state = {k: torch.zeros(num_steps + 1, num_processes, *v) for k, v in state_dim.items()}
        if self.args.policy.pass_latent:
            # latent variables (of VAE)
            self.latent_dim = latent_dim
            self.latent_samples = []
            self.latent_mean = []
            self.latent_logvar = []
            # hidden states of RNN (necessary if we want to re-compute embeddings)
            self.hidden_size = hidden_size
            self.hidden_states = torch.zeros(num_steps + 1, num_processes, hidden_size)
        else:
            self.latent_mean = None
            self.latent_logvar = None
            self.latent_samples = None
        # next_state will include s_N when state was reset, skipping s_0
        # (only used if we need to re-compute embeddings after backpropagating RL loss through encoder)
        self.next_state = {k: torch.zeros(num_steps, num_processes, *v) for k, v in state_dim.items()}
        if self.args.policy.pass_belief:
            self.beliefs = torch.zeros(num_steps + 1, num_processes, belief_dim)
        else:
            self.beliefs = None
        if self.args.policy.pass_task:
            self.tasks = torch.zeros(num_steps + 1, num_processes, task_dim)
        else:
            self.tasks = None

        # rewards and end of trials
        self.rewards_raw = torch.zeros(num_steps, num_processes, 1)
        self.rewards_normalised = torch.zeros(num_steps, num_processes, 1)
        self.done = torch.zeros(num_steps + 1, num_processes, 1)
        self.masks = torch.ones(num_steps + 1, num_processes, 1)
        # masks that indicate whether it's a true terminal state (false) or time limit end state (true)
        self.bad_masks = torch.ones(num_steps + 1, num_processes, 1)

        # Keep track of cliffhanger timesteps (from DCD)
        self.cliffhanger_masks = torch.ones(num_steps + 1, num_processes, 1)

        self.level_seeds = torch.zeros(num_steps, num_processes, 1, dtype=torch.int)

        # actions
        if action_space.__class__.__name__ == 'Discrete':
            action_shape = 1
        else:
            action_shape = action_space.shape[0]
        self.actions = torch.zeros(num_steps, num_processes, action_shape)
        if action_space.__class__.__name__ == 'Discrete':
            self.actions = self.actions.long()
        self.action_log_probs = None
        self.action_log_dist = None

        # values and returns
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)

        # exploration bonus
        self.add_reward_bonus = add_exploration_bonus
        if self.add_reward_bonus:
            self.intrinsic_reward = intrinsic_reward

        self.to_device()

    def to_device(self):
        """
        Send all tensors to GPU device.
        """
        if self.args.policy.pass_state:
            self.prev_state = {k: v.to(DeviceConfig.DEVICE) for k, v in self.prev_state.items()}
        if self.args.policy.pass_latent:
            self.latent_samples = [t.to(DeviceConfig.DEVICE) for t in self.latent_samples]
            self.latent_mean = [t.to(DeviceConfig.DEVICE) for t in self.latent_mean]
            self.latent_logvar = [t.to(DeviceConfig.DEVICE) for t in self.latent_logvar]
            self.hidden_states = self.hidden_states.to(DeviceConfig.DEVICE)
        self.next_state = {k: v.to(DeviceConfig.DEVICE) for k, v in self.next_state.items()}
        if self.args.policy.pass_belief:
            self.beliefs = self.beliefs.to(DeviceConfig.DEVICE)
        if self.args.policy.pass_task:
            self.tasks = self.tasks.to(DeviceConfig.DEVICE)
        self.rewards_raw = self.rewards_raw.to(DeviceConfig.DEVICE)
        self.rewards_normalised = self.rewards_normalised.to(DeviceConfig.DEVICE)
        self.done = self.done.to(DeviceConfig.DEVICE)
        self.masks = self.masks.to(DeviceConfig.DEVICE)
        self.bad_masks = self.bad_masks.to(DeviceConfig.DEVICE)
        self.cliffhanger_masks = self.cliffhanger_masks.to(DeviceConfig.DEVICE)
        self.value_preds = self.value_preds.to(DeviceConfig.DEVICE)
        self.returns = self.returns.to(DeviceConfig.DEVICE)
        self.actions = self.actions.to(DeviceConfig.DEVICE)

        if self.level_seeds is not None:
            self.level_seeds = self.level_seeds.to(DeviceConfig.DEVICE)

    def insert(self,
               state: Dict[str, torch.Tensor],
               belief: torch.Tensor,
               task: torch.Tensor,
               actions: torch.Tensor,
               rewards_raw: torch.Tensor,
               rewards_normalised: torch.Tensor,
               value_preds: torch.Tensor,
               masks: torch.Tensor,
               bad_masks: torch.Tensor,
               done: torch.Tensor,
               hidden_states: torch.Tensor = None,
               latent_sample: torch.Tensor = None,
               latent_mean: torch.Tensor = None,
               latent_logvar: torch.Tensor = None,
               level_seeds: torch.Tensor = None,
               cliffhanger_masks: torch.Tensor = None):
        """
        Insert a transition into the buffer.

        Args:
            state: (Dict[str, torch.Tensor]) containing all the environment state representations
            belief: (torch.Tensor)
            task: (torch.Tensor)
            actions: (torch.Tensor)
            rewards_raw: (torch.Tensor)
            rewards_normalised: (torch.Tensor)
            value_preds: (torch.Tensor)
            masks: (torch.Tensor)
            bad_masks: (torch.Tensor)
            done: (torch.Tensor)
            hidden_states: (torch.Tensor)
            latent_sample: (torch.Tensor)
            latent_mean: (torch.Tensor)
            latent_logvar: (torch.Tensor)
            level_seeds: (torch.Tensor)
        """
        for key in state.keys():
            self.prev_state[key][self.step + 1].copy_(state[key])
        if self.args.policy.pass_belief:
            self.beliefs[self.step + 1].copy_(belief)
        if self.args.policy.pass_task:
            self.tasks[self.step + 1].copy_(task)
        if self.args.policy.pass_latent:
            self.latent_samples.append(latent_sample.detach().clone())
            self.latent_mean.append(latent_mean.detach().clone())
            self.latent_logvar.append(latent_logvar.detach().clone())
            self.hidden_states[self.step + 1].copy_(hidden_states.detach())
        self.actions[self.step] = actions.detach().clone()
        self.rewards_raw[self.step].copy_(rewards_raw)
        self.rewards_normalised[self.step].copy_(rewards_normalised)
        if isinstance(value_preds, list):
            self.value_preds[self.step].copy_(value_preds[0].detach())
        else:
            self.value_preds[self.step].copy_(value_preds.detach())
        self.masks[self.step + 1].copy_(masks)
        self.bad_masks[self.step + 1].copy_(bad_masks)
        self.done[self.step + 1].copy_(done)

        if cliffhanger_masks is not None:
            self.cliffhanger_masks[self.step + 1].copy_(cliffhanger_masks)

        if level_seeds is not None:
            self.level_seeds[self.step].copy_(level_seeds)

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        """
        Copy the last state of the buffer to the first state.
        """
        for k in self.prev_state.keys():
            self.prev_state[k][0].copy_(self.prev_state[k][-1])
        if self.args.policy.pass_belief:
            self.beliefs[0].copy_(self.beliefs[-1])
        if self.args.policy.pass_task:
            self.tasks[0].copy_(self.tasks[-1])
        if self.args.policy.pass_latent:
            self.latent_samples = []
            self.latent_mean = []
            self.latent_logvar = []
            self.hidden_states[0].copy_(self.hidden_states[-1])
        self.done[0].copy_(self.done[-1])
        self.masks[0].copy_(self.masks[-1])
        self.bad_masks[0].copy_(self.bad_masks[-1])
        self.cliffhanger_masks[0].copy_(self.cliffhanger_masks[-1])
        self.action_log_probs = None
        self.action_log_dist = None

    def num_transitions(self):
        """
        Get the total number of transitions in the buffer.

        Returns:
            (int) The total number of transitions in the buffer.
        """
        k = self.prev_state.keys()[0]  # All keys should have same num
        return len(self.prev_state[k]) * self.num_processes

    def before_update(self, 
                      policy: torch.nn.Module):
        """
        Compute the action log probabilities before the update step.

        Args:
            policy: (torch.nn.Module): The policy network.
        """
        latent = utl.get_zs_for_policy(
            self.args,                               
            latent_sample=torch.stack(
                self.latent_samples[:-1]) 
                if self.latent_samples is not None else None,
            latent_mean=torch.stack(
                self.latent_mean[:-1]) 
                if self.latent_mean is not None else None,
            latent_logvar=torch.stack(
                self.latent_logvar[:-1]) 
                if self.latent_mean is not None else None)
        
        # Construct
        prev_state = {
            k: self.prev_state[k][:-1] for k in self.prev_state.keys()
        }

        _, action_log_probs, action_log_dist, _ = policy.evaluate_actions(
            prev_state,
            latent,
            self.beliefs[:-1] if self.beliefs is not None else None,
            self.tasks[:-1] if self.tasks is not None else None,
            self.actions)
        self.action_log_probs = action_log_probs.detach()
        self.action_log_dist = action_log_dist.detach()

    def feed_forward_generator(self,
                               advantages: torch.Tensor,
                               num_mini_batch: int = None,
                               mini_batch_size: int = None):
        """
        Generator that yields training data for PPO. The advantages are computed
        over num_steps steps and stored in the rollouts object.

        Args:
            advantages: (torch.Tensor)
            num_mini_batch: (int) number of batches for ppo
            mini_batch_size: (int) number of samples for each ppo batch
        """
        num_steps, num_processes = self.rewards_raw.size()[0:2]
        batch_size = num_processes * num_steps

        # if (mini_batch_size is None):
        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(num_processes, num_steps, num_processes * num_steps,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch
        
        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)),
            mini_batch_size,
            drop_last=True)
        
        for indices in sampler:

            if self.args.policy.pass_state:
                state_batch = dict()
                for k, v in self.prev_state.items():
                    state_batch[k] = v[:-1].reshape(-1, *self.prev_state[k].size()[2:])[indices]
            else:
                state_batch = None
            if self.args.policy.pass_latent:
                latent_sample_batch = torch.cat(self.latent_samples[:-1])[indices]
                latent_mean_batch = torch.cat(self.latent_mean[:-1])[indices]
                latent_logvar_batch = torch.cat(self.latent_logvar[:-1])[indices]
            else:
                latent_sample_batch = latent_mean_batch = latent_logvar_batch = None
            if self.args.policy.pass_belief:
                belief_batch = self.beliefs[:-1].reshape(-1, *self.beliefs.size()[2:])[indices]
            else:
                belief_batch = None
            if self.args.policy.pass_task:
                task_batch = self.tasks[:-1].reshape(-1, *self.tasks.size()[2:])[indices]
            else:
                task_batch = None

            actions_batch = self.actions.reshape(-1, self.actions.size(-1))[indices]

            value_preds_batch = self.value_preds[:-1].reshape(-1, 1)[indices]
            return_batch = self.returns[:-1].reshape(-1, 1)[indices]

            old_action_log_probs_batch = self.action_log_probs.reshape(-1, 1)[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages.reshape(-1, 1)[indices]

            yield state_batch, belief_batch, task_batch, \
                  actions_batch, \
                  latent_sample_batch, latent_mean_batch, latent_logvar_batch, \
                  value_preds_batch, return_batch, old_action_log_probs_batch, adv_targ

    def recurrent_generator(self,
                            advantages: torch.Tensor,
                            num_mini_batch: int = None,
                            mini_batch_size: int = None):
        del advantages, num_mini_batch, mini_batch_size
        raise NotImplementedError