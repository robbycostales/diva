# Modified from https://github.com/facebookresearch/dcd/
from collections import defaultdict, deque

import numpy as np
import torch
from loguru import logger as logur

INT32_MAX = 2147483647


np.seterr(all='raise')

class LevelSampler():
    def __init__(
        self, 
        seeds, 
        obs_space, 
        action_space, 
        num_actors=1, 
        strategy='random', 
        max_score_coef=0.0,
        replay_schedule='fixed', 
        score_transform='power',
        temperature=1.0,
        eps=0.05,
        rho=1.0, 
        replay_prob=0.95, 
        alpha=1.0, 
        staleness_coef=0, 
        staleness_transform='power', 
        staleness_temperature=1.0, 
        sample_full_distribution=False, 
        seed_buffer_size=0, 
        seed_buffer_priority='replay_support',
        use_dense_rewards=False,
        tscl_window_size=0,
        gamma=0.999):
        """
        Inputs: 
            seeds: List, Seeds that can be sampled.
            obs_space: Gym env observation space.
            action_space: Gym env action space.
            num_actors: int, Number of actors/environment processes.
            strategy: Sampling strategy (random, sequential, policy entropy).
            max_score_coef: float, Coefficient for max score (used for score updates).
            replay_schedule: str, Schedule for sampling seen levels.
            score_transform: str, Transform for level scores. Choices include:
                ['constant', 'max', 'eps_greedy', 'rank', 'power', 'softmax']
            temperature: float, Temperature for score transform.
            eps: float, Epsilon for eps-greedy sampling.
            replay_prob: float, Probability of sampling a replay level.
            alpha: float, Smoothing factor for updating scores using an EWA.
            staleness_coef: float, Coefficient for staleness.
            staleness_transform: str, Transform for staleness. Choices include:
                ['max', 'eps_greedy', 'rank', 'power', 'softmax']
            staleness_temperature: float, Temperature for staleness transform.
            sample_full_distribution: bool, Infinite seed setting? 
                # NOTE: Currently the only mode supported
            seed_buffer_size: int, Size of seed buffer.
            seed_buffer_priority: str, Priority for seed buffer. Choices include:
                ['replay_support', 'score']  (latter is lumped into 'else:')
            use_dense_rewards: bool, Whether we're using dense rewards in env.
            tscl_window_size: int, Window size for TSCL.
            gamma: Discount factor.
        """
        self.obs_space = obs_space
        self.action_space = action_space
        self.num_actors = num_actors
        self.strategy = strategy
        self.max_score_coef = max_score_coef
        self.replay_schedule = replay_schedule
        self.score_transform = score_transform
        self.temperature = temperature
        self.eps = eps
        self.rho = rho
        self.replay_prob = replay_prob # replay prob
        self.alpha = alpha
        self.staleness_coef = staleness_coef
        self.staleness_transform = staleness_transform
        self.staleness_temperature = staleness_temperature
        self.gamma = gamma

        self.use_dense_rewards = use_dense_rewards

        # Track seeds and scores as in np arrays backed by shared memory
        self.seed_buffer_size = seed_buffer_size if not seeds else len(seeds)
        N = self.seed_buffer_size
        self._init_seed_index(seeds)

        self.unseen_seed_weights = np.array([1.]*N)
        self.seed_scores = np.array([0.]*N, dtype=np.float64)
        self.partial_seed_scores = np.zeros((num_actors, N), dtype=np.float64)
        self.partial_seed_max_scores = np.ones((num_actors, N), dtype=np.float64)*float('-inf')
        self.partial_seed_steps = np.zeros((num_actors, N), dtype=np.int32)
        self.seed_staleness = np.array([0.]*N, dtype=np.float64)

        self.running_sample_count = 0

        self.next_seed_index = 0 # Only used for sequential strategy

        self.track_solvable = False

        # Handle grounded value losses
        self.grounded_values = None
        if self.strategy.startswith('grounded'):
            self.grounded_values = np.array([np.NINF]*N, dtype=np.float64)

        # Only used for infinite seed setting
        self.sample_full_distribution = sample_full_distribution
        if self.sample_full_distribution:
            self.seed2actor = defaultdict(set)
            self.working_seed_buffer_size = 0
            self.seed_buffer_priority = seed_buffer_priority
            self.staging_seed_set = set()
            self.working_seed_set = set()

            self.seed2timestamp_buffer = {} # Buffer seeds are unique across actors
            self.partial_seed_scores_buffer = [{} for _ in range(num_actors)]
            self.partial_seed_max_scores_buffer = [{} for _ in range(num_actors)]
            self.partial_seed_steps_buffer = [{} for _ in range(num_actors)]
        else: 
            raise NotImplementedError          

        # TSCL specific data structures
        if self.strategy.startswith('tscl'):
            self.tscl_window_size = tscl_window_size
            self.tscl_return_window = [deque(maxlen=self.tscl_window_size) for _ in range(N)]
            self.tscl_episode_window = [deque(maxlen=self.tscl_window_size) for _ in range(N)]
            self.unseen_seed_weights = np.zeros(N) # Force uniform distribution over seeds

    def seed_range(self):
        if not self.sample_full_distribution:
            return (int(min(self.seeds)), int(max(self.seeds)))
        else:
            return (0, INT32_MAX)

    def _init_seed_index(self, seeds):
        if seeds:
            self.seeds = np.array(seeds, dtype=np.int64)
            self.seed2index = {seed: i for i, seed in enumerate(seeds)}
        else:
            self.seeds = np.zeros(self.seed_buffer_size, dtype=np.int64) - 1
            self.seed2index = {}

    def _init_solvable_tracking(self):
        """
        Prepare data structures for tracking seed solvability.
        Currently only used with externally observed seeds.
        """
        self.track_solvable = True
        self.staging_seed2solvable = {}
        self.seed_solvable = np.ones(self.seed_buffer_size, dtype=np.bool)

    @property
    def _proportion_filled(self):
        if self.sample_full_distribution:
            return self.working_seed_buffer_size/self.seed_buffer_size
        else:
            num_unseen = (self.unseen_seed_weights > 0).sum()
            proportion_seen = (len(self.seeds) - num_unseen)/len(self.seeds)
            return proportion_seen

    def update_with_rollouts(self, rollouts):
        if self.strategy in ['random', 'off']:
            return

        # Update with a RolloutStorage object
        if self.strategy == 'uniform':
            score_function = self._uniform
        elif self.strategy == 'policy_entropy':
            score_function = self._average_entropy
        elif self.strategy == 'least_confidence':
            score_function = self._average_least_confidence
        elif self.strategy == 'min_margin':
            score_function = self._average_min_margin
        elif self.strategy == 'gae':
            score_function = self._average_gae
        elif self.strategy == 'value_l1':
            score_function = self._average_value_l1
        elif self.strategy == 'signed_value_loss':
            score_function = self._average_signed_value_loss
        elif self.strategy == 'positive_value_loss':
            score_function = self._average_positive_value_loss
        elif self.strategy == 'grounded_signed_value_loss':
            score_function = self._average_grounded_signed_value_loss
        elif self.strategy == 'grounded_positive_value_loss':
            score_function = self._average_grounded_positive_value_loss
        elif self.strategy == 'one_step_td_error':
            score_function = self._one_step_td_error
        elif self.strategy == 'alt_advantage_abs':
            score_function = self._average_alt_advantage_abs
        elif self.strategy == 'tscl_window':
            score_function = self._tscl_window
        else:
            raise ValueError(f'Unsupported strategy, {self.strategy}')

        self._update_with_rollouts(rollouts, score_function)

    def update_seed_score(self, actor_index, seed, score, max_score, num_steps):
        if self.sample_full_distribution and seed in self.staging_seed_set:
            score, seed_idx = self._partial_update_seed_score_buffer(actor_index, seed, score, num_steps, done=True)
        else:
            score, seed_idx = self._partial_update_seed_score(actor_index, seed, score, max_score, num_steps, done=True)

        return score, seed_idx

    def _partial_update_seed_score(self, actor_index, seed, score, max_score, num_steps, done=False):
        seed_idx = self.seed2index.get(seed, -1)
        if seed_idx < 0:
            return 0, None
        partial_score = self.partial_seed_scores[actor_index][seed_idx]
        partial_max_score = self.partial_seed_max_scores[actor_index][seed_idx]
        partial_num_steps = self.partial_seed_steps[actor_index][seed_idx]

        running_num_steps = partial_num_steps + num_steps

        merged_score = partial_score + (score - partial_score)*num_steps/float(running_num_steps)
        merged_max_score = max(partial_max_score, max_score)

        if done:
            self.partial_seed_scores[actor_index][seed_idx] = 0. # zero partial score, partial num_steps
            self.partial_seed_max_scores[actor_index][seed_idx] = float('-inf')
            self.partial_seed_steps[actor_index][seed_idx] = 0
            self.unseen_seed_weights[seed_idx] = 0. # No longer unseen
            old_score = self.seed_scores[seed_idx]
            total_score = self.max_score_coef*merged_max_score + (1 - self.max_score_coef)*merged_score
            self.seed_scores[seed_idx] = (1 - self.alpha)*old_score + self.alpha*total_score
        else:
            self.partial_seed_scores[actor_index][seed_idx] = merged_score
            self.partial_seed_max_scores[actor_index][seed_idx] = merged_max_score
            self.partial_seed_steps[actor_index][seed_idx] = running_num_steps

        return merged_score, seed_idx

    @property
    def _next_buffer_index(self):
        if self._proportion_filled < 1.0:
            return self.working_seed_buffer_size
        else:
            if self.seed_buffer_priority == 'replay_support':
                return self.sample_weights().argmin()
            else:
                return self.seed_scores.argmin()

    def _partial_update_seed_score_buffer(self, actor_index, seed, score, num_steps, done=False):
        seed_idx = -1
        self.seed2actor[seed].add(actor_index)
        partial_score = self.partial_seed_scores_buffer[actor_index].get(seed, 0)
        partial_num_steps = self.partial_seed_steps_buffer[actor_index].get(seed, 0)

        running_num_steps = partial_num_steps + num_steps
        merged_score = partial_score + (score - partial_score)*num_steps/float(running_num_steps)

        if done:
            # Move seed into working seed data structures
            seed_idx = self._next_buffer_index
            if self.seed_scores[seed_idx] <= merged_score or self.unseen_seed_weights[seed_idx] > 0:
                self.unseen_seed_weights[seed_idx] = 0. # Unmask this index
                self.working_seed_set.discard(self.seeds[seed_idx])
                self.working_seed_set.add(seed)
                self.seeds[seed_idx] = seed
                self.seed2index[seed] = seed_idx 
                self.seed_scores[seed_idx] = merged_score
                self.partial_seed_scores[:,seed_idx] = 0.
                self.partial_seed_steps[:,seed_idx] = 0 
                self.seed_staleness[seed_idx] = self.running_sample_count - self.seed2timestamp_buffer[seed]
                self.working_seed_buffer_size = min(self.working_seed_buffer_size + 1, self.seed_buffer_size)

                if self.track_solvable:
                    self.seed_solvable[seed_idx] = self.staging_seed2solvable.get(seed, True)
            else:
                seed_idx = None

            # Zero partial score, partial num_steps, remove seed from staging data structures
            for a in self.seed2actor[seed]:
                self.partial_seed_scores_buffer[a].pop(seed, None) 
                self.partial_seed_steps_buffer[a].pop(seed, None)
            del self.seed2timestamp_buffer[seed]
            del self.seed2actor[seed]
            self.staging_seed_set.remove(seed)

            if self.track_solvable:
                del self.staging_seed2solvable[seed]
        else:
            self.partial_seed_scores_buffer[actor_index][seed] = merged_score
            self.partial_seed_steps_buffer[actor_index][seed] = running_num_steps

        return merged_score, seed_idx

    def _uniform(self, **kwargs):
        return 1.0,1.0

    def _average_entropy(self, **kwargs):
        episode_logits = kwargs['episode_logits']
        num_actions = self.action_space.n
        max_entropy = -(1./num_actions)*np.log(1./num_actions)*num_actions

        scores = -torch.exp(episode_logits)*episode_logits.sum(-1)/max_entropy
        mean_score = scores.mean().item()
        max_score = scores.max().item()

        return mean_score, max_score

    def _average_least_confidence(self, **kwargs):
        episode_logits = kwargs['episode_logits']
        scores = 1 - torch.exp(episode_logits.max(-1, keepdim=True)[0])

        mean_score = scores.mean().item()
        max_score = scores.max().item()
        
        return mean_score, max_score

    def _average_min_margin(self, **kwargs):
        episode_logits = kwargs['episode_logits']
        top2_confidence = torch.exp(episode_logits.topk(2, dim=-1)[0])
        scores = top2_confidence[:,0] - top2_confidence[:,1]
        mean_score = 1 - scores.mean().item()
        max_score = 1 - scores.min().item()

        return mean_score, max_score

    def _average_gae(self, **kwargs):
        returns = kwargs['returns']
        value_preds = kwargs['value_preds']

        advantages = returns - value_preds

        mean_score = advantages.mean().item()
        max_score = advantages.max().item()

        return mean_score, max_score

    def _average_value_l1(self, **kwargs):
        returns = kwargs['returns']
        value_preds = kwargs['value_preds']

        abs_advantages = (returns - value_preds).abs()

        mean_score = abs_advantages.mean().item()
        max_score = abs_advantages.max().item()

        return mean_score, max_score

    def _average_signed_value_loss(self, **kwargs):
        returns = kwargs['returns']
        value_preds = kwargs['value_preds']

        advantages = returns - value_preds

        mean_score = advantages.mean().item()
        max_score = advantages.max().item()

        return mean_score, max_score

    def _average_positive_value_loss(self, **kwargs):
        returns = kwargs['returns']
        value_preds = kwargs['value_preds']

        clipped_advantages = (returns - value_preds).clamp(0)

        mean_score = clipped_advantages.mean().item()
        max_score = clipped_advantages.max().item()

        return mean_score, max_score

    def _average_grounded_signed_value_loss(self, **kwargs):
        """
        Currently assumes sparse reward s.t. reward is 0 everywhere except final step
        """
        seed = kwargs['seed']
        seed_idx = self.seed2index.get(seed, None)
        actor_idx= kwargs['actor_index']
        done = kwargs['done']
        value_preds = kwargs['value_preds']
        episode_logits = kwargs['episode_logits']

        partial_steps = 0
        if self.sample_full_distribution and seed in self.partial_seed_steps_buffer[actor_idx]:
            partial_steps = self.partial_seed_steps_buffer[actor_idx][seed]
        elif seed_idx is not None:
            partial_steps = self.partial_seed_steps[actor_idx][seed_idx]
        else:
            partial_steps = 0

        new_steps = len(episode_logits)
        total_steps = partial_steps + new_steps

        grounded_value = kwargs.get('grounded_value', None)

        if done and grounded_value is not None:
            if self.use_dense_rewards:
                advantages = grounded_value - value_preds[0]
            else:
                advantages = grounded_value - value_preds

            mean_score = (total_steps/new_steps)*advantages.mean().item()
            max_score = advantages.max().item()
        else:
            mean_score, max_score = 0,0

        return mean_score, max_score

    def _average_grounded_positive_value_loss(self, **kwargs):
        """
        Currently assumes sparse reward s.t. reward is 0 everywhere except final step
        """
        seed = kwargs['seed']
        seed_idx = self.seed2index.get(seed, None)
        actor_idx= kwargs['actor_index']
        done = kwargs['done']
        value_preds = kwargs['value_preds']
        episode_logits = kwargs['episode_logits']

        partial_steps = 0
        if self.sample_full_distribution and seed in self.partial_seed_steps_buffer[actor_idx]:
            partial_steps = self.partial_seed_steps_buffer[actor_idx][seed]
        elif seed_idx is not None:
            partial_steps = self.partial_seed_steps[actor_idx][seed_idx]
        else:
            partial_steps = 0

        new_steps = len(episode_logits)
        total_steps = partial_steps + new_steps

        grounded_value = kwargs.get('grounded_value', None)

        if done and grounded_value is not None:
            if self.use_dense_rewards:
                advantages = grounded_value - value_preds[0]
            else:
                advantages = grounded_value - value_preds
            advantages = advantages.clamp(0)

            mean_score = (total_steps/new_steps)*advantages.mean().item()
            max_score = advantages.max().item()
        else:
            mean_score, max_score = 0,0

        return mean_score, max_score

    def _one_step_td_error(self, **kwargs):
        rewards = kwargs['rewards']
        value_preds = kwargs['value_preds']

        max_t = len(rewards)
        if max_t > 1:
            td_errors = (rewards[:-1] + self.gamma*value_preds[1:max_t] - value_preds[:max_t-1]).abs()
        else:
            td_errors = rewards[0] - value_preds[0]

        mean_score = td_errors.mean().item()
        max_score = td_errors.max().item()

        return mean_score, max_score

    def _average_alt_advantage_abs(self, **kwargs):
        returns = kwargs['alt_returns']
        value_preds = kwargs['value_preds']

        abs_advantages = (returns - value_preds).abs()

        mean_score = abs_advantages.mean().item()
        max_score = abs_advantages.max().item()

        return mean_score, max_score

    def _tscl_window(self, **kwargs):
        rewards = kwargs['rewards']
        seed = kwargs['seed']

        seed_idx = self.seed2index.get(seed, -1)
        assert(seed_idx >= 0)

        # add rewards to the seed window
        episode_total_reward = rewards.sum().item()
        self.tscl_return_window[seed_idx].append(episode_total_reward)
        self.tscl_episode_window[seed_idx].append(self.running_sample_count)

        # compute linear regression coeficient in the window
        x = self.tscl_episode_window[seed]
        y = self.tscl_return_window[seed]
        A = np.vstack([x, np.ones(len(x))]).T
        c,_ = np.linalg.lstsq(A, y, rcond=None)[0]

        c = abs(c)
        return c, c

    @property
    def requires_value_buffers(self):
        return self.strategy in [
            'gae', 'value_l1', 
            'signed_value_loss', 'positive_value_loss',
            'grounded_signed_value_loss', 'grounded_positive_value_loss',
            'one_step_td_error', 'alt_advantage_abs', 
            'tscl_window']

    @property
    def _has_working_seed_buffer(self):
        return not self.sample_full_distribution or (self.sample_full_distribution and self.seed_buffer_size > 0)

    def _update_with_rollouts(self, rollouts, score_function):
        if not self._has_working_seed_buffer:
            return

        level_seeds = rollouts.level_seeds
        policy_logits = rollouts.action_log_dist
        total_steps, num_actors = policy_logits.shape[:2]
        done = ~(rollouts.masks > 0)
        # early_done = ~(rollouts.bad_masks > 0)
        cliffhanger = ~(rollouts.cliffhanger_masks > 0)

        if not rollouts.use_popart:
            rollouts.denorm_value_preds = rollouts.value_preds

        for actor_index in range(num_actors):
            start_t = 0
            done_steps = done[:,actor_index].nonzero()[:,0]

            for t in done_steps:
                if not start_t < total_steps: 
                    break

                if t == 0: # if t is 0, then this done step caused a full update of previous seed last cycle
                    continue 

                seed_t = level_seeds[start_t,actor_index].item()

                score_function_kwargs = {}
                score_function_kwargs['actor_index'] = actor_index
                score_function_kwargs['done'] = True
                episode_logits = policy_logits[start_t:t,actor_index]
                score_function_kwargs['episode_logits'] = torch.log_softmax(episode_logits, -1)
                score_function_kwargs['seed'] = seed_t

                if self.requires_value_buffers:
                    score_function_kwargs['returns'] = rollouts.returns[start_t:t,actor_index]
                    if self.strategy == 'alt_advantage_abs':
                        score_function_kwargs['alt_returns'] = rollouts.alt_returns[start_t:t,actor_index]
                    score_function_kwargs['rewards'] = rollouts.rewards[start_t:t,actor_index]

                    if rollouts.use_popart:
                        score_function_kwargs['value_preds'] = rollouts.denorm_value_preds[start_t:t,actor_index]
                    else:
                        score_function_kwargs['value_preds'] = rollouts.value_preds[start_t:t,actor_index]

                # Only perform score updates on non-cliffhanger episodes ending in 'done'
                if not cliffhanger[t,actor_index]:
                    # Update grounded values (highest achieved return per seed)
                    grounded_value = None
                    if self.grounded_values is not None:
                        seed_idx = self.seed2index.get(seed_t, None)
                        score_function_kwargs['seed_idx'] = seed_idx
                        grounded_value_ = rollouts.rewards[start_t:t].sum(0)[actor_index]
                        if seed_idx is not None:
                            grounded_value = max(self.grounded_values[seed_idx], grounded_value_)
                        else:
                            grounded_value = grounded_value_ # Should this be discounted?
                        score_function_kwargs['grounded_value'] = grounded_value

                    score, max_score = score_function(**score_function_kwargs)
                    num_steps = len(episode_logits)
                    _, seed_idx = self.update_seed_score(actor_index, seed_t, score, max_score, num_steps)

                    # Track grounded value for future reference
                    if seed_idx is not None and self.grounded_values is not None and grounded_value is not None:
                        self.grounded_values[seed_idx] = grounded_value

                start_t = t.item()

            if start_t < total_steps:
                seed_t = level_seeds[start_t,actor_index].item()

                score_function_kwargs = {}
                score_function_kwargs['actor_index'] = actor_index
                score_function_kwargs['done'] = False
                episode_logits = policy_logits[start_t:,actor_index]
                score_function_kwargs['episode_logits'] = torch.log_softmax(episode_logits, -1)
                score_function_kwargs['seed'] = seed_t

                if self.requires_value_buffers:
                    score_function_kwargs['returns'] = rollouts.returns[start_t:,actor_index]
                    if self.strategy == 'alt_advantage_abs':
                        score_function_kwargs['alt_returns'] = rollouts.alt_returns[start_t:,actor_index]
                    score_function_kwargs['rewards'] = rollouts.rewards[start_t:,actor_index]

                    if rollouts.use_popart:
                        score_function_kwargs['value_preds'] = rollouts.denorm_value_preds[start_t:,actor_index]
                    else:
                        score_function_kwargs['value_preds'] = rollouts.value_preds[start_t:,actor_index]

                score, max_score = score_function(**score_function_kwargs)
                num_steps = len(episode_logits)

                if self.sample_full_distribution and seed_t in self.staging_seed_set:
                    self._partial_update_seed_score_buffer(actor_index, seed_t, score, num_steps)
                else:
                    self._partial_update_seed_score(actor_index, seed_t, score, max_score, num_steps)

    def after_update(self):
        if not self._has_working_seed_buffer:
            return
        
        # Reset partial updates, since weights have changed, and thus logits are now stale
        for actor_index in range(self.partial_seed_scores.shape[0]):
            for seed_idx in range(self.partial_seed_scores.shape[1]):
                if self.partial_seed_scores[actor_index][seed_idx] != 0:
                    self.update_seed_score(actor_index, self.seeds[seed_idx], 0, float('-inf'), 0)

        self.partial_seed_scores.fill(0)
        self.partial_seed_steps.fill(0)

        # Likewise, reset partial update buffers
        if self.sample_full_distribution:
            for actor_index in range(self.num_actors):
                actor_staging_seeds = list(self.partial_seed_scores_buffer[actor_index].keys())
                for seed in actor_staging_seeds:
                    if self.partial_seed_scores_buffer[actor_index][seed] > 0:
                        self.update_seed_score(actor_index, seed, 0, float('-inf'), 0)

    def _update_staleness(self, selected_idx):
        if self.staleness_coef > 0:
            self.seed_staleness = self.seed_staleness + 1
            self.seed_staleness[selected_idx] = 0

    def sample_replay_decision(self):
        logur.debug('LevelSampler.sample_replay_decision > self._proportion_filled = {}'.format(self._proportion_filled))
        if self.sample_full_distribution: 
            proportion_filled = self._proportion_filled
            if self.seed_buffer_size > 0:
                if self.replay_schedule == 'fixed':
                    if proportion_filled >= self.rho and np.random.rand() < self.replay_prob:
                        return True
                    else:
                        return False
                else:
                    if proportion_filled >= self.rho and np.random.rand() < min(proportion_filled, self.replay_prob):
                        return True
                    else:
                        return False
            else:
                # If seed buffer has length 0, then just sample new random seed each time
                return False

        elif self.replay_schedule == 'fixed':
            proportion_seen = self._proportion_filled
            if proportion_seen >= self.rho: 
                # Sample replay level with fixed replay_prob OR if all levels seen
                if np.random.rand() < self.replay_prob or not proportion_seen < 1.0:
                    return True

            # Otherwise, sample a new level
            return False

        else: # Default to proportionate schedule
            proportion_seen = self._proportion_filled
            if proportion_seen >= self.rho and np.random.rand() < proportion_seen:
                return True
            else:
                return False

    @property
    def is_warm(self):
        return self._proportion_filled >= self.rho

    def stage_seeds(self, seeds):
        for seed in seeds:
            self._stage_seed(seed)

    def _stage_seed(self, seed):
        """ Function that allows staging of seed without calling 'sample'. """
        assert self.sample_full_distribution, "Cannot manually stage seed without full distribution sampling"

        if seed in self.staging_seed_set or seed in self.working_seed_set:
            return  # Seed already staged

        self.staging_seed_set.add(seed)
        self.seed2timestamp_buffer[seed] = self.running_sample_count

    def observe_external_unseen_sample(self, seeds, solvable=None):
        for i, seed in enumerate(seeds):
            self.running_sample_count += 1
            if not (seed in self.staging_seed_set or seed in self.working_seed_set):
                self.seed2timestamp_buffer[seed] = self.running_sample_count
                self.staging_seed_set.add(seed)

                if solvable is not None:
                    if not self.track_solvable: # lazy init of solvable tracking
                        self._init_solvable_tracking()
                    self.staging_seed2solvable[seed] = solvable[i]
            else:
                seed_idx = self.seed2index.get(seed, None)
                if seed_idx is not None:
                    self._update_staleness(seed_idx)

    def sample_replay_level(self, update_staleness=True):
        return self._sample_replay_level(update_staleness=update_staleness)

    def _sample_replay_level(self, update_staleness=True):
        sample_weights = self.sample_weights()

        if np.isclose(np.sum(sample_weights), 0):
            sample_weights = np.ones_like(self.seeds, dtype=np.float64)/len(self.seeds)
            sample_weights = sample_weights*(1-self.unseen_seed_weights)
            sample_weights /= np.sum(sample_weights)
        elif np.sum(sample_weights, 0) != 1.0:
            sample_weights = sample_weights/np.sum(sample_weights,0)

        seed_idx = np.random.choice(range(len(self.seeds)), 1, p=sample_weights)[0]
        seed = self.seeds[seed_idx]

        if update_staleness:
            self._update_staleness(seed_idx)

        return int(seed)

    def _sample_unseen_level(self):
        if self.sample_full_distribution:
            seed = int(np.random.randint(1,INT32_MAX))
            # Ensure unique new seed outside of working and staging set
            while seed in self.staging_seed_set or seed in self.working_seed_set:
                seed = int(np.random.randint(1,INT32_MAX))
            self.seed2timestamp_buffer[seed] = self.running_sample_count
            self.staging_seed_set.add(seed)
        else:
            sample_weights = self.unseen_seed_weights/self.unseen_seed_weights.sum()
            seed_idx = np.random.choice(range(len(self.seeds)), 1, p=sample_weights)[0]
            seed = self.seeds[seed_idx]

            self._update_staleness(seed_idx)

        return int(seed)

    def sample(self, strategy=None):
        if strategy == 'full_distribution':
            raise ValueError('One-off sampling via full_distribution strategy is not supported.')

        self.running_sample_count += 1

        if not strategy:
            strategy = self.strategy

        if not self.sample_full_distribution:
            
            if strategy == 'random':
                logur.debug('WARNING: Sampling RANDOM level for PLR.')
                seed_idx = np.random.choice(range((len(self.seeds))))
                seed = self.seeds[seed_idx]
                return int(seed)
            else:
                logur.debug('Sampling level for PLR based off objectives.')

            if strategy == 'sequential':
                seed_idx = self.next_seed_index
                self.next_seed_index = (self.next_seed_index + 1) % len(self.seeds)
                seed = self.seeds[seed_idx]
                return int(seed)

        replay_decision = self.sample_replay_decision()
        logur.debug('Replay decision: {}'.format(replay_decision))
        if replay_decision:
            logur.debug('Sampling REPLAY level for PLR.')
            return self._sample_replay_level()
        else:
            logur.debug('Sampling UNSEEN level for PLR.')
            return self._sample_unseen_level()

    def sample_weights(self):
        weights = self._score_transform(self.score_transform, self.temperature, self.seed_scores)
        weights = weights * (1-self.unseen_seed_weights) # zero out unseen levels

        # NOTE: We had to add this logic because previously, before QD
        # integration, we would not ask for sample_weights in the 'random' case,
        # but now we do, since we're using these weights for QD as well.
        if self.strategy == 'random':
            # Random strategy should always sample uniformly
            weights = np.zeros_like(weights, dtype=np.float64)
            weights[:len(self.seed2index)] = 1.0/len(self.seed2index)
            return weights

        z = np.sum(weights)
        if z > 0:
            weights /= z
        else:
            weights = np.ones_like(weights, dtype=np.float64)/len(weights)
            weights = weights * (1-self.unseen_seed_weights)
            weights /= np.sum(weights)

        staleness_weights = 0
        if self.staleness_coef > 0:
            staleness_weights = self._score_transform(self.staleness_transform, self.staleness_temperature, self.seed_staleness)
            staleness_weights = staleness_weights * (1-self.unseen_seed_weights)
            z = np.sum(staleness_weights)
            if z > 0: 
                staleness_weights /= z
            else:
                staleness_weights = 1./len(staleness_weights)*(1-self.unseen_seed_weights)

            weights = (1 - self.staleness_coef)*weights + self.staleness_coef*staleness_weights

        return weights

    def _score_transform(self, transform, temperature, scores):
        if transform == 'constant':
            weights = np.ones_like(scores)
        if transform == 'max':
            weights = np.zeros_like(scores)
            scores = scores[:]
            scores[self.unseen_seed_weights > 0] = -float('inf') # only argmax over seen levels
            argmax = np.random.choice(np.flatnonzero(np.isclose(scores, scores.max())))
            weights[argmax] = 1.
        elif transform == 'eps_greedy':
            weights = np.zeros_like(scores)
            weights[scores.argmax()] = 1. - self.eps
            weights += self.eps/len(self.seeds)
        elif transform == 'rank':
            temp = np.flip(scores.argsort())
            ranks = np.empty_like(temp)
            ranks[temp] = np.arange(len(temp)) + 1
            weights = 1/ranks ** (1./temperature)
        elif transform == 'power':
            eps = 0 if self.staleness_coef > 0 else 1e-3
            weights = (np.array(scores).clip(0) + eps) ** (1./temperature)
        elif transform == 'softmax':
            weights = np.exp(np.array(scores)/temperature)
        elif transform == 'match':
            weights = np.array([(1-score)*score for score in scores])
            weights = weights ** (1./temperature)
        elif transform == 'match_rank':
            weights = np.array([(1-score)*score for score in scores])
            temp = np.flip(weights.argsort())
            ranks = np.empty_like(temp)
            ranks[temp] = np.arange(len(temp)) + 1
            weights = 1/ranks ** (1./temperature)

        return weights

    @property
    def solvable_mass(self):
        if self.track_solvable:
            sample_weights = self.sample_weights()
            return np.sum(sample_weights[self.seed_solvable])
        else:
            return 1.

    @property
    def max_score(self):
        return max(self.seed_scores)
    
    def _add_seed(self, seed, score):
        """ Add seed to sampler.

        Args:
            seed (int): seed to add
            score (float): score for seed

        Returns:
            success (bool): whether seed was successfully added
        """
        # If seed already exists, just update its score
        if seed in self.seed2index:
            index = self.seed2index[seed]
            self.seed_scores[index] = score
        else:
            # If the seed does not exist:
            # Find the indices where self.seeds has -1, indicating empty slots
            empty_slots = np.where(self.seeds == -1)[0]
            if len(empty_slots) == 0:
                logur.debug('No empty slots in seed buffer')
                return False
            
            index = empty_slots[0]  # Take the first empty slot
            self.seeds[index] = seed
            self.seed_scores[index] = score
            self.unseen_seed_weights[index] = 0.0  # Since it's seen now
            
            # Update the seed2index dict
            self.seed2index[seed] = index
            
            # Add the seed to working seed set
            self.working_seed_set.add(seed)
            
            # Update the working seed buffer size
            self.working_seed_buffer_size += 1

        return True

    def add_seeds(self, seeds, scores):
        """ Add seeds to the sampler.
        
        Args:
            seeds (list): list of seeds to add
            scores (list): list of scores for each seed
        Returns:
            failed_seeds (list): list of seeds that failed to be added
        """
        logur.debug('LS - Adding seeds: {}'.format(seeds))
        failed_seeds = []
        if len(seeds) != len(scores):
            raise ValueError('Seeds and scores must be same length')
        for seed, score in zip(seeds, scores):
            success = self._add_seed(seed, score)
            if not success:
                failed_seeds.append(seed)
        return failed_seeds
    
    def purge_seeds(self, seeds):
        """ Remove seeds from the sampler. 
        
        Args:
            seeds (list): list of seeds to remove in form [(index, seed), ...]
        """
        logur.debug(f'LS - Purging seeds: {seeds}')
        indices = [i for (i, _) in seeds]
        seeds = [seed for (_, seed) in seeds]
        indices = np.array(indices)
        seed_list = seeds
        seeds = np.array(seeds)
        # NOTE: `self.partial_*` values do not need to be updated, since we
        #   assume this is called after a call to `after_update`, which already
        #   clears these values.
        self.seeds = np.pad(np.delete(self.seeds, indices), 
            (0, len(seeds)), mode='constant', constant_values=-1)
        self.seed_scores = np.pad(np.delete(self.seed_scores, indices), 
            (0, len(seeds)), mode='constant', constant_values=0.0)
        self.seed_staleness = np.pad(np.delete(self.seed_staleness, indices), 
            (0, len(seeds)), mode='constant', constant_values=0.0)
        self.seed2index = dict()
        for i, seed in enumerate(self.seeds):
            if seed == -1:
                break
            self.seed2index[seed] = i
        self.unseen_seed_weights = np.pad(np.delete(self.unseen_seed_weights, indices), 
            (0, len(seeds)), mode='constant', constant_values=1.0)
        self.working_seed_buffer_size -= len(seed_list)
        # Remove seeds from working seed set
        for seed in seed_list:
            self.working_seed_set.discard(seed)
        assert self.next_seed_index == 0  # Bc. only used for sequential strategy
        return


    def merge_sampler(self, other_sampler):
        """ Merge another sampler into this one.
        
        Args:
            other_sampler (Sampler): sampler to merge into this one
        """
        logur.debug('Merging samplers...')
        
        # Get all seeds that are in the other sampler but not this one
        other_seeds = set.difference(other_sampler.working_seed_set, self.working_seed_set)

        # Try adding these seeds to this sampler
        other_seeds = list(other_seeds)
        other_seed_scores = [other_sampler.seed_scores[
                                other_sampler.seed2index[seed]] 
                             for seed in other_seeds]
        failed_seeds = self.add_seeds(other_seeds, other_seed_scores)

        logur.debug(f'Added {len(other_seeds) - len(failed_seeds)}/{len(other_seeds)}'
               ' from other sampler.')








    