import time

import numpy as np
import torch

from diva.environments.vec_env import VecEnvWrapper
from diva.utils.torch import tensor


def _get_env_class(gt_type):
    """Get the appropriate environment class based on genotype type."""
    # TODO: Simplify this; would get messy for more than three environments
    from diva.environments.alchemy.alchemy_qd import ExtendedSymbolicAlchemy
    from diva.environments.box2d.car_racing_bezier import CarRacingBezier
    from diva.environments.toygrid.toygrid import ToyGrid

    if 'CP' in gt_type:
        return CarRacingBezier
    elif gt_type == 'natural' or gt_type[0] == 'a':
        return ToyGrid
    else:
        return ExtendedSymbolicAlchemy

def solution_from_seed(seed, level_store, lower_bounds, upper_bounds, solution_dim, gt_type):
    """ Generate or retrieve (if already generated) level from seed. """
    # First, check if seed in level store
    if seed in level_store.seed2level:
        return level_store.seed2level[seed]
    
    # Otherwise, generate level    
    env_class = _get_env_class(gt_type)
    return env_class.genotype_from_seed_static(
        seed, 
        gt_type=gt_type, 
        genotype_lower_bounds=lower_bounds,
        genotype_upper_bounds=upper_bounds,
        genotype_size=solution_dim)

def solutions_from_seeds(seeds, level_store, lower_bounds, upper_bounds, solution_dim, gt_type):
    """ Convert list of seeds to list of generated levels. """
    solutions = []
    for seed in seeds:
        solution = solution_from_seed(seed, level_store, lower_bounds, upper_bounds, solution_dim, gt_type)
        # Add solution to list
        solutions.append(solution)
    return solutions


class VecPyTorch(VecEnvWrapper):
    def __init__(
            self, 
            venv, 
            device, 
            level_sampler=None, 
            level_store=None, 
            start_seeds=None):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        self.device = device
        self.level_sampler = level_sampler
        self.level_store = level_store
        self.start_seeds = np.array(start_seeds)
        self.cur_seeds = self.start_seeds

        self.attributes_get_first = set(
            ['_max_episode_steps',      
             'task_dim',
             'belief_dim', 
             'num_states', 
             'bit_map_size', 
             'genotype_size', 
             'qd_bounds',
             'genotype_bounds', 
             'genotype_lower_bounds', 
             'genotype_upper_bounds',
             'genotype_size',
             'size', 
             'bit_map_shape', 
             'gt_type',
             'compute_measures', 
             'compute_measures_static', 
             'get_measures_info',
             'process_genotype', 
             'is_valid_genotype', 
             'num_tasks'])
        self.attributes_get_all = set(
            ['genotype', 
             'level_rendering', 
             'get_belief'])

        # Set some of these variables locally for quick access (one time!)
        self._set_local_variables()

    def _set_local_variables(self):
        self._genotype_lower_bounds = self.genotype_lower_bounds
        self._genotype_lower_bounds = self.genotype_lower_bounds
        self._genotype_upper_bounds = self.genotype_upper_bounds
        self._genotype_size = self.genotype_size
        self._genotype_bounds = self.genotype_bounds
        self._size = self.size
        self._gt_type = self.gt_type

    def reset_mdp(self, index=None):
        if index is None: 
            # SubprocVecEnv does not have index---we cannot assume we're getting
            # VecNormalize'd envs
            obs = self.venv.reset_mdp()
        else:
            obs = self.venv.reset_mdp(index=index)
        obs = tensor(obs, device=self.device)
        return obs

    def reset(self, index=None, tasks=None):
        seeds = None  # will overwrite if they're applicable
        if index is not None:
            raise NotImplementedError("index: {}".format(index))
        # 1) PLR: Sample level and send seed to environment, if applicable
        if self.level_sampler and tasks is None:
            # Task will only be supplied if we're using QD with PLR 
            # (level_sampler will still be defined)
            # NOTE: we can bypass using QD samples if we set 
            # args.dist.qd.use_plr_for_training; in that case, tasks will
            # not be passed in.
            if index is not None:
                raise NotImplementedError
            tasks, seeds = self._sample_levels()
        else:
            # Set dummy seeds for QD
            seeds = torch.zeros(self.venv.num_envs, dtype=torch.int, device=self.device)
        # 2) Reset the environment
        if tasks is None:
            # Increase seeds by multiple of how many environments there are
            self.cur_seeds += len(self.start_seeds)
            tasks = list(self.cur_seeds)
        else: 
            assert isinstance(tasks, list) or isinstance(tasks, np.ndarray)
        state = self.venv.reset(task=tasks)
        state = tensor(state, device=self.device)
        # 3) PLR: Return the level seed, if applicable
        if not self.level_sampler:
            seeds = None
        else:
            assert seeds is not None
        return state, seeds
    
    def _sample_levels(self):
        tasks = []
        seeds = torch.zeros(self.venv.num_envs, dtype=torch.int)
        for e in range(self.venv.num_envs):
            seed = self.level_sampler.sample('gae')
            seeds[e] = seed
            # This calls SeededSubprocVecEnv.seed()
            self.venv.seed(seed, e)
            if self.level_store is not None:  # for PLR-gen
                task = solution_from_seed(
                    seed, 
                    self.level_store, 
                    self._genotype_lower_bounds, 
                    self._genotype_upper_bounds, 
                    self._genotype_size, 
                    gt_type=self._gt_type)
                tasks.append(task)
            else: 
                tasks.append(seed)
        return tasks, seeds

    def step_async(self, actions):
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        st0 = time.time()
        state, reward, done, info = self.venv.step_wait()
        et0 = time.time()
        info[0]['time/ES-VecPyTorch.step_wait;self.venv.step_wait'] = et0 - st0
        st1 = time.time()
        state = tensor(state, device=self.device)
        # NOTE: We can't just use the `tensor` function below since the
        # unsqueeze is necessary for later dimensionality expectations
        if isinstance(reward, list):  # raw + normalised
            reward = [torch.tensor(r, dtype=torch.float32, device=self.device).unsqueeze(dim=1) for r in reward]
            # reward = [torch.from_numpy(r).unsqueeze(dim=1).float().to(self.device) for r in reward]
        else:
            reward = torch.tensor(reward, dtype=torch.float32, device=self.device).unsqueeze(dim=1)
            # reward = torch.from_numpy(reward).unsqueeze(dim=1).float().to(self.device)
        et1 = time.time()
        info[0]['time/ES-VecPyTorch.step_wait;REST'] = et1 - st1
        return state, reward, done, info

    def __getattr__(self, attr):
        """ If env does not have the attribute then call the attribute in the wrapped_env """
        if attr in self.attributes_get_first:
            # This will get the attribute value for the first env in the env list.
            # We assume all envs have the same value for this attribute.
            return self.unwrapped.get_env_attr(attr)

        if attr in self.attributes_get_all:
            # These are attributes that have different values for each env
            attributes = self.unwrapped.get_env_attrs(attr)

            # NOTE: Originally we needed to remove the list wrapper, but we 
            # no longer allow each remote to run multiple environments in 
            # series, so this consideration is no longer necessary.
            # attributes = [attr[0] for attr in attributes]  # remove list wrapper
            # logger.debug(f'After removing list wrapper for "{attr}": {attributes}')
            return attributes

        # NOTE: self.__getattribute__(attr) has already been attempted, which
        # is why we're in fallback mode (i.e. __getattr__)
        orig_attr = self.unwrapped.__getattribute__(attr) 

        if callable(orig_attr):
            def hooked(*args, **kwargs):
                result = orig_attr(*args, **kwargs)
                return result

            return hooked
        else:
            return orig_attr
