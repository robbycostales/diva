"""
Based on https://github.com/ikostrikov/pytorch-a2c-ppo-acktr
"""

from typing import Any, Optional

import gym
import torch
from gym import spaces
from loguru import logger

from diva.components.level_replay.envs import SeededSubprocVecEnv
from diva.components.level_replay.level_sampler import LevelSampler
from diva.components.level_replay.level_store import LevelStore
from diva.environments.subproc_vec_env import SubprocVecEnv
from diva.environments.vec_env import DummyVecEnv
from diva.environments.vec_normalize import VecNormalize
from diva.environments.vec_pytorch import VecPyTorch


def make_vec_envs(
        env_name, 
        seed: int, 
        num_processes: int, 
        gamma: float,
        device: torch.device, 
        episodes_per_trial: int,
        normalise_rew: bool, 
        ret_rms: Optional[Any], 
        tasks: Optional[Any],
        rank_offset: int = 0,
        add_done_info: Optional[Any] = None,
        qd: bool = False,
        qd_tasks: Optional[Any] = None,
        plr: bool = False,
        plr_level_sampler: Optional[Any] = None,
        plr_level_sampler_args: Optional[Any] = None,
        plr_env_generator: Optional[Any] = None,
        **kwargs):
    """ Make vectorised environments. """
    assert seed is not None  # Current assumption
    if qd_tasks is None or plr_env_generator == 'sb':
        qd_tasks = [None] * num_processes

    # Called by each individual environment
    def make_env(env_id, seed, rank, episodes_per_trial, tasks, add_done_info,
                 qd_task=None, **kwargs):
        def _thunk():
            from diva.environments.wrappers import TimeLimitMask, VariBadWrapper
            env = gym.make(env_id, **kwargs)
            if seed is not None:
                env.seed(seed + rank)
            if str(env.__class__.__name__).find('TimeLimit') >= 0:
                env = TimeLimitMask(env)
            env = VariBadWrapper(env=env, episodes_per_trial=episodes_per_trial, add_done_info=add_done_info)
            if qd_task is not None:
                env.reset_task(task=qd_task)
            return env

        return _thunk

    # 1) Create environments and apply (Seeded)SubprocVecEnv wrapper
    logger.debug('make_vec_envs: Defining environments')

    env_fns = [
        make_env(
            env_id=env_name, 
            seed=seed,
            rank=rank_offset + i,
            episodes_per_trial=episodes_per_trial,
            tasks=tasks,
            add_done_info=add_done_info,
            qd_task=qd_tasks[i],  # Used to set QD genotype
            **kwargs) 
        for i in range(num_processes)]
    logger.debug('make_vec_envs: Creating environments')
    if plr:
        # PLR: Create SeededSubprocVecEnv
        envs = SeededSubprocVecEnvWrapper(env_fns)
    else:
        # non-PLR: Create (vanilla) SubprocVecEnv
        envs = SubprocVecEnv(env_fns) if len(env_fns) > 1 else DummyVecEnv(env_fns)
    logger.info(f'Making {num_processes} environments')

    init_seeds = [seed + rank_offset + i for i in range(num_processes)]

    # 2) Add functional wrappers such as VecNormalize
    if isinstance(envs.observation_space, (spaces.Box)):
        # Only perform VecNormalization if observation space is shape 1
        # (Mujoco: https://github.com/dannysdeng/dqn-pytorch/blob/master/env.py)
        if len(envs.observation_space.shape) == 1:
            envs = VecNormalize(envs, normalise_rew=normalise_rew, ret_rms=ret_rms, gamma=gamma)
    elif isinstance(envs.observation_space, spaces.Dict):
        envs = VecNormalize(envs, normalise_rew=normalise_rew, ret_rms=ret_rms, gamma=gamma)
    else:
        raise NotImplementedError

    # 3) Create VecPyTorch envs (the highest-level wrapper)
    if plr:
        if plr_level_sampler is None:  # Otherwise we assume it is passed in
            # Initialize sampler
            plr_level_sampler = LevelSampler(
                [],  # init_seeds: we don't need these since we're using full distribution
                envs.observation_space, 
                envs.action_space,
                sample_full_distribution=True,
                **(plr_level_sampler_args or {'strategy': 'random'}))
        # Create level store to store genotypes (if applicable)
        plr_level_store = LevelStore() if (plr_env_generator == 'gen' or qd) else None
        envs = VecPyTorch(envs, device, level_sampler=plr_level_sampler, level_store=plr_level_store)
    else:
        plr_level_store = None
        envs = VecPyTorch(envs, device, start_seeds=init_seeds)

    # 4) Return envs and PLR-specific objects (if applicable)
    return envs, (plr_level_sampler, plr_level_store)


class SeededSubprocVecEnvWrapper(SeededSubprocVecEnv):
    """ Used because we get error when using SeededSubprocVecEnv directly. """
    def __init__(self, env_fns):
        super(SeededSubprocVecEnv, self).__init__(env_fns)