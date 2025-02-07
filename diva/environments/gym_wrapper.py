# From https://github.com/deepmind/bsuite/blob/595ca329e48500f3a98df5dc30e978927740c35e/bsuite/utils/gym_wrapper.py#L31
from typing import Any, Dict, Optional, Tuple, Union  # noqa: F401

import dm_env
import gym
import numpy as np
from dm_env import specs
from gym import spaces

# OpenAI gym step format = obs, reward, is_finished, other_info
_GymTimestep = Tuple[np.ndarray, float, bool, Dict[str, Any]]


class GymFromDMEnv(gym.Env):
  """A wrapper that converts a dm_env.Environment to an OpenAI gym.Env."""

  metadata = {'render.modes': ['human', 'rgb_array']}

  def __init__(self, env: dm_env.Environment):
    self._env = env  # type: dm_env.Environment
    self._last_observation = None  # type: Optional[np.ndarray]
    self._last_time_step = None  # type: Optional[dm_env.TimeStep]
    self.viewer = None
    self.game_over = False  # Needed for Dopamine agents.

  def step(self, action: int) -> _GymTimestep:
    timestep = self._env.step(action)
    self._last_observation = timestep.observation
    reward = timestep.reward or 0.
    if timestep.last():
      self.game_over = True
    self._last_time_step = timestep
    return timestep.observation, reward, timestep.last(), {}

  def reset(self) -> np.ndarray:
    self.game_over = False
    timestep = self._env.reset()
    self._last_time_step = timestep
    self._last_observation = timestep.observation
    return timestep.observation

  def render(self, mode: str = 'rgb_array') -> Union[np.ndarray, bool]:
    if self._last_observation is None:
      raise ValueError('Environment not ready to render. Call reset() first.')

    if mode == 'rgb_array':
      return self._last_observation

    if mode == 'human':
      if self.viewer is None:
        # pylint: disable=import-outside-toplevel
        # pylint: disable=g-import-not-at-top
        from gym.envs.classic_control import rendering
        self.viewer = rendering.SimpleImageViewer()
      self.viewer.imshow(self._last_observation)
      return self.viewer.isopen

  @property
  def action_space(self) -> spaces.Discrete:
    action_spec = self._env.action_spec()
    if isinstance(action_spec, specs.BoundedArray):
        # Add 1 because the range is inclusive of the maximum
        return spaces.Discrete(action_spec.maximum + 1)
    elif isinstance(action_spec, specs.DiscreteArray):
        return spaces.Discrete(action_spec.num_values)
    else:
        raise ValueError(f"Unexpected action_spec type: {type(action_spec)}")

  @property
  def observation_space(self) -> spaces.Space:
    obs_spec = self._env.observation_spec()
    # If the observation_spec is a dictionary
    if isinstance(obs_spec, dict):
        if len(obs_spec) != 1:
            raise NotImplementedError("Currently, we only support dictionaries with a single element in the observation space.")
        # Retrieve the first value in the dictionary
        obs_spec = list(obs_spec.values())[0]
    # Determine the type of space based on the instance type of obs_spec
    if isinstance(obs_spec, specs.BoundedArray):
        return spaces.Box(
            low=float(obs_spec.minimum),
            high=float(obs_spec.maximum),
            shape=obs_spec.shape,
            dtype=obs_spec.dtype)
    elif isinstance(obs_spec, specs.Array):
        return spaces.Box(
            low=-float('inf'),
            high=float('inf'),
            shape=obs_spec.shape,
            dtype=obs_spec.dtype)
    else:
        raise ValueError(f"Unexpected observation_spec type: {type(obs_spec)}")

  @property
  def reward_range(self) -> Tuple[float, float]:
    reward_spec = self._env.reward_spec()
    if isinstance(reward_spec, specs.BoundedArray):
      return reward_spec.minimum, reward_spec.maximum
    return -float('inf'), float('inf')

  # def __getattr__(self, attr):
  #   """Delegate attribute access to underlying environment."""
  #   return getattr(self._env, attr)