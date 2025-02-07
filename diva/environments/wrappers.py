import inspect
import time

import gym
import numpy as np
from gym import spaces
from gym.envs.registration import load

try:
    # this is to suppress some warnings (in the newer mujoco versions)
    gym.logger.set_level(40)
except AttributeError:
    pass


def mujoco_wrapper(entry_point, **kwargs):
    # Load the environment from its entry point
    env_cls = load(entry_point)
    env = env_cls(**kwargs)
    return env


class VariBadWrapper(gym.Wrapper):
    def __init__(self,
                 env,
                 episodes_per_trial,
                 add_done_info=None,  # force to turn this on/off
                 ):
        """
        Wrapper, creates a multi-episode (BA)MDP around a one-episode MDP. 
        Automatically deals with
        - horizons H in the MDP vs horizons H+ in the BAMDP,
        - resetting the tasks
        - adding the done info to the state (might be needed to make 
          states markov)
        """

        super().__init__(env)

        # make sure we can call these attributes even if the orig env does not 
        # have them
        if not hasattr(self.env.unwrapped, 'task_dim'):
            self.env.unwrapped.task_dim = 0
        if not hasattr(self.env.unwrapped, 'belief_dim'):
            self.env.unwrapped.belief_dim = 0
        if not hasattr(self.env.unwrapped, 'get_belief'):
            self.env.unwrapped.get_belief = lambda: None
        if not hasattr(self.env.unwrapped, 'num_states'):
            self.env.unwrapped.num_states = None

        # Whether or not the environment has a episode_num argument for reset
        if 'episode_num' in inspect.signature(self.env.unwrapped.reset).parameters:
            self.pass_episode_num = True
        else:
            self.pass_episode_num = False

        # If add_done_info not defined, do it by default if multiple episodes
        if add_done_info is None:
            if episodes_per_trial > 1:
                self.add_done_info = True
            else:
                self.add_done_info = False
        else:
            # If set, we just use the setting
            self.add_done_info = add_done_info
        
        # Add done info to the observation space if applicable
        if self.add_done_info:
            # Original VariBAD environments all fell into this category
            if (isinstance(self.observation_space, spaces.Box) or 
                isinstance(self.observation_space, gym.spaces.Box)):
                # isinstance(self.observation_space, rand_param_envs.gym.spaces.box.Box)):
                # Can't add additional info for obs of more than 1D
                if len(self.observation_space.shape) > 1:
                    raise ValueError
                # Done info is extra bit we are adding to the end of the array
                self.observation_space = spaces.Box(
                    low=np.array([*self.observation_space.low, 0]),
                    # Shape will be deduced from this
                    high=np.array([*self.observation_space.high, 1])
                )
            elif isinstance(self.observation_space, spaces.Dict):
                # Since observation space already dictionary, we just add a key
                # Copy existing spaces and add the 'done' key
                new_spaces = {key: self.observation_space[key] for key in self.observation_space.spaces}
                new_spaces['done'] = spaces.Box(low=np.array((0,)), high=np.array((1,)))
                self.observation_space = spaces.Dict(new_spaces)
            else:
                # "Space", "Discrete", "MultiDiscrete", "MultiBinary", "Tuple",
                # "Dict", "flatdim", "flatten", "unflatten"
                raise NotImplementedError

        # Calculate horizon length H^+
        self.episodes_per_trial = episodes_per_trial
        # Counts the number of episodes
        self.episode_count = 0

        # Count timesteps in BAMDP
        self.step_count_bamdp = 0.0
        # The horizon in the BAMDP is the one in the MDP times the number of \
        # episodes per trial,
        # and if we train a policy that maximises the return over all episodes
        # we add transitions to the reset start in-between episodes
        try:
            self.horizon_bamdp = \
                self.episodes_per_trial * self.env._max_episode_steps
        except AttributeError:
            self.horizon_bamdp = \
                self.episodes_per_trial * self.env.unwrapped._max_episode_steps

        # Add dummy timesteps in-between episodes for resetting the MDP
        self.horizon_bamdp += self.episodes_per_trial - 1

        # This tells us if we have reached the horizon in the underlying MDP
        self.done_mdp = True

    def reset(self, task=None):
        """ Resets the BAMDP """
        # Reset task
        self.env.reset_task(task)
        # Normal reset
        # try:
        state = self.env.reset()
        # except AttributeError:
        #     state = self.env.unwrapped.reset()

        self.episode_count = 0
        self.step_count_bamdp = 0
        self.done_mdp = False
        if self.add_done_info:
            state = np.concatenate((state, [0.0]))
        return state

    def reset_mdp(self):
        """ Resets the underlying MDP only (*not* the task). """
        if self.pass_episode_num:
            state = self.env.reset(episode_num=self.episode_count)
        else:
            state = self.env.reset()
        if self.add_done_info:
            state = np.concatenate((state, [0.0]))
        self.done_mdp = False
        return state

    def step(self, action):
        # Do normal environment step in MDP
        st0 = time.time()
        state, reward, self.done_mdp, info = self.env.step(action)
        info['done_mdp'] = self.done_mdp
        et0 = time.time()
        info['time/ES-VariBADWrapper.step'] = et0 - st0
        
        st0 = time.time()
        if self.add_done_info:
            state = np.concatenate((state, [float(self.done_mdp)]))

        self.step_count_bamdp += 1
        # If we want to maximise performance over multiple episodes,
        # only say "done" when we collected enough episodes for this task
        done_bamdp = False
        if self.done_mdp:
            self.episode_count += 1
            if self.episode_count == self.episodes_per_trial:
                done_bamdp = True
        et0 = time.time()
        info['time/ES-VariBADWrapper.REST'] = et0 - st0

        st0 = time.time()
        if self.done_mdp and not done_bamdp:
            info['start_state'] = self.reset_mdp()
        et0 = time.time()
        info['time/ES-VariBADWrapper.reset_mdp'] = et0 - st0
        
        return state, reward, done_bamdp, info

    def __getattr__(self, attr):
        """
        If env does not have the attribute then call the attribute in the 
        wrapped_env (This one's only needed for mujoco 131)
        """
        # try:
        #     orig_attr = self.__getattribute__(attr)
        # except AttributeError:
        orig_attr = self.unwrapped.__getattribute__(attr)
        if callable(orig_attr):
            def hooked(*args, **kwargs):
                result = orig_attr(*args, **kwargs)
                return result

            return hooked
        else:
            return orig_attr


class TimeLimitMask(gym.Wrapper):

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if done and self.env._max_episode_steps == self.env._elapsed_steps:
            info['bad_transition'] = True
        return obs, rew, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def __getattr__(self, attr):
        """
        If env does not have the attribute then call the attribute in the 
        wrapped_env (This one's only needed for mujoco 131)
        """
        # try:
        #     orig_attr = self.__getattribute__(attr)
        # except AttributeError:
        orig_attr = self.unwrapped.__getattribute__(attr)
        if callable(orig_attr):
            def hooked(*args, **kwargs):
                result = orig_attr(*args, **kwargs)
                return result

            return hooked
        else:
            return orig_attr