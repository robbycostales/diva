"""
Taken from https://github.com/openai/baselines
"""
import numpy as np

from diva.environments.vec_env import VecEnvWrapper


class VecNormalize(VecEnvWrapper):
    """
    A vectorized wrapper that normalizes the observations
    and returns from an environment.
    """

    def __init__(self, venv, ob=False, ret=True, clipobs=10.0, cliprew=10.0, gamma=None, epsilon=1e-8,
                 normalise_rew=False, ret_rms=None):

        if gamma is None:
            gamma = 0.99

        VecEnvWrapper.__init__(self, venv)

        if ret_rms is not None:
            self.ret_rms = ret_rms
        else:
            self.ret_rms = RunningMeanStd(shape=()) if ret else None
        self.ob_rms = RunningMeanStd(shape=self.observation_space.shape) if ob else None

        self.normalise_rew = normalise_rew

        # clip params
        self.clipobs = clipobs
        self.cliprew = cliprew

        # set the running mean and std values
        if self.normalise_rew:
            if ret_rms is None:
                self.ret_rms = RunningMeanStd(shape=())
            else:
                self.ret_rms = ret_rms

        # discounted return for each environment
        self.ret = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon

        self.training = True

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def step_wait(self):
        # execute action
        obs, rews, news, infos = self.venv.step_wait()
        # update discounted return
        self.ret = self.ret * self.gamma + rews
        obs = self._obfilt(obs)
        self.ret[news] = 0.
        # normalise
        rews = self._rewfilt(rews)
        return obs, rews, news, infos

    def _rewfilt(self, rews):
        if self.normalise_rew:
            # update rolling mean / std
            if self.training:
                self.ret_rms.update(self.ret)
            # normalise
            rews_norm = np.clip(rews / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
            return [rews, rews_norm]
        else:
            return [rews, rews]

    def reset_mdp(self, index=None):
        if index is None:
            obs = self.venv.reset_mdp()
        else:
            self.venv.remotes[index].send(('reset_mdp', None))
            obs = self.venv.remotes[index].recv()
        return obs
    
    def _obfilt(self, obs):
        if self.ob_rms:
            self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def reset(self, index=None, task=None):
        self.ret = np.zeros(self.num_envs)
        if index is None:
            obs = self.venv.reset(task=task)
        else:
            try:
                self.venv.remotes[index].send(('reset', task))
                obs = self.venv.remotes[index].recv()
            except AttributeError:
                obs = self.venv.envs[index].reset(task=task)
        return obs

    def __getattr__(self, attr):
        """
        If env does not have the attribute then call the attribute in the wrapped_env
        """
        orig_attr = self.unwrapped.__getattribute__(attr)

        if callable(orig_attr):
            def hooked(*args, **kwargs):
                result = orig_attr(*args, **kwargs)
                return result

            return hooked
        else:
            return orig_attr

class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)


def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count