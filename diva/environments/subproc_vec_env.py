"""
Modified from https://github.com/openai/baselines
"""
import multiprocessing as mp
from multiprocessing import Process

import numpy as np

from diva.environments.vec_env import (
    CloudpickleWrapper,
    VecEnv,
    clear_mpi_env_vars,
)


class SubprocVecEnv(VecEnv):
    """
    VecEnv that runs multiple envs in parallel in subproceses and communicates with them via pipes.
    Recommended to use when num_envs > 1 and step() can be a bottleneck.
    """

    def __init__(self, env_fns):
        """
        Arguments:

        env_fns: iterable of callables -  functions that create envs to run in subprocesses. Need to be cloud-pickleable
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        ctx = mp.get_context('spawn')
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(nenvs)])

        self.processes = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            # Clear MPI env vars to avoid subprocesses starting MPI
            clear_mpi_env_vars()
            args = (work_remote, CloudpickleWrapper(env_fn))
            process = Process(target=self.worker, args=args, daemon=True)
            process.start()
            self.processes.append(process)
            work_remote.close()
        self.ps = self.processes

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        self.viewer = None
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    @staticmethod
    def worker(remote, env_fn_wrapper):
        attributes_get_first = set(
            ['num_states', 
             '_max_episode_steps', 
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
             'get_measures_info', 
             'process_genotype', 
             'is_valid_genotype', 
             'compute_measures_static'])
        env = env_fn_wrapper.x()
        try:
            while True:
                cmd, data = remote.recv()
                if cmd == 'step':
                    ob, reward, done, info = env.step(data)
                    remote.send((ob, reward, done, info))
                elif cmd == 'reset':
                    if data is not None:
                        # Calls VariBAD wrapper in wrappers.py, which resets task
                        # first and then environment
                        ob = env.reset(task=data)
                    else:
                        ob = env.reset()
                    remote.send(ob)
                elif cmd == 'reset_mdp':
                    ob = env.reset_mdp()
                    remote.send(ob)
                elif cmd == 'render':
                    remote.send(env.render(mode='rgb_array'))
                elif cmd == 'close':
                    remote.close()
                    break
                elif cmd == 'get_spaces':
                    remote.send((env.observation_space, env.action_space))
                elif cmd == 'get_task':
                    remote.send(env.get_task())
                elif cmd == 'task_dim':
                    remote.send(env.task_dim)
                elif cmd == 'get_belief':
                    remote.send(env.get_belief())
                elif cmd == 'belief_dim':
                    remote.send(env.belief_dim)
                elif cmd == 'num_tasks':
                    remote.send(env.num_tasks)
                elif cmd == 'reset_task':
                    env.unwrapped.reset_task()
                elif cmd == 'get_spaces_spec':
                    remote.send(CloudpickleWrapper((env.observation_space, env.action_space, env.spec)))
                elif cmd in attributes_get_first:
                    # For these, we assume it is the same for all tasks in distribution
                    remote.send(getattr(env.unwrapped, cmd))
                else:
                    # Try to get the attribute directly
                    remote.send(getattr(env.unwrapped, cmd))
        except KeyboardInterrupt:
            print('SubprocVecEnv worker: got KeyboardInterrupt')
        finally:
            env.close()

    def step_async(self, actions):
        self._assert_not_closed()
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        self._assert_not_closed()
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return _flatten_obs(obs), np.stack(rews), np.stack(dones), infos

    def reset(self, task=None):
        self._assert_not_closed()

        if task is not None: 
            assert len(task) == len(self.remotes)

        for i, remote in enumerate(self.remotes):
            remote.send(('reset', task[i]))

        obs = [remote.recv() for remote in self.remotes]

        obs = _flatten_obs(obs)
        return obs

    def close_extras(self):
        self.closed = True
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()

    def get_images(self):
        self._assert_not_closed()
        for pipe in self.remotes:
            pipe.send(('render', None))
        imgs = [pipe.recv() for pipe in self.remotes]
        return imgs

    def _assert_not_closed(self):
        assert not self.closed, "Trying to operate on a SubprocVecEnv after calling close()"

    def get_env_attr(self, attr):
        self.remotes[0].send((attr, None))
        return self.remotes[0].recv()

    def get_env_attrs(self, attr):
        for remote in self.remotes:
            remote.send((attr, None))
        vals = [remote.recv() for remote in self.remotes]
        return vals

    def get_task(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('get_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def get_belief(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('get_belief', None))
        return np.stack([remote.recv() for remote in self.remotes])
    
    def __del__(self):
        if not self.closed:
            self.close()


def _flatten_obs(obs):
    assert isinstance(obs, (list, tuple))
    assert len(obs) > 0

    if isinstance(obs[0], dict):
        keys = obs[0].keys()
        return {k: np.stack([o[k] for o in obs]) for k in keys}
    else:
        return np.stack(obs)

def _flatten_list(l):
    assert isinstance(l, (list, tuple))
    assert len(l) > 0
    assert all([len(l_) > 0 for l_ in l])

    return [l__ for l_ in l for l__ in l_]