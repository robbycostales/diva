import numpy as np
import torch
import xxhash
from loguru import logger as logur

from diva.environments import utils as utl
from diva.environments.make_envs import make_vec_envs
from diva.environments.utils import EnvLatents
from diva.utils.torch import DeviceConfig


def hash_sol(sol):
    h = xxhash.xxh64()
    h.update(sol)
    return h.intdigest()


def simulate(args,
             policy,
             ret_rms,
             iter_idx,
             tasks,
             policy_storage,
             encoder=None,
             num_episodes=None,
             qd_tasks=None,
             plr_level_sampler=None,
             level_seeds=None,
             qd_envs=None,
             ):
    """ Adapted from evaluate function in evalution.py """
    env_name = args.domain.env_name
    if num_episodes is None:
        num_episodes = args.domain.episodes_per_trial

    if qd_tasks is not None:
        num_processes = len(qd_tasks)
    else:
        num_processes = args.policy.num_processes

    # --- set up the things we want to log ---

    # for each process, we log the returns during the first, second, ... episode
    # (such that we have a minimum of [num_episodes]; the last column is for
    #  any overflow and will be discarded at the end, because we need to wait until
    #  all processes have at least [num_episodes] many episodes)
    returns_per_episode = torch.zeros((num_processes, num_episodes + 1)).to(DeviceConfig.DEVICE)

    # --- initialise environments and latents ---

    if qd_envs is None:
        envs, _ = make_vec_envs(env_name,
                                seed=args.seed * 42 + iter_idx,
                                num_processes=num_processes,
                                gamma=args.policy.gamma,
                                device=DeviceConfig.DEVICE,
                                rank_offset=num_processes + 1,  # to use diff tmp folders than main processes
                                episodes_per_trial=num_episodes,
                                normalise_rew=args.policy.norm_rew,
                                plr=False,
                                ret_rms=ret_rms,
                                tasks=tasks,
                                add_done_info=args.domain.episodes_per_trial > 1,
                                qd_tasks=None,  # NOTE: we set this below
                                dense_rewards=args.dense_rewards,
                                **args.vec_env_kwargs
                                )
    else:
        envs = qd_envs

    num_steps = envs._max_episode_steps
    logur.debug('QD > Simulating QD solutions...')
    # reset environments
    prev_state, belief, task, ls = \
        utl.reset_env(envs, args, task=qd_tasks, eval=True, num_processes=num_processes)
    if level_seeds is None:
        level_seeds = ls
    else:
        level_seeds = torch.unsqueeze(level_seeds, dim=1)
    # this counts how often an agent has done the same task already
    task_count = torch.zeros(num_processes).long().to(DeviceConfig.DEVICE)

    # Insert initial observation / embeddings to rollout storage
    if args.domain.state_is_dict:
        for k in prev_state.keys():
            policy_storage.prev_state[k][0].copy_(prev_state[k])
    else:
        policy_storage.prev_state[0].copy_(prev_state)

    if encoder is not None:
        # reset latent state to prior
        latent_sample, latent_mean, latent_logvar, hidden_state = \
            encoder.prior(num_processes)
    else:
        latent_sample = latent_mean = latent_logvar = hidden_state = None
    env_latents = EnvLatents(latent_sample, latent_mean, latent_logvar, hidden_state)

    for episode_idx in range(num_episodes):
        # NOTE: we treat each episode like a set of args.policy.num_steps in
        #       MetaLearner.train()
        hidden_state = torch.squeeze(hidden_state, dim=0)
        latent_sample = torch.squeeze(latent_sample, dim=0)
        latent_mean = torch.squeeze(latent_mean, dim=0)
        latent_logvar = torch.squeeze(latent_logvar, dim=0)
        # Make sure we emptied buffers
        assert len(policy_storage.latent_mean) == 0  
        # Add initial hidden state we just computed to the policy storage
        policy_storage.hidden_states[0].copy_(hidden_state)
        policy_storage.latent_samples.append(latent_sample.clone())
        policy_storage.latent_mean.append(latent_mean.clone())
        policy_storage.latent_logvar.append(latent_logvar.clone())

        for step_idx in range(num_steps):

            with torch.no_grad():

                value, action = utl.select_action(
                    args=args, 
                    policy=policy, 
                    state=prev_state, 
                    belief=belief,
                    task=task, 
                    env_latents=env_latents,
                    deterministic=False
                )

            # Observe reward and next obs
            [next_state, belief, task], rewards, done, infos = \
                utl.env_step(envs, action, args)
            
            if isinstance(rewards, list) and len(rewards) == 2:
                # Using vector norm wrapper on rewards
                rew_raw, rew_normalised = rewards
            else: 
                # Not using vector norm wrapper on rewards
                rew_raw, rew_normalised = rewards, rewards
            done_mdp = [info['done_mdp'] for info in infos]

            # Compute next embedding (for next for loop and/or value prediction
            # bootstrap)
            if encoder is not None:
                # Update the hidden state
                env_latents = utl.update_encoding(
                    encoder=encoder,
                    next_obs=next_state,
                    action=action,
                    reward=rew_raw,
                    done=None,  # NOTE: Notable diff from training
                    env_latents=env_latents
                )
                latent_sample = env_latents.z_samples
                latent_mean = env_latents.z_means
                latent_logvar = env_latents.z_logvars
                hidden_state = env_latents.hs

            # Add step rewards to returns per episode
            returns_per_episode[range(num_processes), task_count] += rew_raw.view(-1)

            # Reset environments that are done
            for i in np.argwhere(done_mdp).flatten():
                # count task up, but cap at num_episodes + 1  
                # NOTE: zero-indexed, so no +1
                task_count[i] = min(task_count[i] + 1, num_episodes)  
            if np.sum(done) > 0:
                done_indices = np.argwhere(done.flatten()).flatten()
                next_state, belief, task, ls = \
                    utl.reset_env(envs, args, indices=done_indices, 
                                  state=next_state, task=qd_tasks, eval=True, 
                                  num_processes=num_processes)
                
            # NOTE: No need to add to a VAE buffer like in training

            # Add the obs before reset to the policy storage
            if isinstance(next_state, dict):
                for k in next_state.keys():
                    policy_storage.next_state[k][step_idx] = \
                        next_state[k].clone()
            else:
                policy_storage.next_state[step_idx] = next_state.clone()

            # Compute necessary values for policy storage insertion (taking
            # inspiration from MetaLearner.train())
            done = torch.from_numpy(np.array(done, dtype=int)).to(
                    DeviceConfig.DEVICE).float().view((-1, 1))
            masks_done = torch.FloatTensor(
                    [[0.0] if done_ else [1.0] for done_ in done]).to(DeviceConfig.DEVICE)
            bad_masks = torch.FloatTensor(
                    [[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos]).to(DeviceConfig.DEVICE)
            cliffhanger_masks = torch.FloatTensor(
                    [[0.0] if 'cliffhanger' in info.keys() else [1.0] for info in infos]).to(DeviceConfig.DEVICE)
            # Add experience to policy buffer
            policy_storage.insert(state=next_state, belief=belief,
                task=task, actions=action, rewards_raw=rew_raw,
                rewards_normalised=rew_normalised, value_preds=value,
                masks=masks_done, bad_masks=bad_masks, 
                cliffhanger_masks=cliffhanger_masks, done=done,
                hidden_states=hidden_state.squeeze(0),
                latent_sample=latent_sample, latent_mean=latent_mean,
                latent_logvar=latent_logvar, level_seeds=level_seeds
            )
            prev_state = next_state
        
        # This is the end of the episode; we'll treat it like the end of the
        # number of rollout steps in MetaLearner.train(), where the updates occur:
        policy_storage.before_update(policy.actor_critic)
        if plr_level_sampler is not None:
            plr_level_sampler.update_with_rollouts(policy_storage)
        policy_storage.after_update()
        if plr_level_sampler is not None:
            plr_level_sampler.after_update()
    
    if qd_envs is None:
        envs.close()
    return returns_per_episode[:, :num_episodes], plr_level_sampler