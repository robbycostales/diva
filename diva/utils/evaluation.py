import copy
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger as logur

from diva.environments import utils as utl
from diva.environments.make_envs import make_vec_envs
from diva.environments.utils import EnvLatents
from diva.utils.torch import DeviceConfig
from diva.utils.visualization import plot_episode_data


def evaluate(args,
             policy,
             ret_rms,
             iter_idx,
             tasks,
             encoder=None,
             num_episodes=None,
             create_video=False,
             vae=None,
             intrinsic_reward=None,
             logger=None,
             ):
    env_name = args.domain.eval_env_name
    if num_episodes is None:
        num_episodes = args.domain.episodes_per_trial
    num_processes = args.policy.num_processes

    # --- set up the things we want to log ---

    # for each process, we log the returns during the first, second, ... episode
    # (such that we have a minimum of [num_episodes]; the last column is for
    #  any overflow and will be discarded at the end, because we need to wait until
    #  all processes have at least [num_episodes] many episodes)

    logur.debug(f'tasks: {tasks}')
    
    returns_per_episode = torch.zeros((num_processes, num_episodes + 1)).to(DeviceConfig.DEVICE)
    sparse_returns_per_episode = torch.zeros((num_processes, num_episodes + 1)).to(DeviceConfig.DEVICE)
    dense_returns_per_episode = torch.zeros((num_processes, num_episodes + 1)).to(DeviceConfig.DEVICE)
    returns_bonus_per_episode = torch.zeros((num_processes, num_episodes + 1)).to(DeviceConfig.DEVICE)

    # individual
    returns_bonus_belief_per_episode = torch.zeros((num_processes, num_episodes + 1)).to(DeviceConfig.DEVICE)
    returns_bonus_state_per_episode = torch.zeros((num_processes, num_episodes + 1)).to(DeviceConfig.DEVICE)
    returns_bonus_hyperstate_per_episode = torch.zeros((num_processes, num_episodes + 1)).to(DeviceConfig.DEVICE)  
    returns_bonus_vae_loss_per_episode = torch.zeros((num_processes, num_episodes + 1)).to(DeviceConfig.DEVICE)

    successes = torch.zeros((num_processes, num_episodes + 1), dtype=torch.bool).to(DeviceConfig.DEVICE)
    
    # Default dict where each dict is the same value as successes above
    final_metrics = defaultdict(lambda: torch.zeros((num_processes, num_episodes + 1)).to(DeviceConfig.DEVICE))

    # --- initialise environments and latents ---
    envs, plr_components = make_vec_envs(env_name,
                         seed=args.seed * 42 + iter_idx,
                         num_processes=num_processes,
                         gamma=args.policy.gamma,
                         device=DeviceConfig.DEVICE,
                         rank_offset=num_processes + 1,  # to use diff tmp folders than main processes
                         episodes_per_trial=num_episodes,
                         normalise_rew=args.policy.norm_rew,
                         ret_rms=ret_rms,
                         tasks=tasks,
                         plr=False,  # We never use PLR in eval
                         add_done_info=args.domain.episodes_per_trial > 1,
                         visualize=create_video,
                         dense_rewards=args.dense_rewards,
                         **args.vec_env_kwargs
                         )
    num_steps = envs._max_episode_steps

    # reset environments
    state, belief, task, level_seeds = utl.reset_env(envs, args, eval=True)

    if create_video:
        all_images = [[[] for j in range(num_episodes)] 
                      for i in range(num_processes)]

    # this counts how often an agent has done the same task already
    task_count = torch.zeros(num_processes).long().to(DeviceConfig.DEVICE)

    # VAE belief values (to plot)
    episode_latent_means = [[] for _ in range(num_episodes)]
    episode_latent_logvars = [[] for _ in range(num_episodes)]
    episode_events = [[dict() for _ in range(num_processes)] 
                      for _ in range(num_episodes)]  # Vertical lines to plot

    if encoder is not None:
        latent_sample, latent_mean, latent_logvar, hidden_state = encoder.prior(num_processes)
    else:
        latent_sample = latent_mean = latent_logvar = hidden_state = None
    env_latents = EnvLatents(latent_sample, latent_mean, latent_logvar, hidden_state)

    # Copying the same procedure we using for logging training stats
    eval_env_step_info_stats = defaultdict(list)

    for episode_idx in range(num_episodes):
        # TODO: remove done_mdps logic, since it's only true at the end of the
        # fixed number of steps; not when the agent fails/achieves the goal;
        # we would need a different flag to indicate when we should cut off 
        # the visualization. Or we'd need to make the episodes different lengths, 
        # but the current codebase does not support this...
        done_mdps = [False for _ in range(num_processes)]

        if create_video:
            # Get first image observation
            imgs = envs.get_images()
            for i in range(num_processes):
                if imgs[i] is not None:
                    all_images[i][episode_idx].append(imgs[i])

        for step_idx in range(num_steps):

            prev_state = copy.deepcopy(state)

            with torch.no_grad():
                _, action = utl.select_action(
                    args=args,                
                    policy=policy,
                    state=state,
                    belief=belief,
                    task=task,
                    env_latents=env_latents,
                    deterministic=True
                )

            # observe reward and next obs
            [state, belief, task], rewards, done, infos = \
                utl.env_step(envs, action, args)
            
            for i, info in enumerate(infos):
                # Add events to dicts
                if 'event' in info:
                    episode_events[episode_idx][i][step_idx] = info['event']
                # For final episode only
                if episode_idx == num_episodes - 1:
                    for k, v in info.items():
                        # NOTE: we don't log timings
                        if 'env/' in k:
                            # Want to specify these are values for the eval envs
                            eval_env_step_info_stats['eval-env-fin-ep' + k[3:]].append(v)

            if len(rewards) == 2:
                # Using vector norm wrapper on rewards
                rew_raw, rew_normalised = rewards
            else: 
                # Not using vector norm wrapper on rewards
                rew_raw = rewards
            done_mdp = [info['done_mdp'] for info in infos]
            for i in range(num_processes):
                if done_mdp[i]:
                    done_mdps[i] = True
            # TODO: Add sparse and dense reward info to the info dict
            # keep track of dense and sparse rewards
            sparse_rew = torch.tensor([info['sparse_reward'] for info in infos]).view(-1).to(DeviceConfig.DEVICE) if (
                        'sparse_reward' in infos[0]) else 0
            dense_rew = torch.tensor([info['dense_reward'] for info in infos]).view(-1).to(DeviceConfig.DEVICE) if (
                        'dense_reward' in infos[0]) else 0

            if create_video:
                imgs = envs.get_images()
                for i in range(num_processes):
                    if not done_mdps[i] and imgs[i] is not None:
                        all_images[i][episode_idx].append(imgs[i])

            if encoder is not None:
                # update the hidden state
                # import pdb; pdb.set_trace()
                env_latents = utl.update_encoding(
                    encoder=encoder,
                    next_obs=state,
                    action=action,
                    reward=rew_raw,
                    done=None,
                    env_latents=env_latents
                )
                latent_sample = env_latents.z_samples
                latent_mean = env_latents.z_means
                latent_logvar = env_latents.z_logvars
                hidden_state = env_latents.hs
                # Add latent mean and logvar to list (converting to numpy first)
                episode_latent_means[episode_idx].append(latent_mean.detach().cpu().numpy())
                episode_latent_logvars[episode_idx].append(latent_logvar.detach().cpu().numpy())

                # overwrite belief given by env with the latent/mean
                belief = torch.cat((latent_mean, latent_logvar), dim=1)

            # add rewards
            returns_per_episode[range(num_processes), task_count] += rew_raw.view(-1)
            sparse_returns_per_episode[range(num_processes), task_count] += sparse_rew
            dense_returns_per_episode[range(num_processes), task_count] += dense_rew

            # compute reward bonus for the current state/belief pair
            if args.add_exploration_bonus:
                rew_bonus, intrinsic_rew_state, \
                intrinsic_rew_belief, intrinsic_rew_hyperstate, \
                intrinsic_rew_vae_loss = intrinsic_reward.reward(
                    state=state,
                    belief=belief,
                    return_individual=True,
                    vae=vae,
                    latent_mean=[latent_mean],
                    latent_logvar=[latent_logvar],
                    batch_prev_obs=prev_state.unsqueeze(0),
                    batch_next_obs=state.unsqueeze(0),
                    batch_actions=action.unsqueeze(0),
                    batch_rewards=rew_raw.unsqueeze(0),
                    batch_tasks=task.unsqueeze(0),
                    )

                returns_bonus_per_episode[range(num_processes), task_count] += rew_bonus.view(-1)
                if args.exploration_bonus_state:
                    returns_bonus_state_per_episode[range(num_processes), task_count] += intrinsic_rew_state.view(-1)
                if args.exploration_bonus_belief:
                    returns_bonus_belief_per_episode[range(num_processes), task_count] += intrinsic_rew_belief.view(-1)
                if args.exploration_bonus_hyperstate:
                    returns_bonus_hyperstate_per_episode[
                        range(num_processes), task_count] += intrinsic_rew_hyperstate.view(-1)
                if args.exploration_bonus_vae_error and (vae is not None):
                    returns_bonus_vae_loss_per_episode[range(num_processes), task_count] += intrinsic_rew_vae_loss.view(
                        -1)

            # update success rates
            # NOTE: Just need to check if key is in first process info dict
            # because success is returned at every step for each
            if 'success' in infos[0]:  
                # |= does an in-place "or"
                successes[range(num_processes), task_count] |= torch.tensor([bool(info['success']) for info in infos]).to(DeviceConfig.DEVICE)

            for i in range(num_processes):
                for k, v in infos[i].items():
                    if 'final' in k:
                        final_metrics[k][i][episode_idx] = v
            
            for i in np.argwhere(done_mdp).flatten():
                # count task up, but cap at num_episodes + 1
                task_count[i] = min(task_count[i] + 1, num_episodes)  # zero-indexed, so no +1
            if np.sum(done) > 0:
                done_indices = np.argwhere(done.flatten()).flatten()
                state, belief, task, level_seeds = utl.reset_env(
                    envs, args, indices=done_indices, state=state, eval=True)
                
    # Create video from all images
    video_buffer = None
    if create_video:
        video_buffer = utl.video_from_images(fps=args.domain.video_fps, images=all_images)

    # Close environments
    envs.close()

    # Process returns etc. for logging
    returns_per_episode = returns_per_episode[:, :num_episodes]
    sparse_returns_per_episode = sparse_returns_per_episode[:, :num_episodes]
    dense_returns_per_episode = dense_returns_per_episode[:, :num_episodes]
    returns_bonus_per_episode = returns_bonus_per_episode[:, :num_episodes]
    returns_bonus_state_per_episode = returns_bonus_state_per_episode[:, :num_episodes]
    returns_bonus_belief_per_episode = returns_bonus_belief_per_episode[:, :num_episodes]
    returns_bonus_hyperstate_per_episode = returns_bonus_hyperstate_per_episode[:, :num_episodes]
    returns_bonus_vae_loss_per_episode = returns_bonus_vae_loss_per_episode[:, :num_episodes]
    success_per_episode = successes.float()[:, :num_episodes] if 'success' in infos[0] else None
    # rest are already processed

    la, ii = logger.add, iter_idx

    # Log the misc eval stats
    for k, v in eval_env_step_info_stats.items():
        la(k, np.mean(v), iter_idx)

    # Log the return avg/std across tasks (=processes)
    returns_avg = returns_per_episode.mean(dim=0)
    returns_std = returns_per_episode.std(dim=0)
    if success_per_episode is not None:
        success_avg = success_per_episode.mean(dim=0)
    sparse_returns_avg = sparse_returns_per_episode.mean(dim=0)
    dense_returns_avg = dense_returns_per_episode.mean(dim=0)
    returns_bonus_avg = returns_bonus_per_episode.mean(dim=0)
    returns_bonus_state_avg = returns_bonus_state_per_episode.mean(dim=0)
    returns_bonus_belief_avg = returns_bonus_belief_per_episode.mean(dim=0)
    returns_bonus_hyperstate_avg = returns_bonus_hyperstate_per_episode.mean(dim=0)
    returns_bonus_vae_loss_avg = returns_bonus_vae_loss_per_episode.mean(dim=0)
    for k, v in final_metrics.items():
        # Take mean over first dimension
        final_metrics[k] = v.mean(dim=0)

    la('eval/return_avg/diff', returns_avg[-1] - returns_avg[0], ii)
    if success_per_episode is not None:
        la('eval/success_avg/diff', success_avg[-1] - success_avg[0], ii)
    for metric_name, v in final_metrics.items():
        # NOTE: -2 because there's still an extra buffer slot at the
        # end from eval---we removed these for return and success, 
        # but not final metrics
        la(f'eval/{metric_name}/diff', v[-2] - v[0], ii)

    # Log episode-specific data
    for k in range(len(returns_avg)):
        la('eval/return_avg/episode_{}'.format(k + 1), returns_avg[k], ii)
        la('eval/return_std/episode_{}'.format(k + 1), returns_std[k], ii)
        if success_per_episode is not None:
            la('eval/success_avg/episode_{}'.format(k + 1), success_avg[k], ii)
        la('eval/return_avg/sparse/episode_{}'.format(k + 1), sparse_returns_avg[k], ii)
        la('eval/return_avg/dense/episode_{}'.format(k + 1), dense_returns_avg[k], ii)
        la('eval/return_avg/bonus/episode_{}'.format(k + 1), returns_bonus_avg[k], ii)
        
        for metric_name, v in final_metrics.items():
            la(f'eval/{metric_name}/episode_{k+1}', v[k], ii)

        if args.add_exploration_bonus:
            if args.exploration_bonus_state:
                la('eval/return_avg/bonus_state/episode_{}'.format(k + 1), returns_bonus_state_avg[k], ii)
            if args.exploration_bonus_belief:
                la('eval/return_avg/bonus_belief/episode_{}'.format(k + 1), returns_bonus_belief_avg[k], ii)
            if args.exploration_bonus_hyperstate:
                la('eval/return_avg/bonus_hyperstate/episode_{}'.format(k + 1), returns_bonus_hyperstate_avg[k], ii)
            if args.exploration_bonus_vae_error:
                la('eval/return_avg/bonus_vae_loss/episode_{}'.format(k + 1), returns_bonus_vae_loss_avg[k], ii)
    
    # Log video
    if create_video:
        video_name = utl.generate_video_name(ii+1)
        logger.add_video(f'eval/videos/{video_name}', video_buffer, ii+1, fps=args.domain.video_fps)
        
    # Log latent means and logvars plots
    if episode_latent_means is not None:
        for k in range(len(episode_latent_means)): 
            img = plot_episode_data(
                episode_latent_means,
                episode_latent_logvars,
                episode_events,
                episode_num=k,
                process_num=0, 
                ensemble_size=args.ensemble_size if args.vae.use_ensemble else None)
            img = utl.stitch_images([img], n=1)
            image_name = 'episode_{}.png'.format(k+1)
            logger.add_image(f'eval/latent_vals_over_trials/{image_name}', img, ii+1)
    return



def visualise_behaviour(args,
                        policy,
                        image_folder,
                        iter_idx,
                        ret_rms,
                        tasks,
                        encoder=None,
                        reward_decoder=None,
                        state_decoder=None,
                        task_decoder=None,
                        compute_rew_reconstruction_loss=None,
                        compute_task_reconstruction_loss=None,
                        compute_state_reconstruction_loss=None,
                        compute_kl_loss=None,
                        intrinsic_reward=None,
                        vae=None,
                        ):
    del intrinsic_reward, vae  # unused (TODO: see how HyperX uses these in visualizations)
    return
    # initialise environment
    env, plr_components = make_vec_envs(env_name=args.domain.env_name,
                        seed=args.seed * 42 + iter_idx,
                        num_processes=1,
                        gamma=args.policy.gamma,
                        device=DeviceConfig.DEVICE,
                        episodes_per_trial=args.domain.episodes_per_trial,
                        normalise_rew=args.policy.norm_rew, ret_rms=ret_rms,
                        rank_offset=args.policy.num_processes + 42,  # not sure if the temp folders would otherwise clash
                        plr=args.dist.use_plr,
                        tasks=tasks,
                        **args.vec_env_kwargs
                        )
    episode_task = torch.from_numpy(np.array(env.get_task())).to(DeviceConfig.DEVICE).float()

    # get a sample rollout
    unwrapped_env = env.venv.unwrapped.envs[0]
    if hasattr(env.venv.unwrapped.envs[0], 'unwrapped'):
        unwrapped_env = unwrapped_env.unwrapped
    if hasattr(unwrapped_env, 'visualise_behaviour'):
        # if possible, get it from the env directly
        # (this might visualise other things in addition)
        traj = unwrapped_env.visualise_behaviour(env=env,
                                                 args=args,
                                                 policy=policy,
                                                 iter_idx=iter_idx,
                                                 encoder=encoder,
                                                 reward_decoder=reward_decoder,
                                                 state_decoder=state_decoder,
                                                 task_decoder=task_decoder,
                                                 image_folder=image_folder,
                                                 )
        print('Getting viz. rollout from ENV')
    else:
        traj = get_test_rollout(args, env, policy, encoder)
        print('Getting viz. rollout from POLICY (default)')

    latent_means, latent_logvars, episode_prev_obs, episode_next_obs, episode_actions, episode_rewards, episode_returns = traj

    if latent_means is not None:
        plot_latents(latent_means, latent_logvars,
                     image_folder=image_folder,
                     iter_idx=iter_idx
                     )

        if not (args.vae.disable_decoder and args.vae.disable_kl_term):
            plot_vae_loss(args,
                          latent_means,
                          latent_logvars,
                          episode_prev_obs,
                          episode_next_obs,
                          episode_actions,
                          episode_rewards,
                          episode_task,
                          image_folder=image_folder,
                          iter_idx=iter_idx,
                          reward_decoder=reward_decoder,
                          state_decoder=state_decoder,
                          task_decoder=task_decoder,
                          compute_task_reconstruction_loss=compute_task_reconstruction_loss,
                          compute_rew_reconstruction_loss=compute_rew_reconstruction_loss,
                          compute_state_reconstruction_loss=compute_state_reconstruction_loss,
                          compute_kl_loss=compute_kl_loss,
                          )

    env.close()


def get_test_rollout(args, env, policy, encoder=None):
    num_episodes = args.domain.episodes_per_trial

    # --- initialise things we want to keep track of ---

    if encoder is not None:
        episode_latent_samples = [[] for _ in range(num_episodes)]
        episode_latent_means = [[] for _ in range(num_episodes)]
        episode_latent_logvars = [[] for _ in range(num_episodes)]
    else:
        curr_latent_sample = curr_latent_mean = curr_latent_logvar = None
        episode_latent_means = episode_latent_logvars = None

    # --- roll out policy ---

    # (re)set environment
    env.reset_task()
    state, belief, task, level_seeds = utl.reset_env(env, args)
    if isinstance(state, dict):
        state = {k: v.reshape((1, *v.shape)).to(DeviceConfig.DEVICE) for k, v, in state.items()}
    else:
        state = state.reshape((1, *state.shape)).to(DeviceConfig.DEVICE)

    task = task.view(-1) if task is not None else None

    # We need to initialize after we know the form of `state`
    
    if isinstance(state, dict):
        episode_prev_obs = {k: [[] for _ in range(num_episodes)] for k in state.keys()}
        episode_next_obs = {k: [[] for _ in range(num_episodes)] for k in state.keys()}
    else:
        episode_prev_obs = [[] for _ in range(num_episodes)]
        episode_next_obs = [[] for _ in range(num_episodes)]
    episode_actions = [[] for _ in range(num_episodes)]
    episode_rewards = [[] for _ in range(num_episodes)]

    episode_returns = []
    episode_lengths = []

    for episode_idx in range(num_episodes):

        curr_rollout_rew = []

        if encoder is not None:
            if episode_idx == 0:
                # reset to prior
                curr_latent_sample, curr_latent_mean, curr_latent_logvar, hidden_state = encoder.prior(1)
                curr_latent_sample = curr_latent_sample[0].to(DeviceConfig.DEVICE)
                curr_latent_mean = curr_latent_mean[0].to(DeviceConfig.DEVICE)
                curr_latent_logvar = curr_latent_logvar[0].to(DeviceConfig.DEVICE)
            episode_latent_samples[episode_idx].append(curr_latent_sample[0].clone())
            episode_latent_means[episode_idx].append(curr_latent_mean[0].clone())
            episode_latent_logvars[episode_idx].append(curr_latent_logvar[0].clone())

        for step_idx in range(1, env._max_episode_steps + 1):
            if isinstance(state, dict):
                state_clone = {k: v.clone() for k, v in state.items()}
                for k in state_clone.keys():
                    episode_prev_obs[k][episode_idx].append(state_clone[k])
            else:
                state_clone = state.clone()
                episode_prev_obs[episode_idx].append(state_clone)

            latent = utl.get_zs_for_policy(args,
                                               latent_sample=curr_latent_sample,
                                               latent_mean=curr_latent_mean,
                                               latent_logvar=curr_latent_logvar)
            if isinstance(state, dict):
                state_view = {k: torch.squeeze(v, dim=0) for k, v in state.items()}
            else:
                state_view = torch.squeeze(state, dim=0)

            _, action = policy.act(state=state_view, latent=latent, belief=belief, task=task, deterministic=True)
            action = action.reshape((1, *action.shape))

            # observe reward and next obs
            (state, belief, task), rewards, done, infos = utl.env_step(env, action, args)
            if len(rewards) == 2:
                # Using vector norm wrapper on rewards
                rew_raw, rew_normalised = rewards
            else: 
                # Not using vector norm wrapper on rewards
                rew_raw = rewards

            if isinstance(state, dict):
                state = {k: v.reshape((1, *v.shape)).to(DeviceConfig.DEVICE) for k, v, in state.items()}
            else:
                state = state.reshape((1, *state.shape)).to(DeviceConfig.DEVICE)
            task = task.view(-1) if task is not None else None

            if encoder is not None:
                # update task embedding
                curr_latent_sample, curr_latent_mean, curr_latent_logvar, hidden_state = encoder(
                    action.float().to(DeviceConfig.DEVICE),
                    state,
                    rew_raw.reshape((1, 1)).float().to(DeviceConfig.DEVICE),
                    hidden_state,
                    return_prior=False)

                episode_latent_samples[episode_idx].append(curr_latent_sample[0].clone())
                episode_latent_means[episode_idx].append(curr_latent_mean[0].clone())
                episode_latent_logvars[episode_idx].append(curr_latent_logvar[0].clone())
            
            if isinstance(state, dict):
                for k, v in state.items():
                    episode_next_obs[k][episode_idx].append(v.clone())
            else:
                episode_next_obs[episode_idx].append(state.clone())
            episode_rewards[episode_idx].append(rew_raw.clone())
            episode_actions[episode_idx].append(action.clone())

            if infos[0]['done_mdp']:
                break

        episode_returns.append(sum(curr_rollout_rew))
        episode_lengths.append(step_idx)

    # clean up
    if encoder is not None:
        episode_latent_means = [torch.stack(e) for e in episode_latent_means]
        episode_latent_logvars = [torch.stack(e) for e in episode_latent_logvars]

    if isinstance(state, dict):
        for k in episode_next_obs.keys():
            episode_prev_obs[k] = [torch.cat(e) for e in episode_prev_obs[k]]
            episode_next_obs[k] = [torch.cat(e) for e in episode_next_obs[k]]
    else:
        episode_prev_obs = [torch.cat(e) for e in episode_prev_obs]
        episode_next_obs = [torch.cat(e) for e in episode_next_obs]
    episode_actions = [torch.cat(e) for e in episode_actions]
    episode_rewards = [torch.cat(r) for r in episode_rewards]

    return episode_latent_means, episode_latent_logvars, \
           episode_prev_obs, episode_next_obs, episode_actions, episode_rewards, \
           episode_returns


def plot_latents(latent_means,
                 latent_logvars,
                 image_folder,
                 iter_idx,
                 ):
    """
    Plot mean/variance over time
    """

    num_rollouts = len(latent_means)
    num_episode_steps = len(latent_means[0])

    latent_means = torch.cat(latent_means).cpu().detach().numpy()
    latent_logvars = torch.cat(latent_logvars).cpu().detach().numpy()

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(latent_means.shape[0]), latent_means, '-', alpha=0.5)
    plt.plot(range(latent_means.shape[0]), latent_means.mean(axis=1), 'k-')
    for tj in np.cumsum([0, *[num_episode_steps for _ in range(num_rollouts)]]):
        span = latent_means.max() - latent_means.min()
        plt.plot([tj + 0.5, tj + 0.5],
                 [latent_means.min() - span * 0.05, latent_means.max() + span * 0.05],
                 'k--', alpha=0.5)
    plt.xlabel('env steps', fontsize=15)
    plt.ylabel('latent mean', fontsize=15)

    plt.subplot(1, 2, 2)
    latent_var = np.exp(latent_logvars)
    plt.plot(range(latent_logvars.shape[0]), latent_var, '-', alpha=0.5)
    plt.plot(range(latent_logvars.shape[0]), latent_var.mean(axis=1), 'k-')
    for tj in np.cumsum([0, *[num_episode_steps for _ in range(num_rollouts)]]):
        span = latent_var.max() - latent_var.min()
        plt.plot([tj + 0.5, tj + 0.5],
                 [latent_var.min() - span * 0.05, latent_var.max() + span * 0.05],
                 'k--', alpha=0.5)
    plt.xlabel('env steps', fontsize=15)
    plt.ylabel('latent variance', fontsize=15)

    plt.tight_layout()
    if image_folder is not None:
        plt.savefig('{}/{}_latents'.format(image_folder, iter_idx))
        plt.close()
    else:
        plt.show()


def plot_vae_loss(args,
                  latent_means,
                  latent_logvars,
                  prev_obs,
                  next_obs,
                  actions,
                  rewards,
                  task,
                  image_folder,
                  iter_idx,
                  reward_decoder,
                  state_decoder,
                  task_decoder,
                  compute_task_reconstruction_loss,
                  compute_rew_reconstruction_loss,
                  compute_state_reconstruction_loss,
                  compute_kl_loss
                  ):
    num_rollouts = len(latent_means)
    num_episode_steps = len(latent_means[0])
    if not args.vae.disable_stochasticity_in_latent:
        num_samples = 10  # how many samples to use to get an average/std ELBO loss
    else:
        num_samples = 1

    latent_means = torch.cat(latent_means)
    latent_logvars = torch.cat(latent_logvars)

    if isinstance(prev_obs, dict):
        prev_obs = {k: torch.cat(prev_obs[k]).to(DeviceConfig.DEVICE) 
                    for k in prev_obs.keys()}
        next_obs = {k: torch.cat(next_obs[k]).to(DeviceConfig.DEVICE) 
                    for k in next_obs.keys()}
    else:
        prev_obs = torch.cat(prev_obs).to(DeviceConfig.DEVICE)
        next_obs = torch.cat(next_obs).to(DeviceConfig.DEVICE)
    actions = torch.cat(actions).to(DeviceConfig.DEVICE)
    rewards = torch.cat(rewards).to(DeviceConfig.DEVICE)

    # - we will try to make predictions for each tuple in trajectory, hence we need to expand the targets
    if isinstance(prev_obs, dict):
        prev_obs = {k: prev_obs[k].unsqueeze(0).expand(num_samples, *prev_obs[k].shape).to(DeviceConfig.DEVICE) 
                    for k in prev_obs.keys()}
        next_obs = {k: next_obs[k].unsqueeze(0).expand(num_samples, *next_obs[k].shape).to(DeviceConfig.DEVICE) 
                    for k in next_obs.keys()}
    else:
        prev_obs = prev_obs.unsqueeze(0).expand(num_samples, *prev_obs.shape).to(DeviceConfig.DEVICE)
        next_obs = next_obs.unsqueeze(0).expand(num_samples, *next_obs.shape).to(DeviceConfig.DEVICE)
    actions = actions.unsqueeze(0).expand(num_samples, *actions.shape).to(DeviceConfig.DEVICE)
    rewards = rewards.unsqueeze(0).expand(num_samples, *rewards.shape).to(DeviceConfig.DEVICE)

    # TODO: This is a hacky way to deal with single-process case where
    #       actions and rewards have shape e.g. [10, 400, 1], but the target
    #       we are expecting have shape e.g. [10, 400, 1, 1]. To correct, for
    #       now, we are adding a dimension to actions and rewards to make them
    #       e.g. [10, 400, 1, 1]
    if len(actions.shape) == 3:
        actions = actions.unsqueeze(2)
        print('WARNING: shape mismatch (actions) in plot_vae_loss()')
    if len(rewards.shape) == 3:
        rewards = rewards.unsqueeze(2)
        print('WARNING: shape mismatch (rewards) in plot_vae_loss()')

    rew_reconstr_mean = []
    rew_reconstr_std = []
    rew_pred_std = []

    state_reconstr_mean = []
    state_reconstr_std = []
    state_pred_std = []

    task_reconstr_mean = []
    task_reconstr_std = []
    task_pred_std = []

    # compute the sum of ELBO_t's by looping through (trajectory length + prior)
    for i in range(len(latent_means)):

        curr_latent_mean = latent_means[i]
        curr_latent_logvar = latent_logvars[i]

        # compute the reconstruction loss
        if not args.vae.disable_stochasticity_in_latent:
            # take several samples from the latent distribution
            latent_samples = utl.sample_gaussian(curr_latent_mean.view(-1), curr_latent_logvar.view(-1), num_samples)
        else:
            latent_samples = torch.cat((curr_latent_mean.view(-1), curr_latent_logvar.view(-1))).unsqueeze(0)

        # expand: each latent sample will be used to make predictions for the entire trajectory
        if isinstance(prev_obs, dict):
            # We can just use the first key, since all the keys have the same trajectory length
            len_traj = prev_obs[list(prev_obs.keys())[0]].shape[1]
        else:
            len_traj = prev_obs.shape[1]

        # compute reconstruction losses
        if task_decoder is not None:
            loss_task, task_pred = compute_task_reconstruction_loss(latent_samples, task, return_predictions=True)

            # average/std across the different samples
            task_reconstr_mean.append(loss_task.mean())
            task_reconstr_std.append(loss_task.std())
            task_pred_std.append(task_pred.std())

        latent_samples = latent_samples.unsqueeze(1).expand(num_samples, len_traj, latent_samples.shape[-1])

        if reward_decoder is not None:
            loss_rew, rew_pred = compute_rew_reconstruction_loss(latent_samples, prev_obs, next_obs,
                                                                 actions, rewards, return_predictions=True)
            # sum along length of trajectory
            loss_rew = loss_rew.sum(dim=1)
            rew_pred = rew_pred.sum(dim=1)

            # average/std across the different samples
            rew_reconstr_mean.append(loss_rew.mean())
            rew_reconstr_std.append(loss_rew.std())
            rew_pred_std.append(rew_pred.std())

        if state_decoder is not None:
            loss_state, state_pred = compute_state_reconstruction_loss(latent_samples, prev_obs, next_obs,
                                                                       actions, return_predictions=True)
            # sum along length of trajectory
            loss_state = loss_state.sum(dim=1)
            state_pred = state_pred.sum(dim=1)

            # average/std across the different samples
            state_reconstr_mean.append(loss_state.mean())
            state_reconstr_std.append(loss_state.std())
            state_pred_std.append(state_pred.std())

    # kl term
    vae_kl_term = compute_kl_loss(latent_means, latent_logvars, None)

    # --- plot KL term ---

    x = range(len(vae_kl_term))

    plt.plot(x, vae_kl_term.cpu().detach().numpy(), 'b-')
    vae_kl_term = vae_kl_term.cpu()
    for tj in np.cumsum([0, *[num_episode_steps for _ in range(num_rollouts)]]):
        span = vae_kl_term.max() - vae_kl_term.min()
        plt.plot([tj + 0.5, tj + 0.5],
                 [vae_kl_term.min() - span * 0.05, vae_kl_term.max() + span * 0.05],
                 'k--', alpha=0.5)
    plt.xlabel('env steps', fontsize=15)
    plt.ylabel('KL term', fontsize=15)
    plt.tight_layout()
    if image_folder is not None:
        plt.savefig('{}/{}_kl'.format(image_folder, iter_idx))
        plt.close()
    else:
        plt.show()

    # --- plot rew reconstruction ---

    if reward_decoder is not None:

        rew_reconstr_mean = torch.stack(rew_reconstr_mean).detach().cpu().numpy()
        rew_reconstr_std = torch.stack(rew_reconstr_std).detach().cpu().numpy()
        rew_pred_std = torch.stack(rew_pred_std).detach().cpu().numpy()

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        p = plt.plot(x, rew_reconstr_mean, 'b-')
        plt.gca().fill_between(x,
                               rew_reconstr_mean - rew_reconstr_std,
                               rew_reconstr_mean + rew_reconstr_std,
                               facecolor=p[0].get_color(), alpha=0.1)
        for tj in np.cumsum([0, *[num_episode_steps for _ in range(num_rollouts)]]):
            min_y = (rew_reconstr_mean - rew_reconstr_std).min()
            max_y = (rew_reconstr_mean + rew_reconstr_std).max()
            span = max_y - min_y
            plt.plot([tj + 0.5, tj + 0.5],
                     [min_y - span * 0.05, max_y + span * 0.05],
                     'k--', alpha=0.5)
        plt.xlabel('env steps', fontsize=15)
        plt.ylabel('reward reconstruction error', fontsize=15)

        plt.subplot(1, 2, 2)
        plt.plot(x, rew_pred_std, 'b-')
        for tj in np.cumsum([0, *[num_episode_steps for _ in range(num_rollouts)]]):
            span = rew_pred_std.max() - rew_pred_std.min()
            plt.plot([tj + 0.5, tj + 0.5],
                     [rew_pred_std.min() - span * 0.05, rew_pred_std.max() + span * 0.05],
                     'k--', alpha=0.5)
        plt.xlabel('env steps', fontsize=15)
        plt.ylabel('std of rew reconstruction', fontsize=15)
        plt.tight_layout()
        if image_folder is not None:
            plt.savefig('{}/{}_rew_reconstruction'.format(image_folder, iter_idx))
            plt.close()
        else:
            plt.show()

    # --- plot state reconstruction ---

    if state_decoder is not None:

        plt.figure(figsize=(12, 5))

        state_reconstr_mean = torch.stack(state_reconstr_mean).detach().cpu().numpy()
        state_reconstr_std = torch.stack(state_reconstr_std).detach().cpu().numpy()
        state_pred_std = torch.stack(state_pred_std).detach().cpu().numpy()

        plt.subplot(1, 2, 1)
        p = plt.plot(x, state_reconstr_mean, 'b-')
        plt.gca().fill_between(x,
                               state_reconstr_mean - state_reconstr_std,
                               state_reconstr_mean + state_reconstr_std,
                               facecolor=p[0].get_color(), alpha=0.1)
        for tj in np.cumsum([0, *[num_episode_steps for _ in range(num_rollouts)]]):
            min_y = (state_reconstr_mean - state_reconstr_std).min()
            max_y = (state_reconstr_mean + state_reconstr_std).max()
            span = max_y - min_y
            plt.plot([tj + 0.5, tj + 0.5],
                     [min_y - span * 0.05, max_y + span * 0.05],
                     'k--', alpha=0.5)
        plt.xlabel('env steps', fontsize=15)
        plt.ylabel('state reconstruction error', fontsize=15)

        plt.subplot(1, 2, 2)
        plt.plot(x, state_pred_std, 'b-')
        for tj in np.cumsum([0, *[num_episode_steps for _ in range(num_rollouts)]]):
            span = state_pred_std.max() - state_pred_std.min()
            plt.plot([tj + 0.5, tj + 0.5],
                     [state_pred_std.min() - span * 0.05, state_pred_std.max() + span * 0.05],
                     'k--', alpha=0.5)
        plt.xlabel('env steps', fontsize=15)
        plt.ylabel('std of state reconstruction', fontsize=15)
        plt.tight_layout()
        if image_folder is not None:
            plt.savefig('{}/{}_state_reconstruction'.format(image_folder, iter_idx))
            plt.close()
        else:
            plt.show()

    # --- plot task reconstruction ---

    if task_decoder is not None:

        plt.figure(figsize=(12, 5))

        task_reconstr_mean = torch.stack(task_reconstr_mean).detach().cpu().numpy()
        task_reconstr_std = torch.stack(task_reconstr_std).detach().cpu().numpy()
        task_pred_std = torch.stack(task_pred_std).detach().cpu().numpy()

        plt.subplot(1, 2, 1)
        p = plt.plot(x, task_reconstr_mean, 'b-')
        plt.gca().fill_between(x,
                               task_reconstr_mean - task_reconstr_std,
                               task_reconstr_mean + task_reconstr_std,
                               facecolor=p[0].get_color(), alpha=0.1)
        for tj in np.cumsum([0, *[num_episode_steps for _ in range(num_rollouts)]]):
            min_y = (task_reconstr_mean - task_reconstr_std).min()
            max_y = (task_reconstr_mean + task_reconstr_std).max()
            span = max_y - min_y
            plt.plot([tj + 0.5, tj + 0.5],
                     [min_y - span * 0.05, max_y + span * 0.05],
                     'k--', alpha=0.5)
        plt.xlabel('env steps', fontsize=15)
        plt.ylabel('task reconstruction error', fontsize=15)

        plt.subplot(1, 2, 2)
        plt.plot(x, task_pred_std, 'b-')
        for tj in np.cumsum([0, *[num_episode_steps for _ in range(num_rollouts)]]):
            span = task_pred_std.max() - task_pred_std.min()
            plt.plot([tj + 0.5, tj + 0.5],
                     [task_pred_std.min() - span * 0.05, task_pred_std.max() + span * 0.05],
                     'k--', alpha=0.5)
        plt.xlabel('env steps', fontsize=15)
        plt.ylabel('std of task reconstruction', fontsize=15)
        plt.tight_layout()
        if image_folder is not None:
            plt.savefig('{}/{}_task_reconstruction'.format(image_folder, iter_idx))
            plt.close()
        else:
            plt.show()
