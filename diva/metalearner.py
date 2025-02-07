"""
Meta-learner based on VariBAD code; modified for DIVA and UED methods. 
"""
import gc  # noqa: I001
import os
import threading
import time
from collections import defaultdict

import gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger as logur
from scipy.stats import entropy
from tqdm import tqdm
matplotlib.use('Agg') 

from diva.components.exploration.exploration_bonus import ExplorationBonus
from diva.components.policy.networks import Policy
from diva.components.qd.qd_module import QDModule
from diva.components.vae.vae import VaribadVAE
from diva.environments import utils as utl
from diva.environments.utils import EnvLatents, EnvSalients
from diva.utils import evaluation as utl_eval
from diva.utils.constructors import (
    construct_online_storage,
    construct_plr_scheduler,
    construct_policy,
    get_policy_input_info,
)
from diva.utils.loggers import TBLogger, WandBLogger
from diva.utils.torch import DeviceConfig


class MetaLearner:
    """
    Meta-Learner class with the main training loop for VariBAD.
    """
    def __init__(self, args, envs, plr_components, archive_dims=None):
        self.args = args
        self.envs = envs
        self.plr_level_sampler, self.plr_level_store = plr_components
        self.archive_dims = archive_dims
        utl.seed(self.args.seed, self.args.deterministic_execution)

        # Calculate number of updates and keep count of frames/iterations
        self.num_updates = int(args.domain.num_frames) // args.policy.num_steps // args.policy.num_processes
        self.frames = 0
        self.iter_idx = -1
        logur.info(f'policy_num_steps: {args.policy.num_steps}')
        logur.info(f'num_processes: {args.policy.num_processes}')
        
        # Initialise logger
        logger_class = WandBLogger if self.args.use_wandb else TBLogger
        self.logger = logger_class(self.args, self.args.wandb_label)

        if self.args.dist.use_plr:
            self.plr_temperature_scheduler = construct_plr_scheduler(args, self.num_updates)
        
        self.plr_lock = threading.Lock() if self.args.dist.use_plr else None

        assert not self.args.single_task_mode, 'Single task mode not supported'
        self.train_tasks = None

        # Calculate what the maximum length of the trajectories is
        self.args.max_trajectory_len = self.envs._max_episode_steps
        self.args.max_trajectory_len *= self.args.domain.episodes_per_trial

        # Get policy input dimensions
        (self.online_storage_class, 
         self.state_feature_extractor, 
         self.state_dtype) = get_policy_input_info(self.args, self.envs)
        self.args.task_dim = self.envs.task_dim
        self.args.belief_dim = self.envs.belief_dim
        self.args.num_states = self.envs.num_states

        # Get policy output (action) dimensions
        self.action_space = self.envs.action_space
        if isinstance(self.envs.action_space, gym.spaces.discrete.Discrete):
            self.args.action_dim = 1
        else:
            self.args.action_dim = self.envs.action_space.shape[0]

        # Initialise VAE 
        self.vae = VaribadVAE(
            self.args, 
            self.logger, 
            lambda: self.iter_idx,
            self.state_feature_extractor,
            self.state_dtype,)
        
        # Initialise reward bonus
        self.intrinsic_reward = None
        if self.args.explore and self.args.add_exploration_bonus:
            self.intrinsic_reward = ExplorationBonus(
                args=self.args,
                logger=self.logger,
                dim_state=self.args.state_dim,
                encoder=self.vae.encoder,
                rollout_storage=self.vae.rollout_storage)
            
        # Initialise policy
        self.policy = construct_policy(
            Policy,
            self.args,
            self.envs.action_space,
            self.state_feature_extractor,
            self.logger,
            DeviceConfig.DEVICE,
            self.num_updates,
            self.vae.optimiser_vae)
        self.policy_storage = self.initialise_policy_storage()

        # Initialise QD components
        self.qd_module = None
        if (self.args.dist.use_qd):
            self.qd_module = QDModule(self.args, self)

    def initialise_policy_storage(self, num_steps=None, num_processes=None):
        """
        We provide the option to pass in the number of steps and processes
        because QD/PLR/ACCEL simulations may require different arguments. 
        """
        return construct_online_storage(
            self.online_storage_class, 
            self.args, 
            self.policy.actor_critic, 
            num_steps if num_steps is not None else self.args.policy.num_steps, 
            num_processes if num_processes is not None else self.args.policy.num_processes,
            self.action_space,
            self.intrinsic_reward)

    ###########################################################################
    #                              PRE-TRAINING                               #
    ###########################################################################
    
    def kickstart_training(self):
        """ Prepare for training 
        
        Returns:
            skip_training (bool): whether to skip training and close envs
        """
        self.start_time = time.time()
        logur.info('Kickstarting training...')
        self.level_renderings_recent = dict()
        self.genotype_counts_all = defaultdict(lambda: 0)
        # QD warm start
        if self.args.dist.use_qd:
            self.qd_module.do_warm_start()
        # Reset environments
        self.env_salients = EnvSalients()
        self.plr_iter_idx = 0
        logur.info('Initial environment resets...')
        self._reset_done_environments(first_reset=True)
        # End if only doing warm start
        if self.args.dist.qd and self.args.dist.qd.warm_start_only:
            logur.info('Warm start only, and warm start complete!')
            logur.info('Number of genotypes in archive: ', self.qd_module.archive._stats.num_elites)
            logur.info('Percentage of archive filled: ', self.qd_module.archive._stats.num_elites / self.qd_module.archive._cells)
            self.envs.close()
            return True  # skip_training=True
        if self.args.dist.qd and self.args.dist.qd.no_sim_objective and hasattr(self.qd_module, 'qd_envs'):
            logur.info("Closing QD envs...")
            try: 
                self.qd_module.qd_envs.close() 
            except Exception as _: 
                pass
        # Store initial genotypes and counts
        init_genotypes = None
        init_genotypes_processed = None
        if self.args.domain.has_genotype:
            init_genotypes = self.envs.genotype
            init_genotypes_processed = [
                QDModule.process_genotype_static(
                    g, gt_is_continuous=self.args.domain.gt_is_continuous) 
                for g in init_genotypes]
            for i in range(self.args.policy.num_processes):
                self.genotype_counts_all[init_genotypes_processed[i]] += 1
        # Store initial level renderings
        if self.args.domain.eval_save_video and self.args.domain.has_genotype:
            init_level_renderings = self.envs.level_rendering
            for i in range(self.args.policy.num_processes):
                self.level_renderings_recent[init_genotypes_processed[i]] = \
                    init_level_renderings[i]
        # Insert initial observation to rollout storage
        self.policy_storage.add_initial_state(self.env_salients.states)
        # Log once before training
        with torch.no_grad():
            self.eval_and_log(first_log=True)
        self.intrinsic_reward_is_pretrained = False
        return False  # skip_training=False

    ###########################################################################
    #                           MAIN TRAINING LOOP                            #
    ###########################################################################

    def train(self):
        """ Main Meta-Training loop """
        # Kickstart training
        skip_training = self.kickstart_training()
        if skip_training:
            return self.close()
    
        # Training loop
        pbar = tqdm(range(self.num_updates))
        start_time = time.time()
        for self.iter_idx in pbar:
            rate = self.frames / (time.time() - start_time)
            pbar.set_description(f"fr: {self.frames} ({rate:.2f} f/s)")

            #############          PREPARE FOR ROLLOUT            #############

            # First, re-compute the hidden states given the current rollouts, 
            # since we just updated the VAE, and the old values are stale
            with torch.no_grad():
                self.reencode_current_trajectory()
            # Make sure we emptied buffers
            assert len(self.policy_storage.latent_mean) == 0  
            self.policy_storage.insert_initial_latents(self.env_latents)
            # Stats logging (timing and environment metrics)
            self.env_step_info_stats = defaultdict(list)

            #############      PERFORM AND PROCESS ROLLOUT        #############

            # Rollout policies for a few steps
            for step_idx in range(self.args.policy.num_steps):
                done_indices = self._train_policy_step(step_idx)
            
            # Process step infos and log
            for k, v in self.env_step_info_stats.items():
                agg_fn = sum if 'time/' in k else np.mean  # only sum the times
                self.logger.add(k, agg_fn(v), self.iter_idx)

            # Log episodic rewards for completed environments
            if len(done_indices) > 0:
                episodic_rewards_raw = self.policy_storage.rewards_raw.sum(axis=0)[done_indices].cpu().numpy().mean()
                episodic_rewards_normalised = self.policy_storage.rewards_normalised.sum(axis=0)[done_indices].cpu().numpy().mean()
                self.logger.add('train/episodic_rewards_raw', episodic_rewards_raw, self.iter_idx)
                self.logger.add('train/episodic_rewards_normalised', episodic_rewards_normalised, self.iter_idx)

            # Check if we still need to fill the VAE buffer more
            if (len(self.vae.rollout_storage) == 0 and not self.args.vae.buffer_size == 0) or \
                    (self.args.vae.precollect_len > self.frames):
                logur.debug('NOT UPDATING yet; still filling the VAE buffer.')
                self.policy_storage.after_update()
                continue

            #############            LEARNING UPDATES             #############

            if self.args.add_exploration_bonus and not self.intrinsic_reward_is_pretrained:
                self._rnd_pretrain_step()  # pretrain bonus scale
            else:
                self._standard_update_step()  # all updates (meta-RL, VAE, etc.)

            self.policy_storage.after_update()  # clean up

        logur.info('The end! Closing envs...')
        self.close()

    def _train_policy_step(self, step_idx):
        st, es = time.time(), self.env_salients
        # Sample actions from policy
        with torch.no_grad():
            _values, _actions = utl.select_action(
                args=self.args, policy=self.policy, state=es.states,
                belief=es.beliefs, task=es.tasks, deterministic=False,
                env_latents=self.env_latents)
        # Take step in the environment
        [_states, _beliefs, _tasks], rewards, done, infos = utl.env_step(
            self.envs, _actions, self.args, get_belief=False, get_task=False)
        es.store_env_step(_values, _actions, _states, _beliefs, _tasks)
        # Process outputs of step
        self.process_infos_for_logging(infos)
        rew_raw, rew_normalised = (rewards) if len(rewards) == 2 else (rewards, rewards)
        done = torch.tensor(done, dtype=torch.float32, device=DeviceConfig.DEVICE).view(-1, 1)
        (masks_done, bad_masks, clha_masks) = self.compute_step_masks(done, infos)
        # Compute next embedding (for next loop and/or value prediction bootstrap)
        with torch.no_grad():
            self.env_latents = utl.update_encoding(encoder=self.vae.encoder,
                next_obs=es.states, action=es.actions, reward=rew_raw,
                done=done, env_latents=self.env_latents)
        # Add data to vae buffer (prev_state might include useful task info)
        if not (self.args.vae.disable_decoder and self.args.vae.disable_kl_term):
            self.vae.rollout_storage.insert(
                utl.clone_state(es.prev_states),
                es.actions.detach().clone(), 
                utl.clone_state(es.states),
                rew_raw.clone(), 
                done.clone(),
                es.tasks if es.tasks is not None else None  # Should be list
            )
        # Add new observation to intrinsic reward
        if self.args.add_exploration_bonus:
            (z_means, z_logvars) = (self.env_latents.z_means, self.env_latents.z_logvars)
            z_beliefs = torch.cat((z_means, z_logvars), dim=-1)
            self.intrinsic_reward.add(es.states, z_beliefs, es.actions.detach())
        # Reset environments that are done
        self.policy_storage.add_next_state_before_reset(step_idx, utl.clone_state(es.states))
        done_indices = np.argwhere(done.cpu().flatten()).flatten()
        self._reset_done_environments(done_indices=done_indices)
        # Add experience to policy buffer
        self.policy_storage.insert(
            state=es.states, 
            belief=es.beliefs,
            task=es.tasks, 
            actions=es.actions,
            rewards_raw=rew_raw,
            rewards_normalised=rew_normalised, 
            value_preds=es.values,
            masks=masks_done, 
            bad_masks=bad_masks, 
            cliffhanger_masks=clha_masks, 
            done=done,
            hidden_states=self.env_latents.hs.squeeze(0),
            latent_sample=self.env_latents.z_samples, 
            latent_mean=self.env_latents.z_means,
            latent_logvar=self.env_latents.z_logvars,
            level_seeds=es.level_seeds)
        self.frames += self.args.policy.num_processes
        # Log timing states for policy rollouts
        et = time.time(); self.logger.add('time/rollout_policies', et-st, self.iter_idx)  # noqa: E702
        return done_indices

    ###########################################################################
    #                                UPDATES                                  #
    ###########################################################################

    def _standard_update_step(self):
        st = time.time()
        # Check if we'll be doing any update steps at all
        if self.args.vae.precollect_len <= self.frames:
            # Check if we are pre-training the VAE
            if self.args.vae.pretrain_len > self.iter_idx:
                for p in range(self.args.vae.num_updates_per_pretrain):
                    self._vae_pretrain_update(p)
            # Otherwise do the normal update (policy + vae)
            else:
                self._train_stats = self.update()
                with torch.no_grad():
                    self.eval_and_log(first_log=False)
        et = time.time()
        self.logger.add('time/meta_rl_update', et-st, self.iter_idx)

        # INTRINSIC REWARD MODEL UPDATE
        if self.args.add_exploration_bonus:
            if self.iter_idx % self.args.rnd_update_frequency == 0:
                self.intrinsic_reward.update(self.frames, self.iter_idx)

        # PLR UPDATE
        if self.plr_level_sampler and (not self.args.dist.use_qd or self.args.dist.qd.use_plr_for_training):
            # NOTE: we do not perform plr update for QD unless we using it for 
            # sampling levels during training (instead of sampling from the archive)
            self._plr_update()

        # QD EVAL AND ARCHIVE UPDATE
        st = time.time()
        if (self.args.dist.use_qd and
            (self.iter_idx + 1) % self.args.dist.qd.update_interval == 0 and
                not self.args.dist.qd.no_sim_objective):
            for _ in range(self.args.dist.qd.updates_per_iter):
                self.qd_module.update()
                # Update iteration index
                self.qd_module.iter_idx += 1
        et = time.time()
        self.logger.add('time/qd_update', et-st, self.iter_idx)

        if self.plr_level_sampler:
            self.plr_level_sampler.temperature = self.plr_temperature_scheduler()
            self.logger.add('plr/temperature', self.plr_level_sampler.temperature, self.iter_idx)

        # Increment steps since last updated for all archive solutions
        if self.args.dist.use_qd:
            self.qd_module.archive.increment_sslu()
            self.logger.add('qd/max_sslu', self.qd_module.archive.get_max_sslu(), self.iter_idx)


    def update(self):
        """
        Meta-update.
        Here the policy is updated for good average performance across tasks.
        """
        # Update policy (if we are not pre-training, have enough data in the 
        # vae buffer, and are not at iteration 0)
        if self.iter_idx >= self.args.vae.pretrain_len and self.iter_idx > 0:

            # Bootstrap next value prediction
            with torch.no_grad():
                next_value = self.get_value()

            # Compute returns for current rollouts
            self.policy_storage.compute_returns(next_value, vae=self.vae)

            # Update agent (this will also call the VAE update!)
            policy_train_stats = self.policy.update(
                policy_storage=self.policy_storage,
                encoder=self.vae.encoder,
                rlloss_through_encoder=self.args.vae.rlloss_through_encoder,
                compute_vae_loss=self.vae.compute_vae_loss)
        else:
            policy_train_stats = 0, 0, 0, 0
            # Pre-train the VAE
            if self.iter_idx < self.args.vae.pretrain_len:
                self.vae.compute_vae_loss(update=True)

        return policy_train_stats

    def _rnd_pretrain_step(self):
        # compute returns once - this will normalise the RND inputs!
        next_value = self.get_value()
        self.policy_storage.compute_returns(next_value, vae=self.vae)
        self.intrinsic_reward.update(  # (calling with max num of frames to init all networks)
            self.args.domain.num_frames, self.iter_idx, log=False)  
        self.intrinsic_reward_is_pretrained = True

    def _plr_update(self):
        """ Update PLR weights """
        logur.debug('Updating PLR weights...')
        st = time.time()
        # NOTE: self.policy_storage.compute_returns already called in
        # update() above, so policy_storage has correct returns for PLR

        # Update level sampler (NOTE: currently we always update)
        if self.plr_level_sampler and self.args.vae.precollect_len <= self.frames:
            # NOTE: We can't update level sampler until we've updated the 
            # agent, since before_update needs to be called to set
            # storage.action_log_dist, which is used by the following method
            if not self.args.dist.use_qd:
                # We do not updated level sampler with online experience if
                # we are using QD.
                self.plr_level_sampler.update_with_rollouts(self.policy_storage)
            if self.plr_iter_idx > 5:
                self.plr_log()
            self.plr_iter_idx += 1

        # Clean up after update
        self.policy_storage.after_update()
        if self.plr_level_sampler and not self.args.dist.use_qd:
            self.plr_level_sampler.after_update()
        
        et = time.time()
        self.logger.add('time/plr_update', et-st, self.iter_idx)

    def _vae_pretrain_update(self, p):
        self.vae.compute_vae_loss(
            update=True, 
            pretrain_index=self.iter_idx * 
                           self.args.vae.num_updates_per_pretrain + p)

    ###########################################################################
    #                             TRAINING UTILS                              #
    ###########################################################################

    def process_infos_for_logging(self, infos):
        # NOTE: Only log values for the first environment for efficiency
        for k, v in infos[0].items():
            if 'time/' in k: self.env_step_info_stats[k].append(v)  # noqa: E701
            if 'env/' in k:  self.env_step_info_stats['train-' + k].append(v)  # noqa: E701

    def maybe_sample_tasks(self, done_indices):
        if self.args.dist.use_qd and not self.args.dist.qd.use_plr_for_training:
            logur.debug('Using QD to sample new genotypes for training!')
            # If using QD, sample new genotypes from archive
            list_of_arrays = self.qd_module.sample_from_archive(len(done_indices))
            return list_of_arrays
        return None
    
    def save_level_renderings(self, genotypes, done_indices):
        """
        Store level renderings for done environments (so we only do it once).
        """
        if not self.args.domain.has_genotype:
            return
        pg = QDModule.process_genotype_static
        gic = self.args.domain.gt_is_continuous
        keys = [pg(genotypes[i], gt_is_continuous=gic) for i in done_indices]

        # Store genotypes
        for i in done_indices:
            self.genotype_counts_all[keys[i]] += 1
        # Store level renderings
        level_renderings = [None for _ in range(self.args.policy.num_processes)]
        if self.args.domain.eval_save_video:
            level_renderings = self.envs.level_rendering
        for i in done_indices:
            self.level_renderings_recent[keys[i]] = level_renderings[i]
    
    def compute_step_masks(self, done, infos):
        masks_done = torch.tensor(  # mask for trial ends
            [[0.0] if done_ else [1.0] for done_ in done], 
            dtype=torch.float32, device=DeviceConfig.DEVICE)
        bad_masks = torch.tensor(  # trial ended because time limit was reached
            [[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos], 
            dtype=torch.float32, device=DeviceConfig.DEVICE)
        cliffhanger_masks = torch.tensor(  
            [[0.0] if 'cliffhanger' in info.keys() else [1.0] for info in infos],
            dtype=torch.float32, device=DeviceConfig.DEVICE)
        return masks_done, bad_masks, cliffhanger_masks  

    def _reset_done_environments(self, first_reset=False, done_indices=None):
        """
        Check if we need to reset environmets based on done_indices and
        if so, reset the environments.

        Updates the following object attributes:
        - self.env_salients - the observable information for the environments
        - self.env_priors - the priors for the environments (for partial beliefs)
        """
        logur.debug('Resetting done environments...')
        del first_reset  # Don't use at the moment
        if done_indices is None: # Reset all
            done_indices = np.arange(self.args.policy.num_processes) # All
        elif len(done_indices) == 0:  # None to reset
            return
        # Reset environments that are done
        tasks = self.maybe_sample_tasks(done_indices)  # only is not None for QD archive
        states, beliefs, tasks, level_seeds = utl.reset_env(
            self.envs, self.args, indices=done_indices,
            state=self.env_salients.states, task=tasks)
        if self.args.dist.use_plr:
            level_seeds = level_seeds.unsqueeze(-1)
        if self.args.domain.has_genotype:
            genotypes = self.envs.genotype
            self.save_level_renderings(genotypes, done_indices)
        self.env_salients = EnvSalients(states, beliefs, tasks, level_seeds)
        return

    def reencode_current_trajectory(self):
        """
        (Re-)Encodes (for each process) the entire current trajectory.
        Returns sample/mean/logvar and hidden state (if applicable) for the 
        current timestep.
        """
        # Get current batch for each process (zero-padded s/a/r + len indicators)
        _batch = self.vae.rollout_storage.get_running_batch()
        _, obs, act, rew, lens = _batch  # no need for prev_obs
        # Get embedding [H+1, B, input_size] -- includes the prior (+1)!
        (h, return_prior) = (None, True)
        traj_z_samples, traj_z_means, traj_z_logvars, traj_hs = \
            self.vae.encoder(act, obs, rew, h, return_prior)
        # Get the embedding / hidden state of the current time step 
        # (we need lens[i]-1 since they are zero-padded)
        def get_current_timestep(traj_val):
            return torch.stack([traj_val[lens[i]-1][i] for i in range(len(lens))])
        z_samples = get_current_timestep(traj_z_samples)  #.to(DeviceConfig.DEVICE)
        z_means   = get_current_timestep(traj_z_means)  #.to(DeviceConfig.DEVICE)
        z_logvars = get_current_timestep(traj_z_logvars)  #.to(DeviceConfig.DEVICE)
        hs        = get_current_timestep(traj_hs)  #.to(DeviceConfig.DEVICE)
        self.env_latents = EnvLatents(z_samples, z_means, z_logvars, hs)

    def get_value(self, env_salients=None, env_latents=None):
        """
        If env_salients or env_latents are None, the self.env_* value is used.
        """
        es = self.env_salients if env_salients is None else env_salients
        el = self.env_latents if env_latents is None else env_latents
        zs = utl.get_zs_for_policy(self.args, el)  # Get latent
        return self.policy.actor_critic.get_value(state=es.states, 
            belief=es.beliefs, task=es.tasks, latent=zs).detach()
    
    ###########################################################################
    #                                LOGGING                                  #
    ###########################################################################
    
    @property
    def _eval_this_iter(self):
        return (self.iter_idx + 1) % self.args.domain.eval_interval == 0

    @property
    def _save_this_iter(self):
        return (self.iter_idx + 1) % self.args.domain.save_interval == 0

    @property
    def _log_this_iter(self):
        return (self.iter_idx + 1) % self.args.domain.log_interval == 0

    def plr_log(self):
        """ Logging for PLR. """
        # For readability
        la, ii = self.logger.add, self.iter_idx
        try: 
            sample_weights = self.plr_level_sampler.sample_weights()
        except FloatingPointError: 
            sample_weights = None
            logur.debug('Caught floating point error when computing sample weights.')
        except ZeroDivisionError:
            sample_weights = None
            logur.debug('Caught zero division error when computing sample weights.')
        if sample_weights is not None:
            # Log num non-zero sample weights
            perc_nonzero = np.count_nonzero(sample_weights) / len(sample_weights)
            la('plr/perc_nonzero_sample_weights', perc_nonzero, ii)
            # Log entropy of non-zero portion of sample weights
            la('plr/entropy_nonzero_sample_weights', entropy(sample_weights[sample_weights != 0]), ii)
            la('plr/num_seeds', len(sample_weights), ii)

    def eval_and_log(self, first_log=False):
        """ Logging. """
        if first_log:
            run_stats = None
            train_stats = None
        else:
            run_stats = [self.env_salients.actions, 
                         self.policy_storage.action_log_probs, 
                         self.env_salients.values]
            train_stats = self._train_stats

        # For readability
        la, ii, fr = self.logger.add, self.iter_idx, self.frames
        del fr  # Not used

        log_video_this_iter = (
            (self.iter_idx + 1) % self.args.domain.vis_interval == 0 
             and self.args.domain.eval_save_video and self.iter_idx + 1 > 0)

        # --- Evaluate policy ----
        if (self._eval_this_iter and not self.args.skip_eval):
            ret_rms = (self.envs.venv.ret_rms if self.args.policy.norm_rew else None)
            logur.info('Evaluating policy...')
            utl_eval.evaluate(
                args=self.args, 
                policy=self.policy, 
                ret_rms=ret_rms, 
                encoder=self.vae.encoder, 
                iter_idx=self.iter_idx, 
                tasks=self.train_tasks, 
                create_video=log_video_this_iter,
                vae=self.vae, 
                intrinsic_reward=self.intrinsic_reward,
                logger=self.logger
            )
            logur.info('Evaluation complete.')
            # Log level renderings from training
            images_to_stitch = []
            if self.args.domain.eval_save_video: 
                images_to_stitch += list(self.level_renderings_recent.values())[:5*5]
            if len(images_to_stitch) > 0:
                image = utl.stitch_images(images_to_stitch, n=5)
                image_name = 'level_rendering.png'
                self.logger.add_image(f'train_levels/{image_name}', image, ii+1)
            # Clear renderings
            self.level_renderings_recent = dict()
            plt.close('all')
            gc.collect()

        # --- Save models ---
        if self._save_this_iter:
            save_path = os.path.join(self.logger.full_output_folder, 'models')
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            idx_labels = ['']
            if self.args.domain.save_intermediate_models:
                idx_labels.append(int(self.iter_idx))

        # --- Log some other things ---
        if (self._log_this_iter and (train_stats is not None)):
            # Log number of genotypes encountered total
            la('train/num_genotypes_total', len(self.genotype_counts_all), ii)
            # Log environment stats
            if isinstance(self.policy_storage.prev_state, dict):
                for k in self.policy_storage.prev_state.keys():
                    la(f'environment/state_max_{k}', self.policy_storage.prev_state[k].max(), ii)
                    la(f'environment/state_min_{k}', self.policy_storage.prev_state[k].min(), ii)
            else:
                la('environment/state_max', self.policy_storage.prev_state.max(), ii)
                la('environment/state_min', self.policy_storage.prev_state.min(), ii)
            la('environment/rew_max', self.policy_storage.rewards_raw.max(), ii)
            la('environment/rew_min', self.policy_storage.rewards_raw.min(), ii)
            # Log policy stats
            la('policy_losses/value_loss', train_stats[0], ii)
            la('policy_losses/action_loss', train_stats[1], ii)
            la('policy_losses/dist_entropy', train_stats[2], ii)
            la('policy_losses/sum', train_stats[3], ii)
            # la('policy/action', run_stats[0][0].float().mean(), ii)
            if hasattr(self.policy.actor_critic, 'logstd'):
                action_logstd = self.policy.actor_critic.dist.logstd.mean()
                la('policy/action_logstd', action_logstd, ii)
            la('policy/action_logprob', run_stats[1].mean(), ii)
            # la('policy/value', run_stats[2].mean(), ii)
            # Log encoder stats
            la('encoder/latent_mean', torch.cat(self.policy_storage.latent_mean).mean(), ii)
            la('encoder/latent_logvar', torch.cat(self.policy_storage.latent_logvar).mean(), ii)
            # Log average weights and gradients of applicable models 
            for [model, name] in [
                    [self.policy.actor_critic, 'policy'],
                    [self.vae.encoder, 'encoder'],
                    [self.vae.reward_decoder, 'reward_decoder'],
                    [self.vae.state_decoder, 'state_transition_decoder'],
                    [self.vae.task_decoder, 'task_decoder']]:
                if model is None:
                    continue
                # Log the mean weights of the model
                param_list = list(model.parameters())
                param_mean = np.mean([param_list[i].data.cpu().numpy().mean() for i in range(len(param_list))])
                la('weights/mean_{}'.format(name), param_mean, ii)
                param_abs_mean = np.mean([np.abs(param_list[i].data.cpu().numpy()).mean() for i in range(len(param_list))])
                la('weights/mean-abs_{}'.format(name), param_abs_mean, ii)
                # If policy, we also log standard deviation
                if name == 'policy':
                    la('weights/policy_std', param_list[0].data.mean(), ii)
                # Log mean gradient information
                if param_list[0].grad is None: 
                    continue
                grad_list = [param_list[i].grad.cpu().numpy() for i in range(len(param_list))]
                try: 
                    # In the past, we have experienced underflows when we take gradient means
                    param_grad_mean = np.mean([grad_list[i].mean() for i in range(len(grad_list))])
                    la('gradients/mean_{}'.format(name), param_grad_mean, ii)
                    param_grad_abs_mean = np.mean([np.abs(grad_list[i]).mean() for i in range(len(grad_list))])
                    la('gradients/mean-abs_{}'.format(name), param_grad_abs_mean, ii)
                    param_grad_abs_max = np.max([np.abs(grad_list[i]).max() for i in range(len(grad_list))])
                    la('gradients/max-abs_{}'.format(name), param_grad_abs_max, ii)
                except FloatingPointError:
                    logur.info('skipping gradient logging due to underflow')

    def close(self):
        self.envs.close()
        self.logger.close()