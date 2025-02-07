# Modified from https://github.com/lmzintgraf/varibad/tree/master
import warnings

import gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

# import torch
from torch.nn import functional as F
from torch.optim import Adam

from diva.components.decoder import (
    RewardDecoder,
    StateTransitionDecoder,
    TaskDecoder,
)
from diva.components.encoder import RNNEncoder
from diva.components.vae.storage import DictRolloutStorageVAE, RolloutStorageVAE
from diva.environments.utils import get_num_tasks, get_task_dim
from diva.utils.torch import DeviceConfig


class VaribadVAE:
    """
    VAE of VariBAD:
    - has an encoder and decoder
    - can compute the ELBO loss
    - can update the VAE (encoder+decoder)
    """

    def __init__(
            self, 
            args, 
            logger, 
            get_iter_idx,
            state_feature_extractor,
            state_dtype):

        self.args = args
        self.logger = logger
        self.get_iter_idx = get_iter_idx
        self.state_feature_extractor = state_feature_extractor
        self.state_dtype = state_dtype
        self.task_dim = (get_task_dim(self.args) if self.args.vae.decode_task 
                         else None)
        self.num_tasks = (get_num_tasks(self.args) if self.args.vae.decode_task 
                          else None)

        # initialise the encoder
        self.encoder = self.initialise_encoder()

        # initialise the decoders (returns None for unused decoders)
        self.state_decoder, self.reward_decoder, self.task_decoder = \
            self.initialise_decoder()

        # initialise rollout storage for the VAE update
        # (this differs from the data that the on-policy RL algorithm uses)
        if isinstance(self.args.state_dim, dict):
            RSVAE = DictRolloutStorageVAE
        else:
            RSVAE = RolloutStorageVAE
        self.rollout_storage = RSVAE(
            num_processes=self.args.policy.num_processes,
            max_trajectory_len=self.args.max_trajectory_len,
            zero_pad=True,
            max_num_rollouts=self.args.vae.buffer_size,
            state_dim=self.args.state_dim,
            action_dim=self.args.action_dim,
            vae_buffer_add_thresh=self.args.vae.buffer_add_thresh,
            task_dim=self.task_dim,
            state_dtype=self.state_dtype,
            logger=self.logger,
        )

        # initalise optimiser for the encoder and decoders
        decoder_params = []
        if not self.args.vae.disable_decoder:
            if self.args.vae.decode_reward:
                decoder_params.extend(self.reward_decoder.parameters())
            if self.args.vae.decode_state:
                decoder_params.extend(self.state_decoder.parameters())
            if self.args.vae.decode_task:
                decoder_params.extend(self.task_decoder.parameters())
        self.optimiser_vae = Adam(
            [*self.encoder.parameters(), *decoder_params], lr=self.args.vae.lr)
        
        # Create environment so we can get task information
        self.dummy_env = gym.make(self.args.domain.env_name) 
        if self.args.vae.decode_task or self.args.vae.multihead_for_reward:
            assert 'gridworld' in self.args.domain.domain, \
                "The dummy_env for task decoding only works for GridWorld at the moment."
            self.task_to_id = self.dummy_env.task_to_id_fn
        
        self.update_count = 0

    ###########################################################################
    #                            INITIALIZATION                               #
    ###########################################################################

    def initialise_encoder(self):
        """ Initialises and returns an RNN encoder """
        encoder = RNNEncoder(
            args=self.args,
            layers_before_gru=self.args.vae.encoder_layers_before_gru,
            hidden_size=self.args.vae.encoder_gru_hidden_size,
            layers_after_gru=self.args.vae.encoder_layers_after_gru,
            latent_dim=self.args.vae.latent_dim,
            action_dim=self.args.action_dim,
            action_embed_dim=self.args.vae.action_embedding_size,
            state_dim=self.args.state_dim,
            state_embed_dim=self.args.vae.state_embedding_size,
            reward_size=1,
            reward_embed_size=self.args.vae.reward_embedding_size,
            state_feature_extractor=self.state_feature_extractor,
            state_is_image=self.args.domain.state_is_image,
        ).to(DeviceConfig.DEVICE)
        return encoder

    def initialise_decoder(self):
        """
        Initialises and returns the (state/reward/task) decoder as specified 
        in self.args
        """

        if self.args.vae.disable_decoder:
            return None, None, None

        latent_dim = self.args.vae.latent_dim
        # if we don't sample embeddings for the decoder, we feed in 
        # mean & variance
        if self.args.vae.disable_stochasticity_in_latent:
            latent_dim *= 2

        # initialise state decoder for VAE
        if self.args.vae.decode_state:
            state_decoder = StateTransitionDecoder(
                args=self.args,
                layers=self.args.vae.state_decoder_layers,
                latent_dim=latent_dim,
                action_dim=self.args.action_dim,
                action_embed_dim=self.args.vae.action_embedding_size,
                state_dim=self.args.state_dim,
                state_embed_dim=self.args.vae.state_embedding_size,
                pred_type=self.args.vae.state_pred_type,
                state_feature_extractor=self.state_feature_extractor
            ).to(DeviceConfig.DEVICE)
        else:
            state_decoder = None

        # initialise reward decoder for VAE
        if self.args.vae.decode_reward:
            reward_decoder = RewardDecoder(
                args=self.args,
                layers=self.args.vae.reward_decoder_layers,
                latent_dim=latent_dim,
                state_dim=self.args.state_dim,
                state_embed_dim=self.args.vae.state_embedding_size,
                action_dim=self.args.action_dim,
                action_embed_dim=self.args.vae.action_embedding_size,
                num_states=self.args.num_states,
                multi_head=self.args.vae.multihead_for_reward,
                pred_type=self.args.vae.rew_pred_type,
                input_prev_state=self.args.vae.input_prev_state,
                input_action=self.args.vae.input_action,
                state_feature_extractor=self.state_feature_extractor
            ).to(DeviceConfig.DEVICE)
        else:
            reward_decoder = None

        # initialise task decoder for VAE
        if self.args.vae.decode_task:
            assert self.task_dim != 0
            task_decoder = TaskDecoder(
                latent_dim=latent_dim,
                layers=self.args.vae.task_decoder_layers,
                task_dim=self.task_dim,
                num_tasks=self.num_tasks,
                pred_type=self.args.vae.task_pred_type,
            ).to(DeviceConfig.DEVICE)
        else:
            task_decoder = None

        return state_decoder, reward_decoder, task_decoder
    
    ###########################################################################
    #                        MAIN VAE LOSS COMPUTATION                        #
    ###########################################################################

    def compute_loss(
            self, 
            latent_mean: torch.Tensor, 
            latent_logvar: torch.Tensor, 
            vae_prev_obs: torch.Tensor,    
            vae_next_obs: torch.Tensor, 
            vae_actions: torch.Tensor,
            vae_rewards: torch.Tensor, 
            vae_tasks: torch.Tensor, 
            trajectory_lens: np.ndarray,
            ):
        """
        Computes the VAE loss for the given data.
        Batches everything together and therefore needs all trajectories to be 
        of the same length.
        (Important because we need to separate ELBOs and decoding terms so 
        can't collapse those dimensions)
        """
        num_unique_trajectory_lens = len(np.unique(trajectory_lens))

        # Ensure that if we have different trajectory lengths, we are 
        # subsampling both ELBO terms and decoding terms
        assert ((num_unique_trajectory_lens == 1) or 
                (self.args.vae.subsample_elbos and 
                 self.args.vae.subsample_decodes))
        # Ensure that we are not decoding only the past (because loss
        # computation requires decoding the entire trajectory)
        assert not self.args.vae.decode_only_past

        # Cut down the batch to the longest trajectory length to preserve the 
        # structure. NOTE: this means we will possibly waste some computation
        # on zero-padded trajectories that are shorter than max_traj_len.
        H = np.max(trajectory_lens)  # max_traj_len (H for horizon)

        (latent_mean, latent_logvar) = (latent_mean[:H + 1], latent_logvar[:H + 1])
        if isinstance(vae_prev_obs, dict):
            for key in vae_prev_obs.keys():
                vae_prev_obs[key] = vae_prev_obs[key][:H]
                vae_next_obs[key] = vae_next_obs[key][:H]
        else:
            (vae_prev_obs, vae_next_obs) = (vae_prev_obs[:H], vae_next_obs[:H])
        (vae_actions, vae_rewards) = (vae_actions[:H], vae_rewards[:H])

        # take one sample for each ELBO term
        if not self.args.vae.disable_stochasticity_in_latent:
            latent_samples = self.encoder._sample_gaussian(latent_mean, latent_logvar)
        else:
            latent_samples = torch.cat((latent_mean, latent_logvar), dim=-1)

        E = latent_samples.shape[0]  # number of ELBO terms
        D = vae_actions.shape[0]  # number of decoding terms
        B = latent_samples.shape[1]  # number of trajectories

        # Subsample elbo terms: [E B dim] -> [_E B dim]
        if self.args.vae.subsample_elbos is not None:
            _E = self.args.vae.subsample_elbos
            # randomly choose which elbo's to subsample
            if num_unique_trajectory_lens == 1:
                # Trajectories are all of the same length, so we can subsample
                # elbos uniformly across the batch
                elbo_indices = torch.tensor(_E * B, dtype=torch.long).random_(0, E)  # select diff elbos for each task
            else:
                # Trajectories are different lengths; we subsample elbo 
                # indices separately up to their maximum possible encoding 
                # length; only allow duplicates if the sample size would be
                # larger than the number of samples
                elbo_indices = np.concatenate(
                    [np.random.choice(range(0, t + 1), _E, replace=_E > (t+1)) 
                     for t in trajectory_lens])
                if _E > H:
                    warnings.warn(
                        '_E > H so there will be duplicates. To avoid this use '
                        '--split-batches-by-elbo or --split-batches-by-task.')
            # Create selection mask
            task_indices = torch.arange(B).repeat(_E) 
            latent_samples = \
                latent_samples[elbo_indices, task_indices, :].reshape((_E, B, -1))
            # Update number of elbos
            E = _E
        else:
            elbo_indices = None
            E = E  # just to make it explicit

        # Expand the s/r/a decoder inputs (to match size of latents)
        # New shape should be [E, H, B, a/s/r_dim]
        if isinstance(vae_prev_obs, dict):
            dec_prev_obs = dict()
            dec_next_obs = dict()
            for key in vae_prev_obs.keys():
                _expanded_shape = (E, *vae_prev_obs[key].shape)
                dec_prev_obs[key] = vae_prev_obs[key].unsqueeze(0).expand(_expanded_shape)
                dec_next_obs[key] = vae_next_obs[key].unsqueeze(0).expand(_expanded_shape)
        else:
            _expanded_shape = (E, *vae_prev_obs.shape)
            dec_prev_obs = vae_prev_obs.unsqueeze(0).expand(_expanded_shape)
            dec_next_obs = vae_next_obs.unsqueeze(0).expand(_expanded_shape)
        _expanded_shape = (E, *vae_actions.shape)
        dec_actions = vae_actions.unsqueeze(0).expand(_expanded_shape)
        _expanded_shape = (E, *vae_rewards.shape)
        dec_rewards = vae_rewards.unsqueeze(0).expand(_expanded_shape)

        # Subsample reconstruction terms [E D B dim] -> [E _D B dim]
        if self.args.vae.subsample_decodes is not None:
            _D = self.args.vae.subsample_decodes
            # NOTE(vb_authors): This current code will produce duplicates
            I0 = torch.arange(E).repeat(_D * B)  # indices for dim 0
            if num_unique_trajectory_lens == 1:
                I1 = torch.LongTensor(E * _D * B).random_(0, D)
            else:
                I1 = np.concatenate(             # indices for dim 1
                    [np.random.choice(range(0, t), E * _D, replace=True) 
                     for t in trajectory_lens])
            I2 = torch.arange(B).repeat(E * _D)  # indices for dim 2

            _new_shape = (E, _D, B, -1)
            # Dictionary observations
            if isinstance(dec_prev_obs, dict):
                dec_prev_obs = {
                    key: dec_prev_obs[key][I0, I1, I2, :].reshape(_new_shape) 
                         for key in dec_prev_obs.keys()}
                dec_next_obs = {
                    key: dec_next_obs[key][I0, I1, I2, :].reshape(_new_shape) 
                         for key in dec_next_obs.keys()}
            # Vectorized observations
            else:
                dec_prev_obs = dec_prev_obs[I0, I1, I2, :].reshape(_new_shape)
                dec_next_obs = dec_next_obs[I0, I1, I2, :].reshape(_new_shape)
            # Actions and rewards
            dec_actions = dec_actions[I0, I1, I2, :].reshape(_new_shape)
            dec_rewards = dec_rewards[I0, I1, I2, :].reshape(_new_shape)
            D = _D

        # Expand the latent (to match the number of s/r/a decoder inputs)
        # New shape will be [B, E, H, latent_dim]
        dec_embedding = latent_samples.unsqueeze(0).expand(
            (D, *latent_samples.shape)).transpose(1, 0)

        # Reconstruction loss computation; for each timestep encoded, 
        # decode everything and sum it up
        rew_rec_loss, state_rec_loss, task_rec_loss, = 0, 0, 0  # by default
        # Shape for each computed loss is [E, D, B]
        if self.args.vae.decode_reward:                               # Rewards
            rew_rec_loss = self.compute_rew_rec_loss(dec_embedding, 
                dec_prev_obs, dec_next_obs, dec_actions, dec_rewards)
            # (E) Aggregate across individual ELBO terms
            _agg_fn = torch.mean if self.args.vae.avg_elbo_terms else torch.sum
            rew_rec_loss = _agg_fn(rew_rec_loss, dim=0)
            # (D) Aggregate across individual reconstruction (decoding) terms
            _agg_fn = torch.mean if self.args.vae.avg_reconstruction_terms else torch.sum
            rew_rec_loss = _agg_fn(rew_rec_loss, dim=0)
            # (B) Aggregate across tasks (in batch)
            rew_rec_loss = rew_rec_loss.mean()
        if self.args.vae.decode_state:                                # States
            state_rec_loss = self.compute_state_rec_loss(
                dec_embedding, dec_prev_obs, dec_next_obs, dec_actions)
            # (E) Aggregate across individual ELBO terms
            _agg_fn = torch.mean if self.args.vae.avg_elbo_terms else torch.sum
            state_rec_loss = _agg_fn(state_rec_loss, dim=0)
            # (D) Aggregate across individual reconstruction (decoding) terms
            _agg_fn = torch.mean if self.args.vae.avg_reconstruction_terms else torch.sum
            state_rec_loss = _agg_fn(state_rec_loss, dim=0)
            # (B) Aggregate across tasks (in batch)
            state_rec_loss = state_rec_loss.mean()
        if self.args.vae.decode_task:                                 # Tasks
            task_rec_loss = self.compute_task_rec_loss(latent_samples, vae_tasks)
            # (E) Aggregate across individual ELBO terms
            _agg_fn = torch.mean if self.args.vae.avg_elbo_terms else torch.sum
            task_rec_loss = _agg_fn(task_rec_loss, dim=0)
            # (D) Aggregate across individual reconstruction (decoding) terms
            task_rec_loss = task_rec_loss.sum(dim=0)  
            # (B) Aggregate across tasks (in batch)
            task_rec_loss = task_rec_loss.mean()

        # Compute KL loss
        if not self.args.vae.disable_kl_term:
            # compute the KL term for each ELBO term of the current trajectory
            # shape: [num_elbo_terms] x [num_trajectories]
            kl_loss = self.compute_kl_loss(latent_mean, latent_logvar, elbo_indices)
            # avg/sum the elbos
            if self.args.vae.avg_elbo_terms:
                kl_loss = kl_loss.mean(dim=0)
            else:
                kl_loss = kl_loss.sum(dim=0)
            # average across tasks
            kl_loss = kl_loss.sum(dim=0).mean()
        else:
            kl_loss = 0

        return (rew_rec_loss, state_rec_loss, 
                task_rec_loss, kl_loss)

    def compute_vae_loss(
            self, 
            update=False, 
            pretrain_index=None):
        """ Returns the VAE loss """

        if not self.rollout_storage.ready_for_update():
            return 0

        if self.args.vae.disable_decoder and self.args.vae.disable_kl_term:
            return 0

        B = self.args.vae.batch_num_trajs
        # Get a mini-batch; each will be shape [H, B, -1]
        vae_prev_obs, vae_next_obs, vae_actions, vae_rewards, vae_tasks, \
            trajectory_lens = self.rollout_storage.get_batch(batchsize=B)

        # Encode; shape will be [H+1, B, latent_dim] --- includes the prior!
        _, latent_mean, latent_logvar, _ = self.encoder(actions=vae_actions,
            states=vae_next_obs, rewards=vae_rewards, hidden_state=None,
            return_prior=True, detach_every=self.args.vae.tbptt_stepsize 
            if hasattr(self.args, 'tbptt_stepsize') else None)

        # Compute loss. NOTE: We removed the ability to split batches by ELBO; 
        # see original VariBAD/HyperX codebase for that functionality
        losses = self.compute_loss(
            latent_mean, latent_logvar, vae_prev_obs, vae_next_obs, 
            vae_actions, vae_rewards, vae_tasks, trajectory_lens)
        # Collect individual losses
        (rew_reconstruction_loss, state_reconstruction_loss, 
         task_reconstruction_loss, kl_loss) = losses

        # VAE loss = KL loss + reward reconstruction + state transition 
        # reconstruction
        # take average (this is the expectation over p(M))
        loss = (self.args.vae.rew_loss_coeff * rew_reconstruction_loss +
                self.args.vae.state_loss_coeff * state_reconstruction_loss +
                self.args.vae.task_loss_coeff * task_reconstruction_loss +
                self.args.vae.kl_weight * kl_loss).mean()

        # Make sure we can compute gradients
        assert self._can_compute_gradients(losses)

        # Overall loss
        elbo_loss = loss.mean()

        # Update
        if update:
            self.optimiser_vae.zero_grad()  # Zero gradients
            elbo_loss.backward()            # Compute gradients
            self._clip_gradients()          # Clip gradients
            self.optimiser_vae.step()       # Update parameters
            self.update_count += 1

        self.log(elbo_loss, rew_reconstruction_loss, state_reconstruction_loss, 
                 task_reconstruction_loss, kl_loss, pretrain_index)
        return elbo_loss

    ###########################################################################
    #                         INDIVIDUAL LOSSES                               #
    ###########################################################################

    def compute_state_rec_loss(
            self, latent, prev_obs, next_obs, action, return_predictions=False):
        """
        Compute state reconstruction loss. (No reduction of loss along batch 
        dimension is done here; sum/avg has to be done outside)
        """

        state_pred = self.state_decoder(latent, prev_obs, action)

        if self.args.vae.state_pred_type == 'deterministic':
            loss_state = (state_pred - next_obs).pow(2).mean(dim=-1)
        elif self.args.vae.state_pred_type == 'gaussian':
            state_pred_mean = state_pred[:, :state_pred.shape[1] // 2]
            state_pred_std = torch.exp(
                0.5 * state_pred[:, state_pred.shape[1] // 2:])
            m = Normal(state_pred_mean, 
                                                  state_pred_std)
            loss_state = -m.log_prob(next_obs).mean(dim=-1)
        else:
            raise NotImplementedError

        if return_predictions:
            return loss_state, state_pred
        else:
            return loss_state

    def compute_rew_rec_loss(
            self, 
            latent: torch.Tensor, 
            prev_obs: torch.Tensor, 
            next_obs: torch.Tensor,                             
            action: torch.Tensor, 
            reward: torch.Tensor, 
            return_predictions: bool = False):
        """
        Compute reward reconstruction loss. (No reduction of loss along batch 
        dimension is done here; sum/avg has to be done outside)

        E = number of ELBO terms
        H = horizon length; number of steps in trajectory
        B = batch size; number of sampled trajectories
        
        Often T = E+1 (i.e. we have one ELBO term per timestep in the
        trajectory, and one for the prior)

        Args: 
            latent:     [E, H, B, latent_dim]
            prev_obs:   [E, H, B, obs_dim]
            next_obs:   [E, H, B, obs_dim]
            action:     [E, H, B, action_dim]
            reward:     [E, H, B, reward_dim]
            return_predictions: bool
        """

        if self.args.vae.multihead_for_reward:
            # Use multiple heads per reward pred (i.e. per state). 
            # NOTE: This means that we need to be able to enumnerate all states
            # and num_states should therefore be small. Otherwise we will have
            # too many heads, and we are better off sending in the state as 
            # input instead of creating a separate head for each.

            # Use reward decoder to predict reward from encoded latent
            rew_pred = self.reward_decoder(latent, None)
            if self.args.vae.rew_pred_type == 'categorical':
                # Categorical reward
                rew_pred = F.softmax(rew_pred, dim=-1)
            elif self.args.vae.rew_pred_type == 'bernoulli':
                # Bernoulli reward
                rew_pred = torch.sigmoid(rew_pred)

            
            # next_obs.shape = [10, 60, 2] for gridworld; num_steps = 15
            # next_obs.shape = [10, 400, 75] for MazeEnv; num_steps = 100
            # [task_batch_size, num_steps*4 (why?), obs_dim(s)]
            # In utils/evaluation.py, next_obs[1] is called 'traj_len'

            # Use next observations to get the ID of the task we will have 
            # completed by landing in that state (?)
            state_indices = self.task_to_id(next_obs).to(DeviceConfig.DEVICE)

            # Ensure state indices and reward predictions have matching 
            # dimensions; if state_indices has one fewer, we unsqueeze (i.e.
            # adding dimension at the end)
            if state_indices.dim() +1 == rew_pred.dim():
                state_indices = state_indices.unsqueeze(-1)
            elif state_indices.dim() == rew_pred.dim():
                pass
            else: 
                # They should either already match, or we should match by
                # unsqueezing once.
                raise ValueError
            # NOTE: state_indices is now shape:
            #                       [task_batch_size, num_steps, 1]

            # Gather values along final axis (because dim=-1), using task/state
            # IDs as indicies. 
            # NOTE: rew_pred is currently of shape:
            #                       [task_batch_size, num_steps*4, num_states]
            # We use state_indices (which index the states) to choose which
            # reward prediction to use for each item in the batch
            rew_pred = rew_pred.gather(dim=-1, index=state_indices)
            # New shape of rew_pred: 
            #                       [task_batch_size, num_steps*4, 1]
            
            # Depending on the reward prediction type, compute the loss between
            # the reward predictions and the targets
            rew_target = (reward == 1).float()
            if self.args.vae.rew_pred_type == 'deterministic':
                loss_rew = (rew_pred - reward).pow(2).mean(dim=-1)
            elif self.args.vae.rew_pred_type in ['categorical', 'bernoulli']:
                loss_rew = F.binary_cross_entropy(rew_pred, rew_target, 
                                                  reduction='none').mean(dim=-1)
            else:
                raise NotImplementedError
        else:
            # Use one head per reward pred
            # NOTE: This is better when we have too many states to create a new
            # head for each, and we're better of sending in the state as 
            # input.
            rew_pred = self.reward_decoder(latent, next_obs, prev_obs, 
                                           action.float())
            if self.args.vae.rew_pred_type == 'bernoulli':
                rew_pred = torch.sigmoid(rew_pred)
                rew_target = (reward == 1).float()
                loss_rew = F.binary_cross_entropy(rew_pred, rew_target, 
                                                  reduction='none').mean(dim=-1)
            elif self.args.vae.rew_pred_type == 'deterministic':
                loss_rew = (rew_pred - reward).pow(2).mean(dim=-1)
            else:
                raise NotImplementedError

        if return_predictions:
            return loss_rew, rew_pred
        else:
            return loss_rew

    def compute_task_rec_loss(self, latent, task, 
                                         return_predictions=False):
        """
        Compute task reconstruction loss. (No reduction of loss along batch 
        dimension is done here; sum/avg has to be done outside)
        """
        # raise NotImplementedError

        task_pred = self.task_decoder(latent)
        if self.args.vae.task_pred_type == 'task_id':
            task_target = self.task_to_id(task).to(DeviceConfig.DEVICE)
            # expand along first axis (number of ELBO terms)
            task_target = task_target.expand(task_pred.shape[:-1]).reshape(-1)
            loss_task = F.cross_entropy(task_pred.view(-1, task_pred.shape[-1]),
                                        task_target, reduction='none').view(
                                        task_pred.shape[:-1])
        elif self.args.vae.task_pred_type == 'task_description':
            loss_task = (task_pred - task).pow(2).mean(dim=-1)
        else:
            raise NotImplementedError

        if return_predictions:
            return loss_task, task_pred
        else:
            return loss_task

    def compute_kl_loss(self, latent_mean, latent_logvar, elbo_indices):
        # -- KL divergence
        if self.args.vae.kl_to_gauss_prior:
            kl_divergences = (- 0.5 * (1 + latent_logvar - latent_mean.pow(2) - 
                                       latent_logvar.exp()).sum(dim=-1))
        else:
            gauss_dim = latent_mean.shape[-1]
            # add the gaussian prior
            all_means = torch.cat((torch.zeros(
                1, *latent_mean.shape[1:], device=DeviceConfig.DEVICE), latent_mean))
            all_logvars = torch.cat((torch.zeros(
                1, *latent_logvar.shape[1:], device=DeviceConfig.DEVICE), latent_logvar))
            # https://arxiv.org/pdf/1811.09975.pdf
            # KL(N(mu,E)||N(m,S)) = 0.5 * (log(|S|/|E|) - K + tr(S^-1 E) + 
            #                       (m-mu)^T S^-1 (m-mu)))
            mu = all_means[1:]
            m = all_means[:-1]
            logE = all_logvars[1:]
            logS = all_logvars[:-1]
            kl_divergences = (
                0.5 * (torch.sum(logS, dim=-1) - torch.sum(logE, dim=-1) - 
                       gauss_dim + torch.sum(1 / torch.exp(logS) * 
                       torch.exp(logE), dim=-1) + ((m - mu) / torch.exp(logS) *
                       (m - mu)).sum(dim=-1)))

        # returns, for each ELBO_t term, one KL (so H+1 kl's)
        if elbo_indices is not None:
            batchsize = kl_divergences.shape[-1]
            task_indices = torch.arange(batchsize).repeat(
                self.args.vae.subsample_elbos)
            kl_divergences = kl_divergences[elbo_indices, task_indices].reshape(
                (self.args.vae.subsample_elbos, batchsize))

        return kl_divergences


    ###########################################################################
    #                          HELPER FUNCTIONS                               #
    ###########################################################################

    def _can_compute_gradients(self, losses):
        (rew_reconstruction_loss, state_reconstruction_loss, 
         task_reconstruction_loss, kl_loss) = losses
        if not self.args.vae.disable_kl_term:
            assert kl_loss.requires_grad
        if self.args.vae.decode_reward:
            assert rew_reconstruction_loss.requires_grad
        if self.args.vae.decode_state:
            assert state_reconstruction_loss.requires_grad
        if self.args.vae.decode_task:
            assert task_reconstruction_loss.requires_grad
        return True

    def _clip_gradients(self):
        # clip gradients
        if self.args.vae.encoder_max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.encoder.parameters(), 
                                     self.args.encoder_max_grad_norm)
        if self.args.vae.decoder_max_grad_norm is not None:
            if self.args.vae.decode_reward:
                nn.utils.clip_grad_norm_(self.reward_decoder.parameters(), 
                                         self.args.decoder_max_grad_norm)
            if self.args.vae.decode_state:
                nn.utils.clip_grad_norm_(self.state_decoder.parameters(), 
                                         self.args.decoder_max_grad_norm)
            if self.args.vae.decode_task:
                nn.utils.clip_grad_norm_(self.task_decoder.parameters(), 
                                         self.args.decoder_max_grad_norm)

    ###########################################################################
    #                              LOGGING                                    #
    ###########################################################################

    def log(self, elbo_loss, rew_reconstruction_loss, state_reconstruction_loss, 
            task_reconstruction_loss, kl_loss, pretrain_index=None):

        if pretrain_index is None:
            curr_iter_idx = self.get_iter_idx()
        else:
            curr_iter_idx = (- self.args.vae.pretrain_len * 
                             self.args.vae.num_updates_per_pretrain + 
                             pretrain_index)

        if curr_iter_idx % self.args.domain.log_interval == 0:

            if self.args.vae.decode_reward:
                self.logger.add('vae_losses/reward_reconstr_err', 
                                rew_reconstruction_loss.mean(), curr_iter_idx)
            if self.args.vae.decode_state:
                self.logger.add('vae_losses/state_reconstr_err', 
                                state_reconstruction_loss.mean(), curr_iter_idx)
            if self.args.vae.decode_task:
                self.logger.add('vae_losses/task_reconstr_err', 
                                task_reconstruction_loss.mean(), curr_iter_idx)

            if not self.args.vae.disable_kl_term:
                self.logger.add('vae_losses/kl', kl_loss.mean(), curr_iter_idx)
            self.logger.add('vae_losses/sum', elbo_loss, curr_iter_idx)