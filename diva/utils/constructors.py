import argparse
from typing import Any

import gym
import torch

from diva.components.policy.a2c import A2C
from diva.components.policy.ppo import PPO
from diva.components.policy.storage import OnlineStorage
from diva.environments import utils as utl
from diva.utils.schedulers import ConstantSchedule, LinearSchedule


def construct_policy(
    policy_class: Any,  # Policy class
    args: argparse.Namespace,  # Most args come from here
    action_space: gym.Space,
    state_feature_extractor: Any,
    logger: Any,
    device: torch.device,
    num_updates: int,
    optimiser_vae: Any,
):
    # Initialise policy network
    policy_net = policy_class(
        args=args,
        pass_state_to_policy=args.policy.pass_state,
        pass_latent_to_policy=args.policy.pass_latent,
        pass_belief_to_policy=args.policy.pass_belief,
        pass_task_to_policy=args.policy.pass_task,
        dim_state=args.state_dim,
        dim_latent=args.vae.latent_dim * 2,
        dim_belief=args.belief_dim,
        dim_task=args.task_dim,
        hidden_layers=args.policy.layers,
        activation_function=args.policy.activation_function,
        policy_initialisation=args.policy.initialisation,
        action_space=action_space,
        init_std=args.policy.init_std,
        state_feature_extractor=state_feature_extractor,
        state_is_dict=args.domain.state_is_dict,
        use_popart=args.use_popart,
        use_beta=args.policy.use_beta_distribution,
        logger=logger,
    ).to(device)
    # Args and kwargs shared by all policy optimisers
    common_args = [args, policy_net, args.policy.value_loss_coef, 
                    args.policy.entropy_coef]
    common_kwargs = {
        'policy_optimiser': args.policy.optimiser,
        'policy_anneal_lr': args.policy.anneal_lr,
        'train_steps': num_updates,
        'optimiser_vae': optimiser_vae,
        'lr': args.policy.lr,
        'eps': args.policy.eps,
        'logger': logger}
    a2c_kwargs = common_args
    ppo_kwargs = {
        'ppo_epoch': args.policy.ppo.num_epochs,
        'num_mini_batch': args.policy.ppo.num_minibatch,
        'use_clipped_value_loss': args.policy.ppo.use_clipped_value_loss,
        'clip_param': args.policy.ppo.clip_param,
        'use_huber_loss': args.policy.ppo.use_huberloss}
    ppo_kwargs.update(common_kwargs)
    # Initialise policy optimiser
    if args.policy.method == 'a2c':
        policy = A2C(*common_args, **a2c_kwargs)
    elif args.policy.method == 'ppo':
        policy = PPO(*common_args, **ppo_kwargs)
    else:
        raise NotImplementedError
    
    return policy


def get_policy_input_info(args, envs):
    """
    Get the online storage class, state feature extractor and state dtype.
    """
    if isinstance(envs.observation_space, gym.spaces.Dict):
        raise NotImplementedError('For dict case, need to define feature extractor.')
        # online_storage_class = DictOnlineStorage
        # state_keys = list(envs.observation_space.spaces.keys())
        # args.state_dim = {
        #     key: envs.observation_space.spaces[key].shape 
        #     for key in state_keys}
        # state_feature_extractor = \
        #     lambda i, o, a: mze.MazeFeatureExtractor(
        #     i, o, a, relevant_keys=args.state_relevant_keys)
        # state_dtype = torch.float32
    elif isinstance(envs.observation_space, gym.spaces.Box):
        online_storage_class = OnlineStorage
        if len(envs.observation_space.shape) == 3: # image
            args.state_dim = envs.observation_space.shape
            state_feature_extractor = utl.FeatureExtractorConv
            state_dtype = torch.uint8
        elif len(envs.observation_space.shape) == 1: # flat
            args.state_dim = envs.observation_space.shape[0]
            state_feature_extractor = utl.FeatureExtractor
            state_dtype = torch.float32
    else: 
        raise NotImplementedError
    return online_storage_class, state_feature_extractor, state_dtype


def construct_online_storage(
    online_storage_class: OnlineStorage,  # OnlineStorage or DictOnlineStorage
    args: argparse.Namespace, # Most args come from here
    model: torch.nn.Module,
    num_steps: int,
    num_processes: int,
    action_space: gym.Space,
    intrinsic_reward: Any = None,
):
    return online_storage_class(
        args                    =args,
        model                   =model,
        num_steps               =num_steps,
        num_processes           =num_processes,
        state_dim               =args.state_dim,
        latent_dim              =args.vae.latent_dim,
        belief_dim              =args.belief_dim,
        task_dim                =args.task_dim,
        action_space            =action_space,
        hidden_size             =args.vae.encoder_gru_hidden_size,
        normalise_rewards       =args.policy.norm_rew,
        use_gae                 =args.policy.use_gae,
        gamma                   =args.policy.gamma,
        tau                     =args.policy.tau,
        use_proper_time_limits  =args.policy.use_proper_time_limits,
        add_exploration_bonus   =args.add_exploration_bonus,
        intrinsic_reward        =intrinsic_reward,
        use_popart              =args.use_popart)



def get_plr_args_dict(args, plr_num_actors):
    """ Constructs a dictionary of arguments for convenient shorthand access. """
    return dict(   
        num_actors=plr_num_actors,
        strategy=args.dist.plr.level_replay_strategy,
        replay_schedule=args.dist.plr.level_replay_schedule,
        score_transform=args.dist.plr.level_replay_score_transform,
        temperature=args.dist.plr.level_replay_temperature_start,
        eps=args.dist.plr.level_replay_eps,
        rho=args.dist.plr.level_replay_rho,
        replay_prob=args.dist.plr.replay_prob, 
        alpha=args.dist.plr.level_replay_alpha,
        staleness_coef=args.dist.plr.staleness_coef,
        staleness_transform=args.dist.plr.staleness_transform,
        staleness_temperature=args.dist.plr.staleness_temperature,
        seed_buffer_size=args.dist.plr.seed_buffer_size)


def construct_plr_scheduler(args, num_updates):
    # If we use a temperature schedule, we need to initialize it
    if (args.dist.plr.level_replay_temperature_start
            == args.dist.plr.level_replay_temperature_end):
        plr_temperature_scheduler = ConstantSchedule(
            args.dist.plr.level_replay_temperature_start)
    else:
        plr_temperature_scheduler = LinearSchedule(
            start=args.dist.plr.level_replay_temperature_start,
            end=args.dist.plr.level_replay_temperature_end,
            steps=num_updates)
    return plr_temperature_scheduler