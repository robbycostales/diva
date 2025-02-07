"""
Script for running experiments.
"""
import atexit
import os
import subprocess
import sys
import warnings

import hydra
from loguru import logger
from numba import NumbaDeprecationWarning
from omegaconf import DictConfig, OmegaConf

# Suppress specific warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Disables annoying TF warnings
os.environ['PYTHONWARNINGS'] = 'ignore::DeprecationWarning'
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

display_number = "9"
lock_file = f"/tmp/.X{display_number}-lock"

def start_xvfb(display):
    return subprocess.Popen(["Xvfb", display, "-screen", "0", "1024x768x24", "+extension", "GLX", "+render"])

def cleanup():
    if 'xvfb_process' in locals():
        xvfb_process.terminate()

if not os.path.exists(lock_file):
    xvfb_process = start_xvfb(f":{display_number}")
atexit.register(cleanup)
os.environ["DISPLAY"] = f":{display_number}"

import cProfile  # noqa: E402
import pstats  # noqa: E402
from io import StringIO  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import diva.environments.register  # Register environments  # noqa: E402, F401
from diva.environments.make_envs import make_vec_envs  # noqa: E402
from diva.environments.utils import get_vec_env_kwargs  # noqa: E402
from diva.metalearner import MetaLearner  # noqa: E402
from diva.utils.config import omegaconf_to_dict, print_dict  # noqa: E402
from diva.utils.constructors import get_plr_args_dict  # noqa: E402
from diva.utils.torch import (  # noqa: E402
    DeviceConfig,  # noqa: E402
    select_device,  # noqa: E402
)

# For gym API warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

logger.info('Starting experiment!')


@hydra.main(version_base=None, config_path='cfg', config_name='main')
def main(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)  # Allows for dynamic attribute assignment

    # Remove the old handler. Else, the old one will continue to run along with 
    # the new one specified below
    logger.remove()  

    # Specify logger behavior
    logger_format = (
        "<level>{level: <2}</level> <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan>"
        "- <level>{message}</level>"
    )
    if cfg.log_level == "debug" or cfg.debug:
        cfg.log_level = "debug"
        logger.add(sys.stdout, level="DEBUG", format=logger_format)
    elif cfg.log_level == "info":
        logger.add(sys.stdout, level="INFO", format=logger_format)
    else:
        raise ValueError(f"Invalid log level: {cfg.log_level}")

    dict = omegaconf_to_dict(cfg)
    print_dict(dict)
    args = cfg

    # Set args.domain.env_name
    args.domain.env_name = (args.domain.reg_env_id if 'oracle' in args.dist.name 
                            else args.domain.ued_env_id)
    # Set args.domain.eval_env_name
    args.domain.eval_env_name = args.domain.reg_env_id

    # Designed to avoid issues with domains overwriting default algo parameters:
    if args.dist.plr and args.dist.plr.domain_randomization:
        args.dist.plr.level_replay_strategy = 'random'

    if args.dist.qd:
        # If no initial warm starts, set boolean for convenience
        if args.dist.qd.init_warm_start_updates == 0:
            args.dist.qd.use_two_stage_ws = False
        else:
            args.dist.qd.use_two_stage_ws = True
        if args.dist.qd.dual_warm_start_updates is not None:
            args.dist.qd.init_warm_start_updates = args.dist.qd.dual_warm_start_updates
            args.dist.qd.warm_start_updates = args.dist.qd.dual_warm_start_updates

    # check if we're adding an exploration bonus
    if args.explore:
        args.add_exploration_bonus = args.exploration_bonus_hyperstate or \
                                    args.exploration_bonus_state or \
                                    args.exploration_bonus_belief or \
                                    args.exploration_bonus_vae_error
    else:
        args.add_exploration_bonus = False

    # warning for deterministic execution
    if args.deterministic_execution:
        print('Envoking deterministic code execution.')
        if torch.backends.cudnn.enabled:
            warnings.warn('Running with deterministic CUDNN.')
        if args.policy.num_processes > 1:
            raise RuntimeError(
                'If you want fully deterministic, run with num_processes=1.'
                'Warning: This will slow things down and might break A2C if '
                'policy_num_steps < env._max_episode_steps.')

    os.environ['TF_TENSORRT_DISABLED'] = '1'
    
    # Select device
    # GPU selection
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:  # If not defined we use GPU by default
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = os.environ['CUDA_VISIBLE_DEVICES']
    if len(device) == 0:
        # -1 is CPU
        select_device(-1)
        print("\nDevice: CPU")
    else:
        # Non-negative integer is the index of GPU
        select_device(0)
        print("\nDevice: GPU", device)

    # If we're normalising the actions, we have to make sure that the env 
    # expects actions within [-1, 1]
    if args.policy.norm_actions_pre_sampling or args.policy.norm_actions_post_sampling:
        # NOTE: We don't actually use these environments; just for assertions
        envs, _ = make_vec_envs(
            env_name=args.domain.env_name, seed=0, num_processes=args.policy.num_processes,
            gamma=args.policy.gamma, device='cpu',
            episodes_per_trial=args.domain.episodes_per_trial,
            normalise_rew=args.policy.norm_rew, ret_rms=None,
            plr=args.dist.use_plr,
            tasks=None,
            )
        assert np.unique(envs.action_space.low) == [-1]
        assert np.unique(envs.action_space.high) == [1]
        envs.close()
        del envs

    # Clean up arguments
    if args.vae.disable_metalearner or args.vae.disable_decoder:
        args.vae.decode_reward = False
        args.vae.decode_state = False
        args.vae.decode_task = False

    if hasattr(args, 'decode_only_past') and args.vae.decode_only_past:
        args.vae.split_batches_by_elbo = True

    # Init profiler
    if args.profile:
        args.domain.num_frames = 96_000 #50_000
        pr = cProfile.Profile()
        pr.enable()

    # Weights and Biases
    if args.use_wandb:
        assert args.wandb_label is not None
        args.wb_exp_index = args.wandb_label.split('_')[0]

    # We only use one seed for the main training loop
    assert isinstance(args.seed, int) or isinstance(args.seed[0], int)
    args.seed = args.seed if isinstance(args.seed, int) else args.seed[0]
    print('Training with seed: ', args.seed)

    # Prepare environment arguments
    args.action_space = None  # In case it wasn't defined (?)
    args.vec_env_kwargs = get_vec_env_kwargs(args)

    # Prepare QD params
    archive_dims = None
    try:
        if args.dist.qd:
            # Account for two-stage warm-start
            if args.dist.qd.use_two_stage_ws:
                init_archive_dims = args.dist.qd.init_archive_dims
                reg_archive_dims = args.dist.qd.archive_dims
                init_total_cells = np.prod(init_archive_dims)
                reg_total_cells = np.prod(reg_archive_dims)
                # We initialize PLR with maximum total cells between the two archives
                total_cells = max(init_total_cells, reg_total_cells)
                # Set current archive dims to initial archive dims
                archive_dims = init_archive_dims
            else:
                # Set PLR buffer size normally
                archive_dims = args.dist.qd.archive_dims # Default
                total_cells = np.prod(archive_dims)
        if args.dist.plr:
            # NOTE: we add 10_000 in the case that the archive is full and we need
            # space in the PLR buffer for scoring before we try to overwrite
            if args.dist.plr.buffer_size_from_qd:
                args.dist.plr.seed_buffer_size = int(total_cells) + 10_000
            elif args.dist.use_qd and args.dist.use_plr:
                args.dist.plr.seed_buffer_size = args.dist.plr.seed_buffer_size + 10_000
    except Exception:
        archive_dims = np.prod(args.dist.qd.archive_dims)

    # Initialize environments here for memory efficiency
    if args.dist.use_plr:
        if args.dist.use_qd and args.dist.qd.no_sim_objective:
            args.dist.plr.level_replay_strategy = 'random'  # Ignore objectives
        plr_num_actors = (args.dist.qd.batch_size * args.dist.qd.num_emitters if 
                          args.dist.use_qd else args.policy.num_processes) 
        plr_level_sampler_args = get_plr_args_dict(args, plr_num_actors)
    else:
        plr_level_sampler_args = None
    logger.info('Making training environments!')
    envs, plr_components = make_vec_envs(
        env_name=args.domain.env_name, 
        seed=args.seed, 
        num_processes=args.policy.num_processes,
        gamma=args.policy.gamma, 
        device=DeviceConfig.DEVICE,
        episodes_per_trial=args.domain.episodes_per_trial,
        normalise_rew=args.policy.norm_rew, 
        ret_rms=None,
        tasks=None, 
        qd=args.dist.use_qd,
        qd_tasks=None,
        plr=args.dist.use_plr,
        plr_level_sampler_args=plr_level_sampler_args,
        plr_level_sampler=None,  # We initialize inside the function for now
        plr_env_generator=args.dist.plr.env_generator if args.dist.plr else None,
        dense_rewards=args.dense_rewards, 
        **args.vec_env_kwargs)
    
    # Meta-learn
    learner_fn = None if args.vae.disable_metalearner else MetaLearner  # TODO: reimplement base RL learner
    learner = learner_fn(args, envs, plr_components, archive_dims=archive_dims)
    learner.train()
    
    # Profiling output
    if args.profile:
        s = StringIO()
        pr.disable()
        sortby = 'cumtime'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats(250)
        print(s.getvalue())


if __name__ == '__main__':
    main()
