import argparse
import json
import os
import tempfile
import time
from copy import deepcopy
from typing import List

import matplotlib
import numpy as np
import wandb
from loguru import logger as logur
from ribs.schedulers import Scheduler
from scipy.stats import norm
from tqdm import tqdm

matplotlib.use('Agg')

from diva.components.qd import simulation as sim
from diva.components.qd.archives import FlatArchive, GridArchive
from diva.components.qd.emitters import (
    EvolutionStrategyCustomEmitter,
    MapElitesCustomEmitter,
)
from diva.components.qd.measures.measure_selection import MeasureSelector
from diva.environments import utils as utl
from diva.environments.box2d import bezier
from diva.environments.make_envs import make_vec_envs
from diva.utils.torch import DeviceConfig, tensor
from diva.utils.wandb import (
    MatrixWrapper,
    compress_and_encode_matrix,
    download_archive_data,
    hash_name,
)


class QDModule:
    def __init__(self,
                 args: 'argparse.Namespace', 
                 metalearner,
                 only_need_bounds=False,
                 skip_init=False):
        """
        Args:
            args: Arguments
            metalearner: MetaLearner
            only_need_bounds: If True, only need to know bounds of our genotype
                for PLR-GEN so that we can randomly generate genotypes within
                bounds. If False, we are doing a full QD run and generate
                everything.
            skip_init: If True, skip initialization.
        """
        self.args = args
        self.metalearner = metalearner
        self.logger = self.metalearner.logger
        self.plr_lock = self.metalearner.plr_lock
        self.plr_level_store = self.metalearner.plr_level_store
        self.plr_level_sampler = self.metalearner.plr_level_sampler
        self.only_need_bounds = only_need_bounds

        self.init_qd_updates_idx = 0
        self.archive_dims = self.metalearner.archive_dims  # It's set in init
        
        # Variables we should only set once at the beginning
        self.gt_type = self.args.domain.gt_type  # Same for the whole training run
        self.envs_ret_rms = self.metalearner.envs.venv.ret_rms if self.args.policy.norm_rew else None
        self.envs_max_episode_steps = self.metalearner.envs._max_episode_steps
        self.envs_compute_measures = self.metalearner.envs.compute_measures_static
        self.envs_process_genotype = self.metalearner.envs.process_genotype
        self.envs_is_valid_genotype = self.metalearner.envs.is_valid_genotype

        self.qd_warm_start_no_sim_objective = self.args.dist.qd.warm_start_no_sim_objective
        self.qd_no_sim_objective = self.args.dist.qd.no_sim_objective
        self.qd_plr_integration = self.args.dist.qd.plr_integration

        self.needs_reinit = False
        if not skip_init:
            self._init_components()
            self._check_module_state()
        else:
            self.needs_reinit = True

        if self.args.dist.qd.meas_alignment_objective:
            self.al_measures = self.args.dist.qd.meas_alignment_measures
            self.meas_al_normal_params = [self.measures_info[k].normal_params for k in self.og_measures + self.al_measures]
            self.meas_al_means, self.meas_al_std_devs = np.array([p[0] for p in self.meas_al_normal_params]), np.array([p[1] for p in self.meas_al_normal_params])
        else:
            self.al_measures = []
            self.meas_al_normal_params = None
            self.meas_al_means = None
            self.meas_al_std_devs = None

    def _check_module_state(self):
        if self.needs_reinit:
            raise ValueError('QDModule needs reinitialization after loading!')
        
    def _init_qd_envs(self): 
        """ Initialize vector environments for QD. """
        num_emitters = len(self.emitters)

        # Sometimes we'll be using a seedbased dist for archive
        if hasattr(self.args, 'archive_env_name'):
            archive_env_name = self.args.archive_env_name
            assert archive_env_name == self.args.eval_env_name  # For now, downstream
            self.oracle_population = True
        else:
            archive_env_name = self.args.domain.env_name
            self.oracle_population = False

        if self.args.dist.use_qd and not self.args.dist.qd.no_sim_objective or self.oracle_population:
            self.qd_envs, _ = make_vec_envs(archive_env_name,
                seed=self.args.seed * 42,
                num_processes=num_emitters*self.args.dist.qd.batch_size,
                gamma=self.args.policy.gamma,
                device=DeviceConfig.DEVICE,
                rank_offset=num_emitters*self.args.dist.qd.batch_size+1,  # to use diff tmp folders than main processes
                episodes_per_trial=self.args.domain.episodes_per_trial,
                normalise_rew=self.args.policy.norm_rew,
                plr=False,
                ret_rms=(self.metalearner.envs.venv.ret_rms if self.args.policy.norm_rew else None),
                tasks=None,
                add_done_info=self.args.domain.episodes_per_trial > 1,
                qd_tasks=None,  # NOTE: we set this in the actual sim loop
                dense_rewards=self.args.dense_rewards,
                **self.args.vec_env_kwargs
            )
        else:
            logur.info('Skipping QD environment creation (unnecessary)')

    def _compute_measures_info_from_samples(
            self, 
            dr_env_name, 
            ds_env_name, 
            measures,
            base_measures_info):
        """ Compute measures info from samples. """
        measures_info = MeasureSelector.compute_measures_info(
            dr_env_name,
            ds_env_name,
            measures=measures,
            num_samples=self.args.dist.qd.automatic_bounds_num_samples,
            base_measures_info=base_measures_info,
            dr_perc=self.args.dist.qd.automatic_bounds_percentage,
            env_kwargs=self.args.vec_env_kwargs
        )
        return measures_info
        
    def _init_components(self, init_num = 0):
        """ Initialise components for QD 
        
        Requires that environments and other metalearner components are
        initialized first.

        Args:
            init_num: Initialization attempt number---used for seeding.
        """
        # Check that metalearner has initialized necessary components first!
        assert hasattr(self.metalearner, 'envs')
        assert hasattr(self.metalearner, 'policy')

        self.skip_warm_start = False
        loaded_solutions = None
        # Load archive data from previous run if applicable
        if len(self.args.dist.qd.load_archive_from) > 0:
            run_name = self.args.dist.qd.load_archive_from
            run_index = self.args.dist.qd.load_archive_run_index
            loaded_solutions, self.archive_dims, self.args.dist.qd.measures = download_archive_data(run_name, run_index)
            # Skip warm start if we're not integrating PLR (otherwise we need it!)
            if not self.qd_plr_integration:
                self.skip_warm_start = True
            
        # NOTE: we use the same measure names for each warmstart phase, and
        # can shape archive to include/exclude certain measures from each phase
        og_measures = self.args.dist.qd.measures  
        if self.args.dist.qd.use_two_stage_ws and not self.skip_warm_start:
            # We should have already set this (when we set PLR buffer size)
            assert self.archive_dims == self.args.dist.qd.init_archive_dims
            use_normal_prior = False  # We only use in second stage
            sparsity_reweighting = False  # We only use in second stage
        else:
            # We should have already set this (when we set PLR buffer size)
            assert self.archive_dims == self.args.dist.qd.archive_dims
            use_normal_prior = True  # In one stage training, we use from get-go
            sparsity_reweighting = self.args.dist.qd.sparsity_reweighting
        init_seed = self.args.seed + init_num

        # We use archive information from test environment
        eval_env_name = self.args.domain.eval_env_name

        ### DEFINE MEASURE SPACE
        self.measure_selector = None
        kd_prior = None
        init_archive_samples = None

        if not self.only_need_bounds and self.args.dist.qd.use_measure_selector:
            # NOTE: measure selection is currently defunct!
            kd_prior, init_archive_samples = self.run_measure_selector(eval_env_name, init_seed)
        else:
            logur.debug(f'Using pre-defined measures: {og_measures}')
            logur.debug(f'Using automatic bounds? : {self.args.dist.qd.automatic_bounds}')

            measures_info = self.metalearner.envs.get_measures_info(eval_env_name)

            if self.args.dist.qd.automatic_bounds:
                logur.debug('Using AUTOMATIC bounds.')
                self.measures_info = self._compute_measures_info_from_samples(
                    self.args.domain.env_name,
                    eval_env_name,
                    # We will need normal params for alignment measures
                    measures=og_measures + self.args.dist.qd.meas_alignment_measures,
                    base_measures_info=measures_info)
            else:
                logur.debug('Using PRESET bounds.')
                self.measures_info = measures_info

            # Log artifact for measures info
            if self.args.use_wandb:
                measures_info_artifact = wandb.Artifact('measures_info', type='measures_info')
                temp_files = []
                try:
                    for measure, measure_info in self.measures_info.items():
                        dict_data = measure_info.__dict__
                        json_str = json.dumps(dict(dict_data))  # Serialize the dictionary to a JSON string
                        temp_file = tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.json')
                        temp_file.write(json_str)
                        temp_file_name = temp_file.name
                        temp_file.close()
                        measures_info_artifact.add_file(temp_file_name, name=f'measures_info_{measure}.json')
                        temp_files.append(temp_file_name)
                    wandb.log_artifact(measures_info_artifact)
                finally:
                    for temp_file_name in temp_files:
                        os.remove(temp_file_name)  # Remove the temporary files
            
            self.og_measures = og_measures
            self.measures = og_measures
            logur.info(f'measures: {self.measures}')
            self.measure_info = {k: self.measures_info[k] for k in self.measures}
            if self.args.dist.qd.use_two_stage_ws:
                self.measure_ranges = [self.measure_info[k].full_range for k in self.measures]
            else:
                self.measure_ranges = [self.measure_info[k].sample_range for k in self.measures]
            logur.info(f'measure_ranges: {self.measure_ranges}')
            self.sample_ranges = [self.measure_info[k].sample_range for k in self.measures]
            logur.info(f'sample_ranges: {self.sample_ranges}')
            self.distribution_types = [self.measures_info[k].sample_dist for k in self.measures]
            logur.info(f'distribution_types: {self.distribution_types}')
            self.normal_prior_axes = [i for i, x in enumerate(self.distribution_types) if x == 'normal']
            
        self.genotype_size = self.metalearner.envs.genotype_size
        logur.info(f'genotype_size: {self.genotype_size}')

        ### DEFINE ARCHIVE
        if self.args.dist.qd.use_flat_archive:
            self.archive = FlatArchive(
                solution_dim=self.genotype_size,
                cells=np.prod(self.archive_dims),
                seed=init_seed
            )
        else:
            self.archive = GridArchive(
                solution_dim=self.genotype_size,
                dims=self.archive_dims,
                ranges=self.measure_ranges,
                seed=init_seed,
                use_normal_prior=self.args.dist.qd.use_measure_selector or use_normal_prior,
                kd_prior=kd_prior if self.args.dist.qd.use_kd else None,
                sample_ranges=self.sample_ranges,
                update_sample_mask=self.args.dist.qd.update_sample_mask,
                normal_prior_axes=self.normal_prior_axes,
                sol_type=float if self.args.dist.qd.emitter_type == 'es' else np.int8,
                sparsity_reweighting=sparsity_reweighting,
                sparsity_reweighting_sigma=self.args.dist.qd.sparsity_reweighting_sigma,
            )
        if self.args.dist.use_plr:
            assert self.args.dist.plr.seed_buffer_size >= np.prod(self.archive_dims) + 100

        ### CREATE EMITTERS
        self._init_emitters(init_seed, init_archive_samples=init_archive_samples)
        
        self.scheduler = Scheduler(self.archive, self.emitters)

        # Necessary info for PLR-gen, for generating random solutions within
        # bounds, outside of QD code
        self.solution_dim = self.emitters[0].solution_dim
        self.upper_bounds = self.emitters[0].upper_bounds
        self.lower_bounds = self.emitters[0].lower_bounds

        # If warm start used, this flag will tell us if we should stop, i.e.
        # when we can't add anymore seeds to the buffer
        self.warm_start_done = False

        # Initialize vector environments for QD
        self._init_qd_envs()

        if loaded_solutions is not None:
            print('Loaded solutions shape:', loaded_solutions.shape)
            print(f'Computing measures for {len(loaded_solutions)} loaded solutions...')
            meas = [self.envs_compute_measures(genotype=sol, measures=self.og_measures, gt_type=self.gt_type) for sol in loaded_solutions]
            meas_vals = np.array([[sm[k] for k in self.og_measures] for sm in meas])
            obj_vals = np.ones(len(loaded_solutions))
            print('Loading solutions into archive...')
            self.archive.add(loaded_solutions, obj_vals, meas_vals)
            print(f'Loaded {len(self.archive._store)} solutions into archive!')
            # Deal with PLR level sampler
            if self.qd_plr_integration:
                # We set the weights with a kind of fake sample weight update
                self.archive.update_sample_weights(None, None, None)
            else:
                hashables = [tuple(s) for s in loaded_solutions]
                level_seeds = np.array([self.plr_level_store.insert(h, no_iter=True) for h in hashables])
                _ = self._add_seeds_to_level_sampler(level_seeds, warm_start_no_sim_objective=True)
                self._sync_samplers()
            # If we're doing QD/PLR integration, we need to evaluate solutions
            # before adding to PLR buffer

        self.init_seed = init_seed  # Save init seed for later emitter seeding
        self.archive_stage = 'train'

    def _init_emitters(self, init_seed, init_archive_samples=None):
        num_emitters = self.args.dist.qd.num_emitters
        self.num_emitters = num_emitters

        self.qd_archive_has_valid_solution = False
        if self.args.dist.qd.num_downstream_samples_to_use > 0:
            assert init_archive_samples is not None
            assert len(init_archive_samples) >= self.args.dist.qd.num_downstream_samples_to_use
            init_archive_samples = init_archive_samples[:self.args.dist.qd.num_downstream_samples_to_use]

        if self.args.dist.qd.emitter_type == 'es':
            if 'ToyGrid' in self.args.domain.env_name or 'Alchemy' in self.args.domain.env_name:
                initial_params = [[np.zeros(self.genotype_size)] for _ in range(num_emitters)]
                use_x0s = False
            if 'Racing' in self.args.domain.env_name:
                initial_params = [
                    bezier.get_random_points_unscaled(n=int(self.genotype_size/2)).flatten()
                    for _ in range(num_emitters)]
            else:
                initial_params = [np.random.random(self.genotype_size) * 2 - 1 
                                  for _ in range(num_emitters)]
            self.emitters = [
                EvolutionStrategyCustomEmitter(
                    self.archive,
                    initial_params[i],
                    bounds = self.metalearner.envs.genotype_bounds,
                    seed=init_seed+i,
                    sigma0=self.args.dist.qd.es_sigma0,
                    batch_size=self.args.dist.qd.batch_size,
                    initial_population=self.args.dist.qd.initial_population
                ) for i in range(num_emitters)  # Create num_emitters separate emitters.
            ]
        elif self.args.dist.qd.emitter_type == 'me':
            # initial_params = 
            if 'ToyGrid' in self.args.domain.env_name or 'Alchemy' in self.args.domain.env_name:
                initial_params = [[np.zeros(self.genotype_size)] for _ in range(num_emitters)]
                use_x0s = False
            elif init_archive_samples is None or self.args.dist.qd.num_downstream_samples_to_use == 0:
                print('Using all zeros as initial genotype.')
                initial_params = [[np.zeros(self.genotype_size)] for _ in range(num_emitters)]
                use_x0s = True
            else:
                print(f'Using {len(init_archive_samples)} downstream samples as initial genotypes.')
                initial_params = init_archive_samples
                # Split into num_emitters segments
                initial_params = np.array_split(initial_params, num_emitters)
                use_x0s = True
            self.emitters = [
                MapElitesCustomEmitter(
                    self.archive,
                    x0s=initial_params[i],
                    use_x0s=use_x0s,
                    bounds=self.metalearner.envs.genotype_bounds,
                    seed=init_seed+i,
                    batch_size=self.args.dist.qd.batch_size,
                    initial_population=self.args.dist.qd.initial_population,
                    mutation_k=(min(int(self.genotype_size), self.args.dist.qd.mutations_constant) if self.args.dist.qd.use_constant_mutations
                                else int(self.genotype_size * self.args.dist.qd.mutation_percentage)),
                    stepwise_mutations=self.args.dist.qd.stepwise_mutations
                ) for i in range(num_emitters)  # Create num_emitters separate emitters.
            ]
        else:
            raise ValueError(f'Invalid emitter type: {self.args.dist.qd.emitter_type}')

    def do_warm_start(self):
        # Fill the archive with initial solutions
        if not self.skip_warm_start:
            if self.args.dist.qd.use_two_stage_ws:
                logur.info('Performing 2-stage warm start...')
                self.fill_archive(stage=1, two_stage=True)
                self.prepare_second_ws_stage()
                self.fill_archive(stage=2, two_stage=True)
            else:
                logur.info('Performing 1-stage warm start...')
                self.fill_archive()

    def prepare_second_ws_stage(self):
        """ Prepare for second warm start stage. """
        self._check_module_state()

        assert self.args.dist.qd.use_two_stage_ws
        assert not self.skip_warm_start

        # If we're not using the sample mask, and there are no solutions
        # in the target, just keep the same archive!
        if not self.args.dist.qd.update_sample_mask and not self.archive._target_reached:
            self.warm_start_done = False
            return

        self.warm_start_done = False
        self.archive_dims = self.args.dist.qd.archive_dims
        # We use the sample range for the measure range now!
        measure_ranges = self.sample_ranges

        # Create new archive
        assert not self.args.dist.qd.use_flat_archive
        old_archive = self.archive
        self.archive = GridArchive(
            solution_dim=self.genotype_size,
            dims=self.archive_dims,
            ranges=measure_ranges,
            seed=self.init_seed,
            use_normal_prior=True,
            kd_prior=None, # No KD support
            sample_ranges=self.sample_ranges,
            update_sample_mask=self.args.dist.qd.update_sample_mask,
            normal_prior_axes=self.normal_prior_axes,
            sol_type=float if self.args.dist.qd.emitter_type == 'es' else np.int8,
            sparsity_reweighting=self.args.dist.qd.sparsity_reweighting,
            sparsity_reweighting_sigma=self.args.dist.qd.sparsity_reweighting_sigma,
        )
        
        # Create new emitters (NVM we don't want to generate random at start---keep same emitters!)
        # self._init_emitters(self.init_seed, init_archive_samples=None)

        # Create new scheduler
        self.scheduler = Scheduler(self.archive, self.emitters)

        # Add solutions from old archive to new archive
        occupied_indices = old_archive._store.occupied_list
        occupied, data = old_archive._store.retrieve(
            occupied_indices, ['solution', 'measures', 'objective'])
        assert all(occupied)
        solution_batch = data['solution']
        measures_batch = data['measures']
        objective_batch = data['objective'] 
        self.archive.add(solution_batch, objective_batch, measures_batch)
        self._sync_samplers()
        del old_archive

    def fill_archive(self, two_stage=False, stage=1):
        """ Run initial QD updates to fill archive with solutions."""
        self._check_module_state()

        if self.skip_warm_start: 
            return
    
        assert stage == 1 or stage == 2

        self.archive_stage = f'ws{stage}'
        
        if two_stage and stage==1:
            print('Beginning archive fill stage 1/2!')
            self.iter_idx = 0
            self.init_qd_updates_idx = 0
            warm_start_updates = self.args.dist.qd.init_warm_start_updates
        elif two_stage and stage==2:
            print('Beginning archive fill stage 2/2!')
            warm_start_updates = self.args.dist.qd.init_warm_start_updates + self.args.dist.qd.warm_start_updates
            pass  # These are already set!
        else:
            print('Beginning (one-stage) archive fill!')
            self.iter_idx = 0
            self.init_qd_updates_idx = 0
            warm_start_updates = self.args.dist.qd.warm_start_updates

        print('Filling archive...')
        time_prev = time.time()

        pbar = tqdm(total=int(warm_start_updates), desc="Progress", unit="update")

        while not (self.qd_archive_has_valid_solution and 
                self.init_qd_updates_idx >= warm_start_updates):
            # Update tqdm bar every 10 iterations
            if self.init_qd_updates_idx % 10 == 0:
                elapsed_time = np.round((time.time() - time_prev), 2)
                perc_filled = np.round(self.archive._stats.num_elites / self.archive._cells * 100, 2)
                
                pbar.set_postfix({
                    'num_elites': self.archive._stats.num_elites,
                    'perc_filled': f"{perc_filled}%",
                    'time': elapsed_time
                })
                pbar.update(10)  # Assuming 10 iterations are processed
                time_prev = time.time()

            # Perform update
            self.update(warm_start=True)

            if self.init_qd_updates_idx % 100 == 0:
                if not self.args.dist.qd.use_flat_archive:
                    self.archive.update_sample_mask(perc=self.init_qd_updates_idx/warm_start_updates)

            self.init_qd_updates_idx += 1
            if self.warm_start_done:
                break
        
        # Log at end
        self._log(self.latest_stats, iter_idx=self.init_qd_updates_idx)

        print('QD updates finished!')

        if self.args.dist.qd.warm_start_no_sim_objective:
            print('Updating QD objectives for stale values in archive')
            self.archive.reset_sslu()
            self._refresh_archive()

        self.archive_stage = 'train'

    @staticmethod
    def process_genotype_static(genotype, gt_is_continuous=False):
        if isinstance(genotype, (list, tuple)):
            genotype = np.array(genotype)
        # squeeze out extra dimensions
        genotype = genotype.squeeze()
        if gt_is_continuous:
            genotype = genotype.astype(float)
        else:
            genotype = genotype.astype(int)
        genotype = genotype.tolist()
        genotype = tuple(genotype)
        return genotype
    
    def process_genotype(self, genotype):
        self._check_module_state()
        return self.process_genotype_static(genotype, self.args.domain.gt_is_continuous)

    def _sync_samplers(self):
        """ Syncs the level_sampler and QD archive items.
        
        Specifically:
            (1): Removes PLR seeds that are not in the QD archive.
            (2): Updates sample_weights in QD archive from PLR scores.
                 NOTE: If sample strategy is set to random, these will be 
                 uniform weights!
        """
        self._check_module_state()
        # 1. Iterate through level_sampler, check that each item is in archive.
        #    If not, flag to remove from level_sampler (all items in level_sampler
        #    should already be in archive unless the cell has been overwritten).
        # Get archive indices and objectives
        indices = self.archive._store.occupied_list  # Already provides only occupied
        occupied, (sols) = self.archive._store.retrieve(indices, 'solution')
        assert all(occupied)  # All indices should be occupied
        if len(indices) == 0:
            # Archive is empty, so nothing to sync
            return
        # Collect all seeds in QD archive
        archive_seeds = set()
        archive_seed2index = {}
        seeds_debug = []
        count = 0
        for i, sol in zip(indices, sols):
            count += 1
            with self.plr_lock:
                seed = self.plr_level_store.level2seed[tuple(sol)]
            seeds_debug.append(seed)
            archive_seeds.add(seed)
            archive_seed2index[seed] = i

        # Collect seeds to purge from level_sampler (i.e. seeds not in the archive)
        seeds_to_purge = []  # [(index, seed)...]
        for i, seed in enumerate(self.plr_level_sampler.seeds):
            # If we've reached the edge of populated seeds, break
            if seed == -1: 
                break
            # Check if seed is in archive
            if seed not in archive_seeds:
                seeds_to_purge.append((i, seed))

        # 2. Purge seeds from level_sampler that are not in archive.
        if len(seeds_to_purge) > 0:
            self.plr_level_sampler.purge_seeds(seeds_to_purge)
            if self.plr_level_store is not None:
                self.plr_level_store.purge_seeds(seeds_to_purge)
        # NOTE: No need to update plr_level_sampler.next_seed_index. It's
        #   only used for the sequential sampling strategy; we never use this; it
        #   will always be zero and unused.
        # NOTE: We also do not need to purge seeds from level_store---it's
        #   actually beneficial to keep them in there, since we can use them
        #   in the future if an old solution is rediscovered in the archive.

        # 3. Obtain sample weights for each solution in level_sampler. 
        sample_weights = self.plr_level_sampler.sample_weights()
        weight_sum_to_check = sum(sample_weights)
        seed_weights = dict()
        for i, seed in enumerate(self.plr_level_sampler.seeds):
            if seed == -1:
                break
            seed_weights[seed] = sample_weights[i]

        # 4. Add these sample weights to the archive so that they can be used
        # for sampling solutions from the archive.
        self.archive.update_sample_weights(seed_weights, archive_seed2index, 
                                           weight_sum_to_check)

        # 5. Perform checks to ensure purging and updating was done correctly.
        try:
            assert len(seed_weights) == len(self.plr_level_sampler.seed2index), \
                f'{len(seed_weights)} != {len(self.plr_level_sampler.seed2index)}'
            assert len(seed_weights) == self.archive._stats.num_elites, \
                f'{len(seed_weights)} != {self.archive._stats.num_elites}'
        except Exception as _:
            # Print out information to debug:
            print('\nAssertion error in qd_sync_samplers()!')
            print('len(seed_weights):', len(seed_weights))
            print('len(self.plr_level_sampler.seed2index):', len(self.plr_level_sampler.seed2index))
            print('self.archive._stats.num_elites:', self.archive._stats.num_elites)
            raise AssertionError

    def _refresh_archive(self):
        """ Updates archive objectives that haven't been (re)computed in a while. 
        
        Specifically, we:
        1.  Get stale solutions from the archive---i.e. ones that haven't been
            updated in a while. We keep track of this in self.archive, and can
            just call self.archive.get_stale_solutions().
        2.  Compute updated objective values for each stale solution. For this, 
            we first split the solutions into batches of manageable size, and 
            then follow the same procedure as in self._evaluate_solutions():
                (1) Add seeds to level_store (they're already there but we
                    need to get the seeds for simulation)
                (2) Call self._simulate_solutions() to get sim objectives
                (3) Call self._compute_updated_qd_values() to get final 
                    new objectives and measures
        3.  Update the archive with the new objectives using specialized
            archive.update_stale_solutions() method.
        """
        self._check_module_state()

        # 1. Get stale solutions
        sols = self.archive.get_stale_solutions(sslu_threshold=self.args.dist.qd.sslu_threshold)
        if len(sols) == 0:
            # No stale solutions!
            return

        self.logger.lazy_add('qd/refresh_num', len(sols))
        self.logger.lazy_add('qd/refresh_perc', len(sols) / self.archive._stats.num_elites)
        self.logger.push_metrics(self.metalearner.iter_idx)
        
        # Pad sols to length that is divisible by num_emitters*self.args.dist.qd.batch_size by repeating the last item
        remainder = len(sols) % (self.num_emitters*self.args.dist.qd.batch_size)
        num_to_pad = self.num_emitters*self.args.dist.qd.batch_size - remainder
        sols = np.array(sols)
        if num_to_pad > 0:
            # Edge padding uses the last item in sols to pad
            sols = np.pad(sols, ((0, num_to_pad), (0, 0)), 'edge')  # Padding for 2D array

        # Split sols into batches the same size as normal QD batches
        N = self.args.dist.qd.batch_size * self.num_emitters
        num_batches = (len(sols) + N - 1) // N  # Ceiling division
        batches = np.array_split(sols, num_batches)

        # 2. Compute updated objectives for each batch
        objectives = []

        # TODO: Confusing argument behavior; we might want to rethink
        ws_no_sim_obj = False  # When we refresh the archive, we simulate by default (unless global no sim)
        
        # Compute objectives for each batch
        for sol_batch in batches:
            # NOTE Even though we already have seeds in level_store, we need to
            # recompute them here to get the level_seeds for simulation
            hashables = [tuple(s) for s in sol_batch]
            with self.plr_lock:
                level_seeds = np.array(
                    [self.plr_level_store.insert(h, no_iter=True) 
                     for h in hashables])
            # Compute simulated objectives
            if ws_no_sim_obj or self.qd_no_sim_objective:
                results = None
            else:
                results = self._simulate_solutions(sol_batch, level_seeds)
                
            # Process results
            objs, _, _ = self._compute_updated_qd_values_parallel(sol_batch, results, level_seeds)
            objectives.append(objs)

        # 3. Update archive with new objectives
        objectives = np.concatenate(objectives, axis=0)
        self.archive.update_stale_solutions(objectives)
        
        return

    def _simulate_solutions(self, 
                            sols: list[np.ndarray], 
                            level_seeds: list[int]):
        """ Simulate current policy on each solution to get results. 
        
        When we call sim.simulate(), we pass in the plr_level_sampler
        so that it can update the objective values there. Then, we merge
        the updated sampler back into the main sampler.

        NOTE: After calling this method, we must still process these results
        and update the archive with the new objectives now in the 
        plr_level_sampler.
        """
        self._check_module_state()

        ret_rms = self.envs_ret_rms

        # If no solutions, return None
        if len(sols) == 0:
            return None
        
        # Pad sols to length num_emitters*self.args.dist.qd.batch_size by repeating the last item
        num_to_pad = self.num_emitters*self.args.dist.qd.batch_size - len(sols)
        sols = np.array(sols)
        # Edge padding uses the last item in sols to pad
        sols = np.pad(sols, ((0, num_to_pad), (0, 0)), 'edge')  # Padding for 2D array

        # Pad level_seeds the same way
        level_seeds = np.array(level_seeds)
        level_seeds = np.pad(level_seeds, (0, num_to_pad), 'edge')  # Simplified padding for 1D array
            
        tensor_level_seeds = tensor(level_seeds)  # Simulate expects a tensor

        # Before simulating, we must stage the seeds (at least unseen ones;
        # but this is handled by the below method)
        with self.plr_lock:
            self.plr_level_sampler.stage_seeds(level_seeds)
            
        # TODO: Currently, we do not use hyperx bonuses here
        returns_per_episode, plr_level_sampler = \
            sim.simulate(
                args=self.args,
                policy=self.metalearner.policy,
                ret_rms=ret_rms,
                encoder=self.metalearner.vae.encoder,
                iter_idx=self.metalearner.iter_idx,
                # Below different from eval call
                tasks=None,
                policy_storage=self.metalearner.initialise_policy_storage(
                    num_steps=self.envs_max_episode_steps, 
                    num_processes=len(sols)),
                qd_tasks=sols,
                plr_level_sampler=deepcopy(self.plr_level_sampler),
                level_seeds=tensor_level_seeds,
                qd_envs=self.qd_envs
            )
        with self.plr_lock:
            self.plr_level_sampler.merge_sampler(plr_level_sampler)
        
        results = returns_per_episode
        return results
    
    def _compute_gt_diversity_objective_batch(self, sols, samples):
        """
        Compute genotype-based diversity objective for a batch of solutions.
        
        Parameters:
        sols (np.ndarray): 2D array where each row is a solution vector.
        samples (np.ndarray): 2D array where each row is a sample vector.
        
        Returns:
        np.ndarray: 1D array of average distances from each solution to all samples.
        """
        self._check_module_state()
        # Calculate the differences using broadcasting, sols[:, None] adds an extra dimension
        # making it (num_sols, 1, vector_length) and samples[None, :] makes it (1, num_samples, vector_length)
        # This allows for element-wise subtraction across all pairs of solutions and samples
        differences = sols[:, None, :] - samples[None, :, :]
        
        # Compute distances using norm over the last dimension (the vector components)
        distances = np.linalg.norm(differences, axis=2)
        
        # Calculate mean distance for each solution to all samples
        avg_distances = np.mean(distances, axis=1)
        
        return avg_distances
    
    def _compute_meas_diversity_objective_batch(self, sols, samples):
        """
        Compute genotype-based diversity objective for a batch of solutions.
        
        Parameters:
        sols (np.ndarray): 2D array where each row is a solution vector.
        samples (np.ndarray): 2D array where each row is a sample vector.
        
        Returns:
        np.ndarray: 1D array of average distances from each solution to all samples.
        """
        self._check_module_state()
        # First compute measures for each
        
        sols_measures = [self.envs_compute_measures(genotype=sol, measures=self.args.dist.qd.meas_diversity_measures, gt_type=self.gt_type) for sol in sols]
        sols_measures = np.array([[sm[k] for k in self.args.dist.qd.meas_diversity_measures] for sm in sols_measures])

        samples_measures = [self.envs_compute_measures(genotype=sample, measures=self.args.dist.qd.meas_diversity_measures, gt_type=self.gt_type) for sample in samples]
        samples_measures = np.array([[sm[k] for k in self.args.dist.qd.meas_diversity_measures] for sm in samples_measures])
        
        # Calculate the differences using broadcasting, sols[:, None] adds an extra dimension
        # making it (num_sols, 1, vector_length) and samples[None, :] makes it (1, num_samples, vector_length)
        # This allows for element-wise subtraction across all pairs of solutions and samples
        differences = sols_measures[:, None, :] - samples_measures[None, :, :]

        # Compute distances using norm over the last dimension (the vector components)
        distances = np.linalg.norm(differences, axis=2)
        
        # Calculate mean distance for each solution to all samples
        avg_distances = np.mean(distances, axis=1)
        
        return avg_distances

    @staticmethod
    def _process_solution(
            sol, 
            seed, 
            failed, 
            gt_type,
            qd_meas_diversity_objective,
            qd_meas_alignment_objective,
            meas_al_means,
            meas_al_std_devs,
            qd_randomize_objective,
            qd_gt_diversity_objective, 
            gt_diversity_distance,
            meas_diversity_distance,
            seed_idx,
            seed_score,
            compute_measures,
            is_valid_genotype,
            og_measures,
            measures_to_compute,
            archive_stage):

        ###  Measures  ###  
        rel_sol_meas, pg = compute_measures(
            genotype=sol, measures=measures_to_compute, 
            gt_type=gt_type, return_pg=True)
        rel_sol_meas = [rel_sol_meas[k] for k in measures_to_compute]
        sol_meas = rel_sol_meas
    
        ###  Objectives  ###   
        # (1) Simulation objectives         
        if seed_idx is not None:
            # If seed is in seed2index, then we have a score for it; NOTE:
            # it *should* be in here; otherwise it was rejected; e.g. if
            # the buffer was full. 
            # NOTE: We don't update seed score until it's nonzero (?)
            # See LevelSampler.after_update() in plr_level_sampler.py
            seed_idx = seed_idx
            sol_obj = seed_score
        elif failed is not None and seed in failed:
            # We failed to add it to plr_level_sampler using add_seeds().
            sol_obj = -1e6  
        else:
            # Not added to plr_level_sampler because it was full and the 
            # seed was rejected due to its score being low.
            sol_obj = -1e6  

        # (2) Genotype-based objectives
        if qd_gt_diversity_objective:
            # Compute genotype-based diversity objective
            sol_obj += gt_diversity_distance

        # (3) Measure-based diversity objective
        if qd_meas_diversity_objective:
            # Compute measure-based diversity objective
            sol_obj += meas_diversity_distance

        # (4) Measure-based objectives
        if qd_meas_alignment_objective:
            # Compute measure-based alignment objective

            if archive_stage == 'ws1':
                sol_mvals = sol_meas  # Use all for the objective in the first stage
                start_idx = 0
            else:
                sol_mvals = sol_meas[len(og_measures):]  # Use only the specified measures in second stage
                start_idx = len(og_measures)

            if qd_randomize_objective:
                # Generate random shifts from a normal distribution with mean 0 and std_devs
                random_shifts = np.random.normal(0, meas_al_std_devs[start_idx:], size=meas_al_std_devs[start_idx:].shape)
                adjusted_means = meas_al_means[start_idx:] + random_shifts
            else:
                adjusted_means = meas_al_means[start_idx:]
            log_probs = norm.logpdf(sol_mvals, adjusted_means, meas_al_std_devs[start_idx:])
            sum_log_probs = np.sum(log_probs)
            pos_log_prob_sum = max([(sum_log_probs + len(sol_mvals)*100_000) / (len(sol_mvals)*10_000), 0])  # To make the objective non-negative
            sol_obj += pos_log_prob_sum
        else:
            if qd_randomize_objective and sol_obj > 0:
                sol_obj = np.random.normal(sol_obj, 0.1)

        # (5) If solution invalid, set objective to -1e6
        valid, reason = is_valid_genotype(
            pg, gt_type=gt_type)
        if not valid:
            sol_obj = -1e6

        return sol_obj, sol_meas, valid, reason

    def _compute_updated_qd_values_parallel(self, sols, sim_results, level_seeds, failed=None):
        self._check_module_state()

        # GT objectives
        if self.args.dist.qd.gt_diversity_objective or self.args.dist.qd.meas_diversity_objective:
            # Sample n solutions from archive
            if self.archive._stats.num_elites < 20:
                _samples = sols  # Use the solutions themselves at beginning
            else:
                _elite_batch = self.archive.sample_elites(20)
                _samples = list(_elite_batch['solution'])
            
            if self.args.dist.qd.gt_diversity_objective:
                gt_diversity_samples = np.array(_samples)
                gt_diversity_distances = self._compute_gt_diversity_objective_batch(
                    sols, gt_diversity_samples)
            else:
                gt_diversity_distances = [0 for _ in sols]

            if self.args.dist.qd.meas_diversity_objective:
                meas_diversity_samples = np.array(_samples)
                meas_diversity_distances = self._compute_meas_diversity_objective_batch(
                    sols, meas_diversity_samples)
            else:
                meas_diversity_distances = [0 for _ in sols]
        else:
            gt_diversity_distances = [0 for _ in sols]
            meas_diversity_distances = [0 for _ in sols]

        if self.args.dist.qd.meas_alignment_objective:
            measures_to_compute = self.og_measures + self.al_measures
        else:
            measures_to_compute = self.og_measures
        
        pls = self.plr_level_sampler
        with self.plr_lock:
            seed_idxs = [pls.seed2index[seed] if seed in pls.seed2index else None for seed in level_seeds]
            seed_scores = [pls.seed_scores[seed_idx] if seed_idx is not None else None for seed_idx in seed_idxs]

        results = [self._process_solution(sol, seed, failed, self.gt_type, 
                self.args.dist.qd.meas_diversity_objective,
                self.args.dist.qd.meas_alignment_objective,
                self.meas_al_means, self.meas_al_std_devs,
                self.args.dist.qd.randomize_objective,
                self.args.dist.qd.gt_diversity_objective, 
                gt_diversity_distances[i], 
                meas_diversity_distances[i],
                seed_idxs[i], seed_scores[i], self.envs_compute_measures,
                self.envs_is_valid_genotype,
                self.og_measures, measures_to_compute, self.archive_stage)
            for i, (sol, seed) in enumerate(zip(sols, level_seeds))]
            
        sol_objs, sol_meas, valid_flags, reasons = zip(*results)
        sol_meas = [mval[:len(self.og_measures)] for mval in sol_meas]

        num_valid = sum(valid_flags)
        num_invalid = len(valid_flags) - num_valid
        self.qd_archive_has_valid_solution = num_valid > 0
        
        # Combine objectives and measures from sim and sols
        # TODO: use sim_results to compute more sophisticated objectives and measures
        objs, meas = sol_objs, sol_meas  # NOTE: placeholder

        if self.args.dist.qd.use_measure_selector:
            meas = np.array(meas)
            meas, _ = self.measure_selector.transform_data(meas)

        stats = {
            'perc_valid_generated': num_valid / len(sols),
            'perc_invalid_generated': num_invalid / len(sols),
            'archive_has_valid': int(self.qd_archive_has_valid_solution)
        }

        return objs, meas, stats      

    def _add_seeds_to_level_sampler(self, level_seeds, warm_start_no_sim_objective=False):
        """ Adds seeds to level_sampler. 
        
        NOTE: By default, we add seeds to the level_sampler with scores ~0.
        """
        self._check_module_state()
        # Add seeds to level_sampler
        if warm_start_no_sim_objective and self.args.dist.qd.bias_new_solutions:
            scores = [(1 + self.init_qd_updates_idx) * 1e-9 for _ in level_seeds]
        else:
            scores = [0 for _ in level_seeds]
        
        with self.plr_lock:
            failed = self.plr_level_sampler.add_seeds(level_seeds, scores)
        
        if len(failed) > 0:
            print('Failed to add seeds to level_sampler; syncing samplers.')
            self.warm_start_done = True        

    def _evaluate_solutions(self, sols, validities, ws_no_sim_obj=False):
        """ Returns objectives and measures for a set of solutions. """
        self._check_module_state()
        # Gather valid solutions and level_seeds
        valid_sols = [s for s, v in zip(sols, validities) if v]
        valid_idx = [i for i, v in enumerate(validities) if v]
        hashables = [tuple(s) for s in sols]
        with self.plr_lock:
            level_seeds = np.array([self.plr_level_store.insert(h, no_iter=True) for h in hashables])
        valid_level_seeds = level_seeds[valid_idx]

        # Simulate solutions
        if ws_no_sim_obj or self.qd_no_sim_objective:
            # If no simulation, add seeds to level sampler directly
            failed = self._add_seeds_to_level_sampler(
                valid_level_seeds, warm_start_no_sim_objective=True)
            sim_results = None
        else:
            failed = None
            sim_results = self._simulate_solutions(valid_sols, valid_level_seeds)

        # Compute objectives and measures from solutions themselves and sim results 
        objs, meas, stats = self._compute_updated_qd_values_parallel(
            sols, sim_results, level_seeds, failed=failed)

        return objs, meas, stats

    def sample_from_archive(self, num_samples, sample_random=False):
        """ Custom sampling function for archive. """
        self._check_module_state()
        valid_sols, num_valid, num_left, num_attempts = [], 0, num_samples, 0
        # We sometimes store invalid archive solutions as partial solutions
        while num_valid < num_samples:
            num_attempts += 1
            if sample_random:
                sols = self.emitters[0].rng.integers(
                            low=self.emitters[0].lower_bounds,
                            high=self.emitters[0].upper_bounds + 1,
                            size=(num_samples, self.emitters[0].solution_dim))
                sols = list(sols)
            else:
                elite_batch = self.archive.sample_elites(num_left*2+10)  # heuristic
                sols = list(elite_batch['solution'])

            for sol in sols:
                # Check if solution is valid
                pg = self.envs_process_genotype(sol, gt_type=self.gt_type)
                valid, reason = self.envs_is_valid_genotype(
                    pg, gt_type=self.gt_type)
                if valid:
                    valid_sols.append(sol)
                    num_valid += 1
                    num_left -= 1
                    if num_valid >= num_samples:
                        break
                else:
                    raise RuntimeError("Invalid solution sampled from archive.")

            if num_attempts > 100:
                raise RuntimeError("Could not sample enough valid solutions "
                                   "from archive ({} valid).".format(num_valid))
        return valid_sols

    def _get_random_oracle_sols(
        self, num_sols: int) -> List[np.ndarray]:
        """ Get random solutions from the oracle population. """
        self._check_module_state()
        # Get random seed values
        seeds = np.random.randint(0, 2**25, size=num_sols)
        # Reset environments to these seeds
        _ = utl.reset_env(self.qd_envs, self.args, task=seeds, eval=True, num_processes=num_sols)
        # Get genotypes from environments
        sols = np.array(self.qd_envs.genotype)
        return sols, seeds
        
    def update(self, warm_start=False):
        """ Method called to perform an update iteration on the archive. """
        self._check_module_state()
        # Refresh archive if necessary
        if self.args.dist.qd.refresh_archive and self.archive.get_max_sslu() > self.args.dist.qd.sslu_threshold:
            self._refresh_archive()

        # Request solutions
        self.archive.warm_start_percentage_complete = self.init_qd_updates_idx / self.args.dist.qd.warm_start_updates
        
        if self.oracle_population:
            # Sample from downstream seed-based distribution
            sols, seeds = self._get_random_oracle_sols(
                self.num_emitters*self.args.dist.qd.batch_size)
        elif self.qd_plr_integration:
            # Sample from QD archive (but don't call ask)
            sols = self.sample_from_archive(
                self.num_emitters*self.args.dist.qd.batch_size)
            seeds = None  # Seeds are created automatically during _evaluate_solutions
            assert not self.qd_no_sim_objective
        else:
            sols, seeds = self.scheduler.ask(), None

        validities = []
        for sol in sols:
            pg = self.envs_process_genotype(sol, gt_type=self.gt_type)
            valid, _ = self.envs_is_valid_genotype(
                pg, gt_type=self.gt_type)
            validities.append(valid)
        
        # Evaluate all solutions
        ws_no_sim_obj = self.qd_warm_start_no_sim_objective if warm_start else False
        objs, meas, stats = self._evaluate_solutions(sols, validities, ws_no_sim_obj=ws_no_sim_obj)

        # Compute how many elites currently in archive
        stats['prev_num_elites'] = self.archive.stats.num_elites
        if self.args.dist.qd.use_flat_archive:
            meas = np.zeros((len(sols), 1))

        # Send the results back to the scheduler.
        if self.oracle_population:
            # Cannot tell---need to do something else
            self.plr_level_sampler.add_seeds(seeds, scores=objs)
            add_info = self.archive.add(sols, objs, meas)
            del add_info  # Not used
        elif self.qd_plr_integration:
            pass  # Seeds are already added, but no "tell" required
        else:
            self.scheduler.tell(objs, meas)

        stats['new_num_elites'] = self.archive.stats.num_elites
        diff = stats['new_num_elites'] - stats['prev_num_elites']
        stats['perc_new_cells'] = diff / len(sols) if len(sols) > 0 else 0
        stats['perc_overwritten_cells'] = 1 - stats['perc_new_cells']

        if not self.args.dist.qd.use_flat_archive:
            stats['target_num_solutions'] = self.archive.num_sols_in_target
            stats['target_percentage_covered'] = self.archive.target_percentage_covered

        # Logging during normal training (RPLR, ACCEL, DIVA+)
        if (not self.args.dist.qd.no_sim_objective and 
                (self.iter_idx + 1) % self.args.dist.qd.log_interval == 0):
            self._log(stats)

        # Logging during warm start (DIVA)
        if warm_start and self.init_qd_updates_idx % self.args.dist.qd.ws_log_interval == 0:
            self._log(stats, iter_idx=self.init_qd_updates_idx)

        # Always sync samplers after an update UNLESS we're warm starting,
        # in which case we only sync every 10 updates.
        if not warm_start or self.init_qd_updates_idx % 200 == 0:
            if not self.qd_plr_integration:
                # No sync if we're using PLR integration
                self._sync_samplers()
        
        self.latest_stats = stats
    
    def _log(self, stats, iter_idx=None):
        """ Logging for QD. """
        self._check_module_state()
        if iter_idx is None:
            iter_idx = self.metalearner.iter_idx
        # NOTE: we use normal iter_idx, not qd_module.iter_idx
        lla, ii = self.logger.lazy_add, iter_idx
        lla('qd/num_elites', self.archive.stats.num_elites)
        lla('qd/coverage', self.archive.stats.coverage)
        lla('qd/qd_score', self.archive.stats.qd_score)
        lla('qd/obj_max', self.archive.stats.obj_max)
        lla('qd/obj_mean', self.archive.stats.obj_mean)
        lla('qd/perc_valid_generated', stats['perc_valid_generated'])
        lla('qd/perc_invalid_generated', stats['perc_invalid_generated'])
        lla('qd/perc_overwritten_cells', stats['perc_overwritten_cells'])
        lla('qd/perc_new_cells', stats['perc_new_cells'])
        if not self.args.dist.qd.use_flat_archive:
            lla('qd/target_percentage_covered', stats['target_percentage_covered'])
            lla('qd/target_num_solutions', stats['target_num_solutions'])
        # Push these metrics first so these still make it if the others fail
        self.logger.push_metrics(ii)

        logur.debug(f'LOGGING AT ITERATION: {ii}')

        # Log archive heatmap data
        if self.args.use_wandb:
            archive = self.archive
            print('Logging heatmap data to WandB')
            matrices = archive.get_matrices()
            hashed_label = hash_name(self.args.wandb_label)
            exp_num = self.args.wb_exp_index
            artifact_name = f'QD_{exp_num}_{hashed_label}_{self.archive_stage}'
            print('Artifact name: ', artifact_name)
            artifact = wandb.Artifact(artifact_name, type='qd-data')
            # Log all relevant matrices
            files_to_remove = []
            for matrix_name, matrix in matrices.items():
                print(f'Logging matrix {matrix_name}')
                if matrix is None:
                    continue
                encoded_matrix = compress_and_encode_matrix(matrix)
                matrix_wrapped = MatrixWrapper(encoded_matrix)
                json_obj = matrix_wrapped.to_json()

                pid = os.getpid()
                # Create a local file
                filename = f'_logs/wandb/_tmp__{pid}__{matrix_name}.json'
                print('Adding...', filename)
                with open(filename, 'w') as f:
                    f.write(json_obj)
                    f.flush()
                
                artifact.add_file(filename, f'{matrix_name}.json')
                files_to_remove.append(filename)

            print('Artifact logged!')
            wandb.log_artifact(artifact)
            
            print('Removing local (temporary) artifact files...')
            for filename in files_to_remove:
                os.remove(filename)
            
            # Log sample mask boundaries / distances
            if not self.args.dist.qd.use_flat_archive:
                lla('qd/sample_mask_edge_boundaries', json.dumps(self.archive._edge_boundaries))
                lla('qd/sample_mask_edge_distances', json.dumps(self.archive._edge_distances))
                self.logger.push_metrics(ii)