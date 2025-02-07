import numpy as np

from diva.components.qd.measures.alchemy_measures import AlchemyMeasures
from diva.environments.alchemy.alchemy_gym import (
    AlchemyMods,
    SymbolicAlchemyGymEnv,
    get_chemistry_from_seed,
    get_fixed_chemistries,
    get_items_from_seed,
)
from diva.environments.alchemy.genotypes.custom import CustomGenotype
from diva.environments.alchemy.genotypes.default import DefaultGenotype
from diva.environments.alchemy.genotypes.helpers import GRAPH_LIST
from dm_alchemy.types import utils as type_utils

Chemistry = type_utils.Chemistry
TrialItems = type_utils.TrialItems
EpisodeItems = type_utils.EpisodeItems


class ExtendedSymbolicAlchemy(SymbolicAlchemyGymEnv):
    def __init__(self,
                 distribution_type: str = 'SB',
                 variable_episode_lengths: bool = False,
                 gt_type: str = 's10-d-d-d-c6',
                 dense_rewards: bool = False,
                 alchemy_mods: list[str] = None,
                 use_dynamic_items: bool = True,
                 num_trials: int = 10,
                 *args, 
                 **kwargs):
        super(ExtendedSymbolicAlchemy, self).__init__(*args, **kwargs)
        # self.reset()
        
        # Set distribution type
        self.distribution_type = distribution_type
        assert self.distribution_type in ['SB', 'QD']
        # Set variable_episode_lengths
        self.variable_episode_lengths = variable_episode_lengths
        # Set genotype type
        self.gt_type = gt_type

        # Set simplifying modifications to the alchemy environment
        self.use_alchemy_mods = alchemy_mods is not None and len(alchemy_mods) > 0
        self.fix_stone_map = self.use_alchemy_mods and 'fix-st-map' in alchemy_mods
        self.fix_potion_map = self.use_alchemy_mods and 'fix-pt-map' in alchemy_mods
        self.fix_rotation = self.use_alchemy_mods and 'fix-st-rot' in alchemy_mods
        self.fix_graph = self.use_alchemy_mods and 'fix-graph' in alchemy_mods
        self.alchemy_mods = AlchemyMods(
            fix_stone_map=self.fix_stone_map,
            fix_potion_map=self.fix_potion_map,
            fix_rotation=self.fix_rotation,
            fix_graph=self.fix_graph)
        # NOTE: For now, we do not change genotype; we just override the genotype
        # when setting Alchemy variables, depending on the modifications.

        # Create genotype object
        self.use_dynamic_items = use_dynamic_items
        self.num_trials = num_trials
        self.gt = self.create_genotype_object(
            gt_type=self.gt_type, use_dynamic_items=self.use_dynamic_items, 
            num_trials=self.num_trials)
        self.current_episode_num = 0

        # For QD+metaRL
        self.genotype = None
        # Set genotype info
        self.set_genotype_info()
        self.size = self.genotype_size
        self.fixed_environment = False  
        self.dense_rewards = dense_rewards
        if self.distribution_type == 'QD':
            self.genotype_set = False
        
    def seed(self, seed=None):
        """ Set seed. """
        if seed is not None:
            seed = int(seed)
        self.rng_seed = seed

    def set_genotype_info(self):
        """ Extract information from genotype and genotype type. """
        self.measures = AlchemyMeasures.get_all_measures() 

        genotype_info = self.gt.genotype_info
        self.genotype_size = genotype_info.genotype_size
        self.genotype_lower_bounds = genotype_info.genotype_lower_bounds
        self.genotype_upper_bounds = genotype_info.genotype_upper_bounds
        self.genotype_bounds = genotype_info.genotype_bounds

    def get_measures_info(self, env_name):
        """ Get information on environment measures. """
        return AlchemyMeasures.get_measures_info(env_name)

    @staticmethod
    def compute_measures_static(genotype=None, measures=None, gt_type=None, return_pg=False):
        """ Compute measures for a given genotype. """
        # Process genotype
        pg = ExtendedSymbolicAlchemy.process_genotype(genotype, gt_type=gt_type)
        stone_latent_states = [sls[:ns] for ns, sls in zip(pg.num_stones, pg.stone_latent_states)]
        potion_effects = [pe[:npo] for pe, npo in zip(pg.potion_effects, pg.num_potions)]

        # Compute measures
        meas = AlchemyMeasures.compute_measures(
            stone_latent_states=stone_latent_states,
            stone_reflection=pg.stone_reflection,
            stone_rotation=pg.stone_rotation,
            potion_effects=potion_effects,
            potion_reflection=pg.potion_reflection,
            potion_permutation=pg.potion_permutation,
            graph_topology=pg.c_graph,
            measures=measures)
        
        if return_pg:
            return meas, pg
        else:
            return meas

    def compute_measures(self, genotype=None, measures=None, return_pg=False):
        """ Compute measures for a given genotype. """
        # Use current maze measures and genotype if none provided
        if measures is None:
            measures = self.measures
        if genotype is None:
            genotype = self.genotype

        # Process genotype
        pg = self._process_genotype(genotype)
        stone_latent_states = [sls[:ns] for ns, sls in zip(pg.num_stones, pg.stone_latent_states)]
        potion_effects = [pe[:npo] for pe, npo in zip(pg.potion_effects, pg.num_potions)]

        # Compute measures
        meas = AlchemyMeasures.compute_measures(
            stone_latent_states=stone_latent_states,
            stone_reflection=pg.stone_reflection,
            stone_rotation=pg.stone_rotation,
            potion_effects=potion_effects,
            potion_reflection=pg.potion_reflection,
            potion_permutation=pg.potion_permutation,
            graph_topology=pg.c_graph,
            measures=measures)
        
        if return_pg:
            return meas, pg
        else:
            return meas

    def generate_level_from_seed(self, seed):
        """ Generate a level from a given seed. """
        # Set the seed
        if seed is not None:
            self.seed(seed=seed)

        # Set self._generated_chemistry 
        self._generated_chemistry = get_chemistry_from_seed(
            self.rng_seed, alchemy_mods=self.alchemy_mods)

        # Set self._generated_items
        if self.rng_seed is None:
            self._generated_items = [get_items_from_seed(self.rng_seed) for _ in range(self.num_trials)]
        else:
            self._generated_items = [get_items_from_seed(self.rng_seed + 300_000*i) for i in range(self.num_trials)]
        
        # Call reset, which will generate environment from seed
        self.reset()

    @staticmethod
    def genotype_from_seed_static(
            seed, 
            gt_type,
            genotype_lower_bounds,
            genotype_upper_bounds,
            genotype_size):
        """ Generate level from seed. """
        # NOTE: Currently, the lower bounds and genotype size etc are needed
        # for CarRacing, but not for Alchemy; should clean this up later
        num_attempts = 0
        rng = np.random.default_rng(seed)
        gt = ExtendedSymbolicAlchemy.create_genotype_object(gt_type=gt_type)
        gt.set_rng(rng)
        while True:
            # Keep using this seed to generate environments until one is valid
            num_attempts += 1
            pg, valid, reason = gt.genotype_from_seed()
            genotype = pg.genotype
            del reason 
            if valid:
                break
            if num_attempts == 100:
                print('WARNING: Could not sample a valid solution after 100 attempts')
            if num_attempts > 1000:
                raise RuntimeError("Could not sample a valid solution")        
        return genotype

    def genotype_from_seed(self, seed, level_store=None):
        """ Generate or retrieve (if already generated) level from seed. """
        # First check if seed in level store
        if level_store is not None and seed in level_store.seed2level:
            return level_store.seed2level[seed]
        # Otherwise, generate level from seed
        num_attempts = 0
        rng = np.random.default_rng(seed)
        self.gt.set_rng(rng)
        while True:
            num_attempts += 1
            pg, valid, reason = self.gt.genotype_from_seed()
            genotype = pg.genotype
            del reason
            if valid:
                break
            if num_attempts == 100:
                print('WARNING: genotype_from_seed failed after 100 attempts')
            if num_attempts > 1000:
                raise ValueError('Could not sample a valid solution')
        return genotype

    def _process_genotype(self, genotype):
        """ Extract information from genotype and genotype type. """
        gisn = genotype is None

        assert gisn or len(genotype) == self.genotype_size, \
            f'Genotype length {len(genotype)} != {self.genotype_size}'

        return self.gt.process_genotype(genotype)

    @staticmethod
    def process_genotype(genotype, size=None, gt_type='default', use_dynamic_items=True, num_trials=10):
        """ Extract information from genotype and genotype type. 
        
        Ensure that the process_genotype method returns a Namespace with the
        following attributes:
            - num_stones
            - num_potions
            - stone_latent_states
            - stone_rotation
            - stone_reflection
            - potion_effects
            - potion_reflection
            - potion_permutation
            - graph_topology
            - p_stone_latent_coords
            - c_stones
            - c_potions
            - c_stone_map
            - c_potion_map
            - c_rotation
            - c_graph
            - genotype
            - genotype_size
            - genotype_lower_bounds
            - genotype_upper_bounds
            - genotype_bounds
        """
        gt = ExtendedSymbolicAlchemy.create_genotype_object(
            gt_type=gt_type, use_dynamic_items=use_dynamic_items, num_trials=num_trials)
        return gt.process_genotype(genotype, size=size)

    @staticmethod
    def create_genotype_object(gt_type='default', use_dynamic_items=True, num_trials=10):
        """ Create a genotype object. """
        if gt_type == 'default':
            if use_dynamic_items:
                raise NotImplementedError('Dynamic stones not implemented for default genotype')
            else:
                raise NotImplementedError('Static stones also not supported for default genotype')
            gt = DefaultGenotype()
        elif gt_type.count('-') == 4: # e.g. 'd-d-d-d-d' (for all defaults)
            # Split genotype string into components
            gt = CustomGenotype(gt_type=gt_type, use_dynamic_items=use_dynamic_items, num_trials=num_trials)
        else:
            raise ValueError('Unknown genotype type: {}'.format(gt_type))
        
        return gt

    @staticmethod
    def is_valid_genotype(pg, gt_type='default', use_dynamic_items=True):
        """ Check if genotype is valid. """
        gt = ExtendedSymbolicAlchemy.create_genotype_object(
            gt_type=gt_type, use_dynamic_items=use_dynamic_items)
        return gt.is_valid_genotype(pg)

    def generate_level_from_genotype(self, genotype, only_return=False):
        """ Generate a level from a given genotype. """
        if genotype is not None:
            self.genotype = np.array(genotype)
        
        if genotype is None and self.genotype_set:
            return
        
        # Use the pre-existing method to process the genotype
        processed_genotype = self._process_genotype(genotype)

        # Unpack the processed information
        c_stone_map = processed_genotype.c_stone_map
        c_potion_map = processed_genotype.c_potion_map
        c_rotation = processed_genotype.c_rotation
        c_graph = processed_genotype.c_graph
        c_stones = processed_genotype.c_stones
        c_potions = processed_genotype.c_potions

        # Fix relevant chemistries
        if self.alchemy_mods is not None:
            f_stone_map, f_rotation, f_potion_map, f_graph = get_fixed_chemistries()
            if self.alchemy_mods.fix_stone_map:
                c_stone_map = f_stone_map
            if self.alchemy_mods.fix_rotation:
                c_rotation = f_rotation
            if self.alchemy_mods.fix_potion_map:
                c_potion_map = f_potion_map
            if self.alchemy_mods.fix_graph:
                c_graph = f_graph

        # Set common variables
        self.genotype_size = processed_genotype.genotype_size
        self.genotype_lower_bounds = processed_genotype.genotype_lower_bounds
        self.genotype_upper_bounds = processed_genotype.genotype_upper_bounds
        self.genotype_bounds = processed_genotype.genotype_bounds
        self.genotype = processed_genotype.genotype

        # Create an instance of the Chemistry class and set it to the class variable
        generated_chemistry = Chemistry(
            potion_map=c_potion_map,
            stone_map=c_stone_map,
            graph=c_graph,
            rotation=c_rotation
        )

        # Create an instance of the TrialItems class and set it to the class variable
        generated_items = [TrialItems(potions=cp, stones=cs) for cp, cs in zip(c_potions, c_stones)]

        if not only_return:
            self._generated_chemistry = generated_chemistry
            self._generated_items = generated_items

        # Indicate that genotype is set
        if genotype is None:
            # NOTE: This is important because we might pass in a None genotype
            # here after we've already set one
            self.genotype_set = False
        else:
            self.genotype_set = True

        return generated_chemistry, generated_items

    def set_genotype_from_current_task(self, only_return=False):
        """ Set the genotype from the current task. """
        # Retrieve current Chemistry object
        chemistry = self.env_chemistry
        items = self.env_items

        # Reverse-engineer to get values for the genotype
        c_stone_map = chemistry.stone_map
        c_potion_map = chemistry.potion_map
        c_rotation = chemistry.rotation
        c_graph = chemistry.graph
        c_stones = [it.stones for it in items]
        c_potions = [it.potions for it in items]

        # Task to genotype
        genotype = self.gt.get_genotype_from_task_info(
            c_stone_map=c_stone_map,
            c_potion_map=c_potion_map,
            c_rotation=c_rotation,
            c_graph=c_graph,
            c_stones=c_stones,
            c_potions=c_potions
        )
        # DEBUG: To test resulting genotype
        # test_chem, test_items = self.generate_level_from_genotype(genotype, only_return=True)

        if only_return:
            return genotype
        else:
            self.genotype = genotype
            return genotype

    def reset(self, episode_num=0) -> np.ndarray:
        """ Reset the environment. """
        self.current_episode_num = episode_num

        if self.distribution_type == 'QD' and not self.genotype_set:
            # NOTE: If we're using QD, genotype needs to be set before we
            # generate the chemistry, etc.; this is message inherited from
            # MazeEnv (not sure if necessary here...)
            return None
        
        if self.fixed_environment:
            self.seed(self.seed_value)

        obs = super().reset()
        return obs

    def generated_chemistry_fn(self):
        """ Return the generated chemistry. """
        return self._generated_chemistry

    def generated_items_fn(self, episode_num=None, use_passed_episode_num=False):
        """ Return the generated items. """
        if use_passed_episode_num:
            episode_num = episode_num
        else:
            episode_num = self.current_episode_num
        
        # Called by SymbolicAlchemyEnv, which currently
        # doesn't have correct episode_numbers tracked; we track it here 
        # independently
        # TODO: Align trial numbers between this class and SymbolicAlchemy
        return self._generated_items[episode_num * int(self.use_dynamic_items)]

    def reset_task(self, task=None) -> None:
        """
        Reset current task (i.e. genotype, seed, etc.).

        Reset the task, either at random (if task=None) or the given task.
        """
        self.current_episode_num = 0

        # Set the chemistry_gen variable in self._env so that when
        # self._env.reset() is called, it will use the new chemistry_gen 
        # function to generate the chemistry. 
        if self.distribution_type == 'SB':
            # If we're using a seed-based distribution, we need to generate a
            # new seed and then generate the level from that seed
            self.seed(task)
            self.generate_level_from_seed(task)  # Sets self._generated_* variables
        elif self.distribution_type == 'QD':
            self.generate_level_from_genotype(task)  # Sets self._generated_* variables
        else:
            raise ValueError(f'Unknown distribution_type: {self.distribution_type}')
        
        # Set episode item/chemistry generation functions
        self._env._chemistry_gen = self.generated_chemistry_fn
        self._env._items_gen = self.generated_items_fn

        self.env_chemistry = self.generated_chemistry_fn()
        self.env_items = [self.generated_items_fn(i, use_passed_episode_num=True) for i in range(self.gt.num_item_specs)]

        self.reset(episode_num=0)

         # Set self.genotype from current task (generated from seed)
        if self.distribution_type == 'SB':
            _ = self.set_genotype_from_current_task()
        
        return self.genotype

    def get_task(self):
        """ Return the ground truth task. """
        # TODO: more thoughtful implementation
        if hasattr(self, 'genotype') and self.genotype is not None:
            return np.asarray(self.genotype).copy()
        else:
            return np.array((0.0,))

    def _reset_belief(self) -> np.ndarray:
        raise NotImplementedError('Oracle not implemented for Alchemy.')

    def update_belief(self, state, action) -> np.ndarray:
        raise NotImplementedError('Oracle not implemented for Alchemy.')

    def get_belief(self):
        raise NotImplementedError('Oracle not implemented for Alchemy.')


if __name__ == '__main__':

    for graph in GRAPH_LIST:
        print('\nnode_list:', graph.node_list)
        print('edge_list:', graph.edge_list)

    print(len(GRAPH_LIST))

    print(GRAPH_LIST[0])