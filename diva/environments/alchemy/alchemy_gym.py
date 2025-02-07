# This file modifies/extends dm_alchemy/symbolic_alchemy.py
import functools
from typing import (
    Any,
    Mapping,
    Optional,
    Sequence,
)

import numpy as np

from diva.environments.alchemy import symbolic_alchemy
from diva.environments.gym_wrapper import GymFromDMEnv
from dm_alchemy import symbolic_alchemy_trackers

# from dm_alchemy import symbolic_alchemy
from dm_alchemy.ideal_observer import precomputed_maps
from dm_alchemy.types import graphs, helpers, stones_and_potions
from dm_alchemy.types import utils as type_utils
from dm_alchemy.types.stones_and_potions import (
    latent_potion_from_index,
    latent_stone_from_index,
    possible_rotations,
    potion_map_from_index,
    stone_map_from_index,
)

Stone = stones_and_potions.Stone
Potion = stones_and_potions.Potion
LatentStoneIndex = stones_and_potions.LatentStoneIndex
LatentStone = stones_and_potions.LatentStone
LatentPotion = stones_and_potions.LatentPotion
AlignedStoneIndex = stones_and_potions.AlignedStoneIndex
PerceivedPotionIndex = stones_and_potions.PerceivedPotionIndex
AlignedStone = stones_and_potions.AlignedStone
PerceivedStone = stones_and_potions.PerceivedStone
PerceivedPotion = stones_and_potions.PerceivedPotion
CAULDRON = stones_and_potions.CAULDRON
random_stone_map = stones_and_potions.random_stone_map
random_potion_map = stones_and_potions.random_potion_map
random_latent_stone = stones_and_potions.random_latent_stone
random_latent_potion = stones_and_potions.random_latent_potion
random_rotation = stones_and_potions.random_rotation
random_graph = graphs.random_graph
graph_distr = graphs.graph_distr
create_graph_from_constraint = graphs.create_graph_from_constraint
possible_constraints = graphs.possible_constraints
bottleneck1_constraints = graphs.bottleneck1_constraints
bottleneck2_constraints = graphs.bottleneck2_constraints
bottleneck3_constraints = graphs.bottleneck3_constraints
no_bottleneck_constraints = graphs.no_bottleneck_constraints
Chemistry = type_utils.Chemistry
TrialItems = type_utils.TrialItems
ElementContent = type_utils.ElementContent
SeeChemistry = type_utils.ChemistrySeen
GetChemistryObs = type_utils.GetChemistryObsFns

_NUM_TRIALS = 2
_TRIAL_STONES = [Stone(0, [-1, -1, 1]),
                 Stone(1, [1, 1, 1]),
                 Stone(2, [1, 1, -1])]
_TEST_STONES = [_TRIAL_STONES for _ in range(_NUM_TRIALS)]

_TRIAL_POTIONS = [Potion(0, 1, 1),
                  Potion(1, 1, -1),
                  Potion(2, 1, 1),
                  Potion(3, 1, 1),
                  Potion(4, 2, 1),
                  Potion(5, 1, 1),
                  Potion(6, 2, -1),
                  Potion(7, 0, 1),
                  Potion(8, 2, -1),
                  Potion(9, 2, 1),
                  Potion(10, 1, 1),
                  Potion(11, 1, -1)]
_TEST_POTIONS = [_TRIAL_POTIONS for _ in range(_NUM_TRIALS)]

_MAX_STEPS_PER_TRIAL = 20

_FIXED_POTION_MAP = stones_and_potions.all_fixed_potion_map()
_FIXED_STONE_MAP = stones_and_potions.all_fixed_stone_map()
_FIXED_ROTATION = np.eye(3, dtype=int)

_CHEM_NAME = 'test_chem'

SymbolicAlchemyTracker = symbolic_alchemy_trackers.SymbolicAlchemyTracker
ActionInfo = symbolic_alchemy_trackers.ActionInfo

STONE_COUNT_SCALE = 3.0
POTION_COUNT_SCALE = 12.0
POTION_TYPE_SCALE = PerceivedPotion.num_types / 2.0
REWARD_SCALE = stones_and_potions.max_reward()
END_TRIAL = helpers.END_TRIAL
NO_OP = -1
UNKNOWN_TYPE = -3
MAX_STONES = 3
MAX_POTIONS = 12
NO_EDGE = graphs.NO_EDGE
DEFAULT_MAX_STEPS_PER_TRIAL = 20


class AlchemyMods:
    def __init__(self,
                 fix_stone_map=False,
                 fix_potion_map=False,
                 fix_rotation=False,
                 fix_graph=False):
        self.fix_stone_map = fix_stone_map
        self.fix_potion_map = fix_potion_map
        self.fix_rotation = fix_rotation
        self.fix_graph = fix_graph

    def __str__(self) -> str:
        # Print to show all attribute values
        return str(self.__dict__)


def reward_fcn():
    return stones_and_potions.RewardWeights([1, 1, 1], 0, 12)


def make_fixed_chem_env(
    constraint=None, potion_map=_FIXED_POTION_MAP, stone_map=_FIXED_STONE_MAP,
    rotation=_FIXED_ROTATION, test_stones=None, test_potions=None, **kwargs
):
    if constraint is None:
        constraint = graphs.no_bottleneck_constraints()[0]
    env = get_symbolic_alchemy_fixed(
        episode_items=type_utils.EpisodeItems(
            potions=test_potions or _TEST_POTIONS,
            stones=test_stones or _TEST_STONES),
        chemistry=type_utils.Chemistry(
            graph=graphs.create_graph_from_constraint(constraint),
            potion_map=potion_map, 
            stone_map=stone_map, 
            rotation=rotation),
        reward_weights=reward_fcn(), max_steps_per_trial=_MAX_STEPS_PER_TRIAL,
        **kwargs)
    return env


def make_random_chem_env(**kwargs):
    # NOTE: This is just to get an initial environment, before we set the task
    env = get_symbolic_alchemy_level(
        level_name='perceptual_mapping_randomized_with_random_bottleneck',
        reward_weights=reward_fcn(), 
        max_steps_per_trial=_MAX_STEPS_PER_TRIAL,
        num_trials=10,  # TODO: Should not be hardcoded
        **kwargs)
    return env


class SymbolicAlchemyGymEnv(GymFromDMEnv):
    """ Gym wrapper for symbolic alchemy environment. """
    def __init__(self, 
                 env_type='random',
                 max_steps=20,
                 visualize=False):
        if env_type == 'fixed':
            env = make_fixed_chem_env()  
        elif env_type == 'random':
            env = make_random_chem_env()
        else:
            raise ValueError(f"Unknown env_type: {env_type}")
        super().__init__(env)
        self._visualize = visualize
        self._max_episode_steps = max_steps
        self._current_episode_steps = 0
        
        if self._visualize:
            raise NotImplementedError('Visualization not implemented yet!')
            # print('Initializing visualizer!')
            # self._visualizer = AlchemyVisualizer()
            # self._visualizer_setup = False
    
    def step(self, action):
        # We expect action to be an integer
        if isinstance(action, np.ndarray) and action.size == 1:
            action = action.item()
        elif isinstance(action, int):
            pass
        else:
            raise ValueError(f"Unknown action type: {type(action)}")
        obs, reward, done, info = super().step(action)
        self._current_episode_steps += 1
        if isinstance(obs, dict):
            obs = obs['symbolic_obs']
        # Varibad expects done to be True when max_episode_steps is reached
        # for each trial, and then it will reset the environment, etc.
        if self._current_episode_steps >= self._max_episode_steps:
            done = True

        # Visualization step:
        if self._visualize:
            print('Visualization step!')
            self._visualizer.step(
                game_state=self._env.game_state,
                state=self._last_observation,
                action=action,
                next_state=obs,
                timestep=self._env._last_time_step,
                trial_number=self._env.trial_number,
                is_new_trial=self._env.is_new_trial()
            )

        # If done, log a few things:
        if done:
            # We have to use "prev_game_state", because Alchemy has reset the
            # environment to start a new trial automatically,
            # if we've received 'done'. The previous game state 
            # gives us the final game_state of the last trial, which is what
            # we want.
            stones = self._env.prev_game_state._stones
            latents = [s.latent for s in stones]
            goal = np.ones(3)
            if info is None:
                info = {}
            # Percentage of stones in final state (i.e. latents[i] == goal)
            info['final_stone_percent'] = np.mean(np.all(np.array(latents) == goal, axis=1))
            # Average stone L1 norm distance from final state
            info['final_stone_distance'] = np.sum(np.abs(np.array(latents) - goal), axis=1).mean()
            # Success
            info['success'] = (info['final_stone_percent'] == 1)
            # Get number of stones remaining (ones that haven't been used)
            info['final_num_existing_stones'] = len(self._env.prev_game_state._existing_stones)
            info['final_num_existing_potions'] = len(self._env.prev_game_state._existing_potions)

        self._last_observation = obs
        return obs, reward, done, info
    
    def reset(self):
        self._current_episode_steps = 0
        obs = super().reset()
        if isinstance(obs, dict):
            obs = obs['symbolic_obs']
        self._last_observation = obs

        # Visualization setup:
        if self._visualize and not self._visualizer_setup:
            print('Setting up visualizer!')
            self._visualizer.setup(self._env.game_state, self._last_observation)

        return obs

    def render(self, mode='human'):
        assert self._visualize, "Can only render when visualize=True"
        return self._visualizer.last_rendered_frame

def graph_list(
        constraints: Sequence[graphs.Constraint]
) -> Sequence[graphs.Graph,]:
    """Returns list of all valid graphs."""
    num_constraints = graphs.get_num_constraints(constraints)
    valid_graphs = [
        (graphs.create_graph_from_constraint(constraint), constraint_count)
        for constraint, constraint_count in zip(constraints, num_constraints)]
    valid_graphs = [g[0] for g in valid_graphs]
    return valid_graphs


GRAPH_LIST = graph_list(possible_constraints())


def get_graph_from_index(index: int) -> graphs.Graph:
    return GRAPH_LIST[index]


LATENT_STONE_LIST = [latent_stone_from_index(index) for index in range(8)]
LATENT_POTION_LIST = [latent_potion_from_index(index) for index in range(6)]


def potion_to_latent_potion(potion: Potion) -> LatentPotion:
    return LatentPotion(latent_dim=potion.dimension, latent_dir=potion.direction)


def get_chemistry_from_seed(seed: int, 
                            alchemy_mods: AlchemyMods = None) -> Chemistry:
    """Gets a chemistry from a seed.
    
    Modified so that we can pass in a set of alchemy mods to simplify the
    environment's chemistry.
    """
    am = alchemy_mods
    # TODO: Can remove full randomization to make environment easier later on.
    
    # Create random generators for each element
    random_state = np.random.RandomState(seed)
    # Stone map
    if am is not None and am.fix_stone_map:
        stone_map_gen = lambda: stone_map_from_index(0)  # noqa: E731
    else:
        stone_map_gen = functools.partial(random_stone_map, random_state=random_state)
    stone_map = stone_map_gen()
    # Stone rotation
    if am is not None and am.fix_rotation:
        rotation = possible_rotations()[0]
    else:
        rotation = random_rotation(random_state=random_state)
    # Potion map
    if am is not None and am.fix_potion_map:
        _, index_to_perm_index = precomputed_maps.get_perm_index_conversion()
        potion_map_gen = lambda: potion_map_from_index(0, index_to_perm_index)  # noqa: E731
    else:
        seeded_rand_potion_map = functools.partial(random_potion_map, random_state=random_state)
        _, index_to_perm_index = precomputed_maps.get_perm_index_conversion()
        potion_map_gen = lambda: seeded_rand_potion_map(index_to_perm_index)  # noqa: E731
    potion_map = potion_map_gen()
    # Graph
    if am is not None and am.fix_graph:
        constraint = no_bottleneck_constraints()[0]
        graph = create_graph_from_constraint(constraint)
    else:
        graph = random_graph(graph_distr(possible_constraints()), random_state=random_state)

    # Generate chemistry
    chemistry = Chemistry(
        potion_map=potion_map,
        stone_map=stone_map,
        graph=graph,
        rotation=rotation)
    return chemistry


def get_fixed_chemistries() -> Sequence[Any]:
    """Gets fixed chemistries."""
    # Stone map
    stone_map = stone_map_from_index(0)
    # Stone rotation
    rotation = possible_rotations()[0]
    # Potion map
    _, index_to_perm_index = precomputed_maps.get_perm_index_conversion()
    potion_map = potion_map_from_index(0, index_to_perm_index)
    # Graph
    constraint = no_bottleneck_constraints()[0]
    graph = create_graph_from_constraint(constraint)
    return stone_map, rotation, potion_map, graph


def get_items_from_seed(seed: int) -> TrialItems:
    """Gets items from a seed."""
    random_state = np.random.RandomState(seed)
    # TODO: For now we are defining uniform dist over number of stones/potions
    num_stones = random_state.randint(1, MAX_STONES + 1)
    num_potions = random_state.randint(1, MAX_POTIONS + 1)
    stones_in_trial = [random_latent_stone(random_state=random_state)
                       for _ in range(num_stones)]
    potions_in_trial = [random_latent_potion(random_state=random_state)
                        for _ in range(num_potions)]

    trial_items = TrialItems(potions=potions_in_trial, stones=stones_in_trial)

    return trial_items


class ChemistryGen:
    def __init__(self, potion_map_gen, stone_map_gen, graph_gen, rotation_gen):
        self.potion_map_gen = potion_map_gen
        self.stone_map_gen = stone_map_gen
        self.graph_gen = graph_gen
        self.rotation_gen = rotation_gen

    def __call__(self):
        return Chemistry(
            self.potion_map_gen(),
            self.stone_map_gen(),
            self.graph_gen(),
            self.rotation_gen()
        )


class PotionMapGen:
    def __init__(self, random_state, index_to_perm_index):
        self.random_state = random_state
        self.index_to_perm_index = index_to_perm_index

    def __call__(self):
        # NOTE: Why these are being called in the opposite order from the
        # original code is beyond me...
        return random_potion_map(self.index_to_perm_index, self.random_state)


class StoneMapGen:
    def __init__(self, random_state):
        self.random_state = random_state

    def __call__(self):
        return random_stone_map(random_state=self.random_state)


class RotationGen:
    def __init__(self, random_state, level_name):
        self.random_state = random_state
        self.level_name = level_name

    def __call__(self):
        if 'rotation' in self.level_name:
            return random_rotation(random_state=self.random_state)
        else:
            return np.eye(3)


class ItemsGen:
    def __init__(self, random_state, num_stones_per_trial, num_potions_per_trial):
        self.random_state = random_state
        self.num_stones_per_trial = num_stones_per_trial
        self.num_potions_per_trial = num_potions_per_trial

    def __call__(self, unused_trial_number):
        del unused_trial_number
        stones_in_trial = [random_latent_stone(random_state=self.random_state) for _ in range(self.num_stones_per_trial)]
        potions_in_trial = [random_latent_potion(random_state=self.random_state) for _ in range(self.num_potions_per_trial)]
        return TrialItems(potions=potions_in_trial, stones=stones_in_trial)


class GraphGen:
    def __init__(self, random_state, level_name):
        self.random_state = random_state
        self.level_name = level_name

    def __call__(self):
        seeded_rand_graph = functools.partial(random_graph, random_state=self.random_state)
        if 'random_bottleneck' in self.level_name:
            return seeded_rand_graph(graph_distr(possible_constraints()))
        elif 'bottleneck1' in self.level_name:
            return seeded_rand_graph(graph_distr(bottleneck1_constraints()))
        elif 'bottleneck2' in self.level_name:
            return seeded_rand_graph(graph_distr(bottleneck2_constraints()))
        elif 'bottleneck3' in self.level_name:
            return seeded_rand_graph(graph_distr(bottleneck3_constraints()))
        else:
            return seeded_rand_graph(graph_distr(no_bottleneck_constraints()))


def get_symbolic_alchemy_level(
    level_name: str, 
    observe_used: bool = True, 
    end_trial_action: bool = False,
    num_trials: int = 10, 
    num_stones_per_trial: int = 3, 
    num_potions_per_trial: int = 12, 
    seed: int = None,
    reward_weights: stones_and_potions.RewardWeights = None, 
    max_steps_per_trial: int = DEFAULT_MAX_STEPS_PER_TRIAL,
    see_chemistries: Optional[Mapping[str, type_utils.ChemistrySeen]] = None, 
    generate_events: bool = False
):
    """Gets a symbolic alchemy instance of the level passed in.
    
    Args:
        level_name: Name of the level to get.
        observe_used: Whether to have a feature for each item slot which is 
            set to 1 if the item is used and 0 otherwise.
        end_trial_action: Whether the agent has an action to end the trial early.
        num_trials: The number of trials in each episode.
        num_stones_per_trial: The number of stones in each trial.
        num_potions_per_trial: The number of potions in each trial.
        seed: The random seed to use.
        reward_weights: Structure which tells us the reward for a given stone.
        max_steps_per_trial: The number of steps the agent can take before the
            trial is automatically ended.
        see_chemistries: Optional map from name to a structure containing
            information about how to form a chemistry, i.e. which parts and whether
            the content should be ground truth or the belief state. These are added
            to the observation dictionary. If None, then no chemistries are added.
        generate_events: Whether to track items generated and potions and stones
            used and return this information when events is called on the
            environment. This is not necessary during training but is used when we
            run analysis on the environment.
    """
    random_state = np.random.RandomState(seed)

    if 'perceptual_mapping_randomized' in level_name:
        _, index_to_perm_index = precomputed_maps.get_perm_index_conversion()
        stone_map_gen = StoneMapGen(random_state)
        potion_map_gen = PotionMapGen(random_state, index_to_perm_index)
    else:
        stone_map_gen = stones_and_potions.all_fixed_stone_map
        potion_map_gen = stones_and_potions.all_fixed_potion_map

    graph_gen = GraphGen(random_state=random_state, level_name=level_name)
    rotation_gen = RotationGen(random_state=random_state, level_name=level_name)
    items_gen = ItemsGen(random_state, num_stones_per_trial, num_potions_per_trial)
    chemistry_gen = ChemistryGen(potion_map_gen, stone_map_gen, graph_gen, rotation_gen)

    return symbolic_alchemy.SymbolicAlchemy(
        observe_used=observe_used,
        chemistry_gen=chemistry_gen,
        reward_weights=reward_weights,
        items_gen=items_gen,
        num_trials=num_trials,
        end_trial_action=end_trial_action,
        max_steps_per_trial=max_steps_per_trial,
        see_chemistries=see_chemistries,
        generate_events=generate_events)


def get_symbolic_alchemy_fixed(
    episode_items, chemistry, observe_used=True, reward_weights=None,
    end_trial_action=False, max_steps_per_trial=DEFAULT_MAX_STEPS_PER_TRIAL,
    see_chemistries=None, generate_events=False):
    """Symbolic alchemy which generates same chemistry and items every episode."""

    return symbolic_alchemy.SymbolicAlchemy(
        observe_used=observe_used,
        chemistry_gen=lambda: chemistry,
        reward_weights=reward_weights,
        items_gen=lambda i: episode_items.trials[i],
        num_trials=episode_items.num_trials,
        end_trial_action=end_trial_action,
        max_steps_per_trial=max_steps_per_trial,
        see_chemistries=see_chemistries,
        generate_events=generate_events)