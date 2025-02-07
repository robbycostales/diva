from argparse import Namespace

import numpy as np

from diva.environments.alchemy.alchemy_gym import (
    LATENT_POTION_LIST,
    MAX_POTIONS,
    MAX_STONES,
    potion_to_latent_potion,
)
from diva.environments.alchemy.genotypes.helpers import (
    GRAPH_LIST,
    IDX_TO_PERMUTATION,
    PERMUTATION_TO_IDX,
)
from dm_alchemy.types import utils as type_utils
from dm_alchemy.types.stones_and_potions import (
    LatentStone,
    PotionMap,
    StoneMap,
    latent_potion_from_index,
    possible_rotations,
)

Chemistry = type_utils.Chemistry
TrialItems = type_utils.TrialItems
EpisodeItems = type_utils.EpisodeItems


class DefaultGenotype(object):
    """
    Structure of default genotype:
    - (1) num_stones: number of stones (1-MAX_STONES)
    - (1) num_potions: number of potions (1-MAX_POTIONS)
    - (3 * MAX_STONES) stone_latent_states: latent state of each stone
    - (1) stone_rotation: rotation of stones
    - (3) stone_reflection: reflection of stones
    - (MAX_POTIONS) potion_effects: effects of potions
    - (3) potion_reflection: reflection of potions
    - (1) potion_permutation: permutation of potions
    - (1) graph_topology: topology of graph, expressed as an index from 0-108
    """
    def __init__(self):
        self.genotype_size = 11 + MAX_STONES*3 + MAX_POTIONS
        self.genotype_lower_bounds = np.concatenate(
            [[1],                   # num stones
             [1],                   # num potions
             [0] * MAX_STONES * 3,  # stone latent states
             [0],                   # stone rotation
             [0] * 3,               # stone reflection
             [0] * MAX_POTIONS,     # potion effects
             [0] * 3,               # potion reflection
             [0],                   # potion permutation
             [0]],                  # graph topology
            axis=0)
        self.genotype_upper_bounds = np.concatenate(
            [[MAX_STONES],          # num stones
             [MAX_POTIONS],         # num potions
             [1] * MAX_STONES * 3,  # stone latent states
             [3],                   # stone rotation
             [1] * 3,               # stone reflection
             [5] * MAX_POTIONS,     # potion effects
             [1] * 3,               # potion reflection
             [5],                   # potion permutation
             [108]],                # graph topology
            axis=0)
        self.genotype_bounds = [(l, u) for l, u in  # noqa: E741
                                zip(list(self.genotype_lower_bounds),
                                    list(self.genotype_upper_bounds))]
        
    @property
    def genotype_info(self):
        return Namespace(
            genotype_size=self.genotype_size,
            genotype_lower_bounds=self.genotype_lower_bounds,
            genotype_upper_bounds=self.genotype_upper_bounds,
            genotype_bounds=self.genotype_bounds,
        )
    
    def process_genotype(self, genotype, size=None):
        gisn = genotype is None

        if size is None:
            size = self.genotype_size

        if gisn:
            # Raw information from genotype
            num_stones, num_potions = None, None
            stone_latent_states, stone_rotation, stone_reflection = None, None, None
            potion_effects, potion_reflection, potion_permutation = None, None, None
            graph_topology_idx = None
        else:
            genotype = np.array(genotype).astype(int)
            # Get raw information from genotype
            num_stones = genotype[0]
            num_potions = genotype[1]
            idx = 2
            stone_latent_states = genotype[idx : idx + MAX_STONES * 3]; idx += MAX_STONES * 3  # noqa: E702
            stone_rotation = genotype[idx : idx + 1]; idx += 1  # noqa: E702
            stone_reflection = genotype[idx : idx + 3]; idx += 3  # noqa: E702
            potion_effects = genotype[idx : idx + MAX_POTIONS]; idx += MAX_POTIONS  # noqa: E702
            potion_reflection = genotype[idx : idx + 3]; idx += 3  # noqa: E702
            potion_permutation = genotype[idx : idx + 1]; idx += 1  # noqa: E702
            graph_topology_idx = genotype[idx : idx + 1]; idx += 1  # noqa: E702
            
            # Get processed information from genotype
            # From {0, 1} to {-1, 1}
            p_stone_reflection = 2 * stone_reflection - 1
            # From {0, 1} to {-1, 1}
            p_potion_reflection = 2 * potion_reflection - 1
            # From {0, 1, 2, 3, 4, 5} to permutation list
            p_potion_permutation = list(IDX_TO_PERMUTATION[potion_permutation[0]])
            # From flat latent stones to list of latent stone latent values
            p_stone_latent_coords = 2 * np.reshape(stone_latent_states, (MAX_STONES, 3)) - 1

            # Get chemistry variables
            # From [[latent1], [latent2], ...] to LatentStone objects
            c_stones = [LatentStone(coords) for coords in p_stone_latent_coords[:num_stones]]
            # From {0, 1, 2, 3, 4, 5} to LatentPotion objects
            c_potions = [latent_potion_from_index(potion_idx) for potion_idx in potion_effects[:num_potions]]
            # From {-1, 1}^3 to StoneMap object
            c_stone_map = StoneMap(pos_dir=p_stone_reflection)
            # From perm/reflection to PotionMap object
            c_potion_map = PotionMap(dim_map=p_potion_permutation, dir_map=p_potion_reflection)
            # From {0, 1, 2, 3} to {I, Rx45, Ry45, Rz45}
            c_rotation = possible_rotations()[stone_rotation[0]]
            # From index {0, ..., 108} to graph topology
            c_graph = GRAPH_LIST[graph_topology_idx[0]]
        
        # Return processed genotype
        processed_genotype = {
            # Raw info
            'num_stones': num_stones,
            'num_potions': num_potions,
            'stone_latent_states': stone_latent_states,
            'stone_rotation': stone_rotation,
            'stone_reflection': stone_reflection,
            'potion_effects': potion_effects,
            'potion_effects_raw': [LATENT_POTION_LIST[i] for i in potion_effects],
            'potion_reflection': potion_reflection,
            'potion_permutation': potion_permutation,
            'graph_topology_idx': graph_topology_idx,
            # Processed values
            'p_stone_latent_coords': p_stone_latent_coords,
            # Chemistry variables
            'c_stones': c_stones,
            'c_potions': c_potions,
            'c_stone_map': c_stone_map,
            'c_potion_map': c_potion_map,
            'c_rotation': c_rotation,
            'c_graph': c_graph,
            # Genotype info
            'genotype': genotype,
            'genotype_size': size,
            'genotype_lower_bounds': self.genotype_lower_bounds,
            'genotype_upper_bounds': self.genotype_upper_bounds,
            'genotype_bounds': self.genotype_bounds,
        }
        return Namespace(**processed_genotype)
    
    def set_rng(self, rng):
        self.rng = rng
    
    def genotype_from_seed(self):
        rng = self.rng

        ###  Construuct genotype
        # 1. Decide on number of stones and number of potions
        num_stones = rng.integers(1, MAX_STONES + 1)
        num_potions = rng.integers(1, MAX_POTIONS + 1)
        # 2. Decide on initial stone latent state, c for each stone;
        #    c in { -1, 1 }^3  (representing corners of cube)
        # We produce a flattened vector of length 3*MAX_STONES
        # NOTE: when num_stones < MAX_STONES, then we will have useless DNA
        # NOTE: 0, 1 -> -1, 1
        stone_latent_states = rng.choice([0, 1], size=(MAX_STONES*3,))
        # 3. Decide on stone rotation for all stones, where
        #    S_rotate = U({I, Rx45, Ry45, Rz45)})
        stone_rotation = rng.integers(0, 4, size=(1,))
        # 4. Decide on stone reflection for each stone, where 
        #    S_reflect = diag(s) for s ~ U({-1, 1}^3)
        # NOTE: 0, 1 -> -1, 1
        stone_reflection = rng.choice([0, 1], size=(3,))
        # 5. Decide on potion effect for each potion, where
        #    p in P, where P := {+-e^(0), +-e^(1), +-e^(2)}  (so 6 options)
        #    where e^(i) is the ith standard basis vector; note, for
        #    ordering, we do all positives first, then negatives
        potion_effects = rng.integers(0, 6, size=(MAX_POTIONS,))
        # 6. Decide on potion reflection for all potions (same as S_reflect)
        # NOTE: 0, 1 -> -1, 1
        potion_reflection = rng.choice([0, 1], size=(3,))
        # 7. Decide on potion permutation matrix for each potion; 
        #    note there are six possible permutations, so we can just use a 
        #    single integer to represent the permutation:
        #    0 -> (0, 1, 2); 1 -> (0, 2, 1); 2 -> (1, 0, 2); 
        #    3 -> (1, 2, 0); 4 -> (2, 0, 1); 5 -> (2, 1, 0)
        potion_permutation = rng.integers(0, 6, size=(1,))
        # 8. Decide on graph adjacency matrix; there are 109 possible
        #    graphs, so we use a single integer to represent the graph
        #    topology
        #    NOTE: from dm_alchemy/types/graphs.py>graph_distr:
        #    Equal probability is given to the set of graphs with each 
        #    valid number of constraints. In practice this means:
        #       1/4 probability for the constraint 0 case
        #       (1/4)*(1/12) probability for each case with one constraint (12 cases)
        #       (1/4)*(1/48) probability for each case with two constraints (48 cases)
        #       (1/4)*(1/48) probability for each case with three constraints (48 cases)
        graph_topology = rng.integers(0, 109, size=(1,))

        # Create genotype from these components by concatenating them
        genotype = np.concatenate([num_stones, num_potions,
            stone_latent_states, stone_rotation, stone_reflection,
            potion_effects, potion_reflection, potion_permutation,
            graph_topology
        ], axis=0)

        # Check if solution is valid
        pg = self.process_genotype(genotype)
        valid, reason = self.is_valid_genotype(pg)

        return pg, valid, reason
    
    def get_genotype_from_task_info(
            self,
            c_stone_map,
            c_potion_map,
            c_rotation,
            c_graph,
            c_stones,
            c_potions):

        # Convert these back to their genotype representations
        num_stones = len(c_stones) 
        num_potions = len(c_potions)
        stone_latent_states = (np.array([s.latent for s in c_stones]).flatten() + 1) / 2
        pad_length = MAX_STONES * 3 - len(stone_latent_states)
        stone_latent_states = np.pad(stone_latent_states, (0, pad_length), 'constant')
        stone_rotation = next((i for i, rotation in enumerate(possible_rotations()) 
                               if np.array_equal(c_rotation, rotation)), None)
        stone_reflection = (np.array(c_stone_map.latent_pos_dir) + 1) / 2
        potion_effects = np.array([LATENT_POTION_LIST.index(
                                                potion_to_latent_potion(potion)) 
                                   for potion in c_potions])
        pad_length = MAX_POTIONS - len(potion_effects)
        potion_effects = np.pad(potion_effects, (0, pad_length), 'constant')
        potion_reflection = (np.array(c_potion_map.dir_map) + 1) / 2
        potion_permutation = PERMUTATION_TO_IDX[tuple(c_potion_map.dim_map)]
        graph_topology = GRAPH_LIST.index(c_graph)

        # Now concatenate all these into a genotype
        genotype = [num_stones, num_potions]
        genotype.extend(stone_latent_states)
        genotype.extend([stone_rotation])
        genotype.extend(stone_reflection)
        genotype.extend(potion_effects)
        genotype.extend(potion_reflection)
        genotype.extend([potion_permutation])
        genotype.extend([graph_topology])

        genotype = np.array(genotype).astype(int)

        return genotype
        
    
    def is_valid_genotype(self, pg):
        # Check if genotype within bounds
        if pg.genotype is None:
            return False, 'none_genotype'
        if np.any(pg.genotype < pg.genotype_lower_bounds):
            return False, 'lower_bound_violation'
        if np.any(pg.genotype > pg.genotype_upper_bounds):
            return False, 'upper_bound_violation'
        # Otherwise, genotype will be valid!
        return True, None