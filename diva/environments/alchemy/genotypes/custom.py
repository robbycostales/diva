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
from dm_alchemy.types.graphs import (
    constraint_from_graph,
    create_graph_from_constraint,
    no_bottleneck_constraints,
)
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


CONSTRAINT_INT_TO_STR = {
    -1: '-1',
    0: '*',
    1: '1',
}
CONSTRAINT_STR_TO_INT = {v: k for k, v in CONSTRAINT_INT_TO_STR.items()}


class CustomGenotype(object):
    """
    High-level structure of custom genotype:
    (a) Stone specifications (number of stones, latent states)
    (b) Potion specifications (number of potions, effects)
    (c) Stone chemistry (rotation, reflection)
    (d) Potion chemistry (reflection, permutation)
    (e) Graph topology; i.e. transition chemistry

    Each gt_type string has a form of 'a-b-c-d-e', where each letter
    corresponds to the five high-level structures above. Each subcomponent
    of the genotype accepts a certain set of strings. See code for details.

    Default default value is 'd-d-d-d-d', which means default.
    """
    def __init__(self, 
                 gt_type='d-d-d-d-d',
                 use_dynamic_items=True,
                 num_trials=10):
        """
        Args:
            gt_type (str): Genotype type
            use_dynamic_items (bool): Whether to use dynamic stones
            num_trials (int): Number of trials (only used for dynamic stones)
        """
        # Split genotype type into subcomponents and handle each separately
        gt_scs = gt_type.split('-')
        self.use_dynamic_items = use_dynamic_items
        self.gt_scs = gt_scs
        genotype_size = 0

        # Deal with dynamic/fixed stone specs
        if self.use_dynamic_items:
            self.num_item_specs = num_trials
        else:
            self.num_item_specs = 1  # I.e. fixed case
        
        # (a) Stone specifications
        if gt_scs[0] == 'd':
            # Designed for simplicity and so that all values are valid
            # 1. Num stones as integer
            # 2. Latent states as {0, 1}^3 (padded to length MAX_STONES * 3)
            genotype_size += (self.num_item_specs) + (MAX_STONES * 3 * self.num_item_specs)
            genotype_lower_bounds = np.concatenate(
                [[1] * self.num_item_specs,
                [0] * MAX_STONES * 3 * self.num_item_specs], axis=0)
            genotype_upper_bounds = np.concatenate(
                [[MAX_STONES] * self.num_item_specs, 
                [1] * MAX_STONES * 3 * self.num_item_specs], axis=0)
        elif 's' in gt_scs[0]:
            n = int(gt_scs[0][1:])
            # Designed to make it difficult to generate diverse stones
            # 1. Num stones as integer
            # 2. Latent states as {0, 1}^{3*n} (padded to length MAX_STONES * 3)
            genotype_size += (self.num_item_specs) + (n * 3 * MAX_STONES * self.num_item_specs)
            genotype_lower_bounds = np.concatenate(
                [[1] * self.num_item_specs,
                [0] * n * 3 * MAX_STONES * self.num_item_specs], axis=0)
            genotype_upper_bounds = np.concatenate(
                [[MAX_STONES] * self.num_item_specs, 
                [1] * n * 3 * MAX_STONES * self.num_item_specs], axis=0)
        else:
            raise ValueError('Invalid gt_sc for stone specifications: {}'.format(gt_scs[0]))

        # (b) Potion specifications
        if gt_scs[1] == 'd':
            # Designed for simplicity and so that all values are valid
            # 1. Num potions as integer
            # 2. Potion effects as {0, 1, 2, 3, 4, 5} (padded to length MAX_POTIONS)
            genotype_size += (self.num_item_specs) + (MAX_POTIONS * self.num_item_specs)
            genotype_lower_bounds = np.concatenate(
                [genotype_lower_bounds, 
                 [1] * self.num_item_specs, 
                 [0] * MAX_POTIONS * self.num_item_specs], axis=0)
            genotype_upper_bounds = np.concatenate(
                [genotype_upper_bounds, 
                 [MAX_POTIONS] * self.num_item_specs, 
                 [5] * MAX_POTIONS * self.num_item_specs], axis=0)
        else:
            raise ValueError('Invalid gt_sc for potion specifications: {}'.format(gt_scs[1]))
                               
        # (c) Stone chemistry
        if gt_scs[2] == 'd':
            # Designed for simplicity and so that all values are valid
            # 1. Stone rotation as integer {0, 1, 2, 3}
            # 2. Stone reflection as {0, 1}^3 (applies to all stones)
            genotype_size += 1 + 3
            genotype_lower_bounds = np.concatenate(
                [genotype_lower_bounds, 
                 [0], 
                 [0] * 3], axis=0)
            genotype_upper_bounds = np.concatenate(
                [genotype_upper_bounds, 
                 [3], 
                 [1] * 3], axis=0)
        else:
            raise ValueError('Invalid gt_sc for stone chemistry: {}'.format(gt_scs[2]))

        # (d) Potion chemistry
        if gt_scs[3] == 'd':
            # Designed for simplicity and so that all values are valid
            # 1. Potion reflection as {0, 1}^3 (applies to all potions)
            # 2. Potion permutation as integer {0, 1, 2, 3, 4, 5}
            genotype_size += 1 + 3
            genotype_lower_bounds = np.concatenate(
                [genotype_lower_bounds, 
                 [0] * 3, 
                 [0]], axis=0)
            genotype_upper_bounds = np.concatenate(
                [genotype_upper_bounds, 
                 [1] * 3, 
                 [5]], axis=0)
        else:
            raise ValueError('Invalid gt_sc for potion chemistry: {}'.format(gt_scs[3]))

        # (e) Graph topology
        if gt_scs[4] == 'd':
            # Designed for simplicity and so that all values are valid
            # NOTE: We are removing lots of structural knowledge with this
            # simplification---and making it easy to generate diversity
            # 1. Graph topology as integer {0, ..., 108}
            genotype_size += 1
            genotype_lower_bounds = np.concatenate(
                [genotype_lower_bounds, 
                 [0]], axis=0)
            genotype_upper_bounds = np.concatenate(
                [genotype_upper_bounds, 
                 [108]], axis=0)
        elif gt_scs[4] == 'c6':
            # Designed so that each 6 slots (non-diagonal entries in the 3x3
            # grid) specifying constraints are specified directly as {-1, 0, 1}.
            # NOTE: We can generate invalid graphs with this genotype---in the
            # sense that we will have more bottlenecks than in the downstream
            # distribution. 
            # 1. Graph topology constraints as {-1, 0, 1}^6
            genotype_size += 6
            genotype_lower_bounds = np.concatenate(
                [genotype_lower_bounds, 
                 [-1] * 6], axis=0)
            genotype_upper_bounds = np.concatenate(
                [genotype_upper_bounds, 
                 [1] * 6], axis=0)
        elif gt_scs[4] == 'c9':
            # Designed so that each 9 slots. First three slots indicate
            # if we're using the constraint specification (binary), the next
            # three indicate the position of the constraint (index 0-5), and
            # the last three indicate the constraint value {-1, 0, 1}.
            # NOTE: All genotypes under this specification are valid, but
            # we introduce one-to-many (since unactive/redudant constraints are
            # ignored).
            # 1. Graph topology constraints as {0, 1}^3 x {0, ..., 5}^3 x {-1, 0, 1}^3
            genotype_size += 9
            genotype_lower_bounds = np.concatenate(
                [genotype_lower_bounds, 
                 [0] * 3, 
                 [0] * 3, 
                 [-1] * 3], axis=0)
            genotype_upper_bounds = np.concatenate(
                [genotype_upper_bounds, 
                 [1] * 3, 
                 [5] * 3, 
                 [1] * 3], axis=0)
        else:
            raise ValueError('Invalid gt_sc for graph topology: {}'.format(gt_scs[4]))

        self.genotype_size = genotype_size
        self.genotype_lower_bounds = genotype_lower_bounds
        self.genotype_upper_bounds = genotype_upper_bounds
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
            c_stones, c_potions, c_stone_map, c_potion_map, c_rotation, c_graph \
                = None, None, None, None, None, None
            graph_topology_idx = None
        else:
            genotype = np.array(genotype).astype(int)
            # Get raw information from genotype

            idx = 0
            # (a) Stone specifications
            if self.gt_scs[0] == 'd':
                num_stones = []
                for _ in range(self.num_item_specs):
                    num_stones.append(genotype[idx]); idx += 1  # noqa: E702
                stone_latent_states = genotype[idx : idx + MAX_STONES * 3 * self.num_item_specs]
                idx += MAX_STONES * 3 * self.num_item_specs
            elif 's' in self.gt_scs[0]:
                n = int(self.gt_scs[0][1:])  # How many bits for each stone
                num_stones = []
                for _ in range(self.num_item_specs):
                    num_stones.append(genotype[idx]); idx += 1  # noqa: E702
                raw_stone_latent_states = genotype[idx: idx + n * 3 * MAX_STONES * self.num_item_specs] 
                idx += n * 3 * MAX_STONES * self.num_item_specs
                # Reshape to 3d array of shape: (num_stones, 3, n)
                raw_stone_latent_states = np.reshape(raw_stone_latent_states, (self.num_item_specs * MAX_STONES, 3, n))
                # There are n bits for each stone latent state; we want to set
                # the latent state to '0' only if there are 2 or fewer '1' bits.
                stone_latent_states = np.sum(raw_stone_latent_states, axis=2)
                stone_latent_states = (stone_latent_states >= 2).astype(int) 
                # Flatten to 1d array
                stone_latent_states = stone_latent_states.flatten()
            
            # (b) Potion specifications
            if self.gt_scs[1] == 'd':
                num_potions = []
                for _ in range(self.num_item_specs):
                    num_potions.append(genotype[idx]); idx += 1  # noqa: E702
                potion_effects = genotype[idx : idx + MAX_POTIONS * self.num_item_specs]
                idx += MAX_POTIONS * self.num_item_specs
                potion_effects = np.reshape(potion_effects, (self.num_item_specs, MAX_POTIONS))
            
            # (c) Stone chemistry
            if self.gt_scs[2] == 'd':
                stone_rotation = genotype[idx : idx + 1]; idx += 1  # noqa: E702
                stone_reflection = genotype[idx : idx + 3]; idx += 3  # noqa: E702
            
            # (d) Potion chemistry
            if self.gt_scs[3] == 'd':
                potion_reflection = genotype[idx : idx + 3]; idx += 3  # noqa: E702
                potion_permutation = genotype[idx : idx + 1]; idx += 1  # noqa: E702
            
            # (e) Graph topology
            if self.gt_scs[4] == 'd':  
                graph_topology_idx = genotype[idx : idx + 1]; idx += 1  # noqa: E702
            elif self.gt_scs[4] == 'c6':
                # Get constraints from genotype
                constraints = no_bottleneck_constraints()  # Initialize with no constraints
                # Flatten constraints 2d list
                constraints = np.array(constraints).flatten()
                indices = [1, 2,   3, 5,   6, 7]  # Non-diagonal indices in 3x3 grid
                # Use indices to get constraints from genotype
                for i in indices:
                    constraints[i] = CONSTRAINT_INT_TO_STR[genotype[idx]]; idx += 1  # noqa: E702
                # Reshape constraints to 3x3 grid and convert to list
                constraints = np.reshape(constraints, (3, 3)).tolist()
                # Initialize graph object from constraints
                graph = create_graph_from_constraint(constraints)
                # Get graph topology index from graph object by
                # checking for equality (__eq__ is implemented for graphs
                # so that they are equal if they share the same constraints)
                
                # For efficiency, no need to compute graph topology index
                # try:
                #     graph_topology_idx = [GRAPH_LIST.index(graph)]
                # except ValueError:
                #     graph_topology_idx = None
                graph_topology_idx = None
            elif self.gt_scs[4] == 'c9':
                # Get constraints from genotype
                constraints = no_bottleneck_constraints()  # Initialize with no constraints
                # Flatten constraints 2d list
                constraints = np.array(constraints).flatten()
                indices = [1, 2,   3, 5,   6, 7]  # Non-diagonal indices in 3x3 grid
                for i in range(3):  # For each potential constraint
                    # Get binary indicator for whether constraint is active
                    active = genotype[idx + 0 + i]
                    # Get index of constraint
                    constraint_idx = genotype[idx + 3 + i]
                    constraint_idx = indices[constraint_idx]  # Get actual index
                    # Get constraint value
                    constraint_val = CONSTRAINT_INT_TO_STR[genotype[idx + 6 + i]]
                    # If constraint is active, then set it
                    if active:
                        constraints[constraint_idx] = constraint_val
                # Reshape constraints to 3x3 grid and convert to list
                constraints = np.reshape(constraints, (3, 3)).tolist()
                # Initialize graph object from constraints
                graph = create_graph_from_constraint(constraints)
                # Get graph topology index from graph object by
                # checking for equality (__eq__ is implemented for graphs
                # so that they are equal if they share the same constraints)

                # For efficiency, no need to compute graph topology index
                # try:
                #     graph_topology_idx = [GRAPH_LIST.index(graph)]
                # except ValueError:
                #     # Unlike for c6, for c9 we should always have a valid graph
                #     raise ValueError('Invalid graph topology: {}'.format(graph))
                graph_topology_idx = None
            
            # Get processed information from genotype
            # From {0, 1} to {-1, 1}
            p_stone_reflection = 2 * stone_reflection - 1
            # From {0, 1} to {-1, 1}
            p_potion_reflection = 2 * potion_reflection - 1
            # From {0, 1, 2, 3, 4, 5} to permutation list
            p_potion_permutation = list(IDX_TO_PERMUTATION[potion_permutation[0]])
            # From flat latent stones to list of latent stone latent values (padded)
            p_stone_latent_coords = 2 * np.reshape(stone_latent_states, (self.num_item_specs, MAX_STONES, 3)) - 1

            # Get chemistry variables
            # From [[latent1], [latent2], ...] to LatentStone objects
            c_stones = []
            for i, ns in enumerate(num_stones):
                c_stones.append([LatentStone(coords) for coords in p_stone_latent_coords[i][:ns]])
            # From {0, 1, 2, 3, 4, 5} to LatentPotion objects
            c_potions = []
            for i, npo in enumerate(num_potions):
                c_potions.append([latent_potion_from_index(potion_idx) for potion_idx in potion_effects[i][:npo]])
            # From {-1, 1}^3 to StoneMap object
            c_stone_map = StoneMap(pos_dir=p_stone_reflection)
            # From perm/reflection to PotionMap object
            c_potion_map = PotionMap(dim_map=p_potion_permutation, dir_map=p_potion_reflection)
            # From {0, 1, 2, 3} to {I, Rx45, Ry45, Rz45}
            c_rotation = possible_rotations()[stone_rotation[0]]
            # From index {0, ..., 108} to graph topology
            if graph_topology_idx is None:
                c_graph = graph
            else:
                c_graph = GRAPH_LIST[graph_topology_idx[0]]
        
        # Return processed genotype
        processed_genotype = {
            # Raw info
            'num_stones': num_stones,
            'num_potions': num_potions,
            'stone_latent_states': np.reshape(stone_latent_states, (self.num_item_specs, MAX_STONES, 3)),
            'stone_rotation': stone_rotation,
            'stone_reflection': stone_reflection,
            'potion_effects': potion_effects,
            'potion_effects_raw': [[LATENT_POTION_LIST[i] for i in pe] for pe in potion_effects],
            'potion_reflection': potion_reflection,
            'potion_permutation': potion_permutation,
            'graph_topology_idx': graph_topology_idx,
            # Processed values
            'p_stone_latent_coords': p_stone_latent_coords,
            # Chemistry variables
            'c_stones': c_stones,  # Now a list, for each trial
            'c_potions': c_potions,  # Now a list, for each trial
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

        genotype = []

        # (a) Stone specifications
        if self.gt_scs[0] == 'd':
            # Designed for simplicity and so that all values are valid
            # 1. Num stones as integer
            # 2. Latent states as {0, 1}^3 (padded to length MAX_STONES * 3)
            num_stones = rng.integers(1, MAX_STONES + 1, size=(self.num_item_specs,))
            genotype.append(num_stones)
            
            # Decide on initial stone latent state, c for each stone;
            # c in { -1, 1 }^3  (representing corners of cube)
            # We produce a flattened vector of length 3*MAX_STONES
            # NOTE: when num_stones < MAX_STONES, then we will have useless DNA
            # NOTE: 0, 1 -> -1, 1
            stone_latent_states = rng.choice([0, 1], size=(self.num_item_specs*MAX_STONES*3,))
            genotype.append(stone_latent_states)
        elif 's' in self.gt_scs[0]:
            n = int(self.gt_scs[0][1:])
            # Designed to make it difficult to generate diverse stones
            # 1. Num stones as integer
            # 2. Latent states as {0, 1}^{3*n} (padded to length MAX_STONES * 3 * n)
            num_stones = rng.integers(1, MAX_STONES + 1, size=(self.num_item_specs,))
            genotype.append(num_stones)

            # Decide on initial stone latent state, c for each stone;
            # c in { -1, 1 }^3  (representing corners of cube)
            # We produce a flattened vector of length 3*MAX_STONES
            # NOTE: when num_stones < MAX_STONES, then we will have useless DNA
            # NOTE: 0, 1 -> -1, 1
            stone_latent_states = rng.choice([0, 1], size=(self.num_item_specs*MAX_STONES*3,))
            # Extend to size 3*n*MAX_STONES
            raw_stone_latent_states = np.tile(stone_latent_states, n)
            raw_stone_latent_states = np.reshape(raw_stone_latent_states, (self.num_item_specs*MAX_STONES, 3, n))
            raw_stone_latent_states = raw_stone_latent_states.flatten()
            # NOTE: As-is, this is a valid representation (since zeros will be
            # all zeros, and ones will be all ones)
            genotype.append(raw_stone_latent_states)

        # (b) Potion specifications
        if self.gt_scs[1] == 'd':
            # Designed for simplicity and so that all values are valid
            # 1. Num potions as integer
            # 2. Potion effects as {0, 1, 2, 3, 4, 5} (padded to length MAX_POTIONS)
            num_potions = rng.integers(1, MAX_POTIONS + 1, size=(self.num_item_specs,))
            genotype.append(num_potions)
            
            # Decide on potion effect for each potion, where
            # p in P, where P := {+-e^(0), +-e^(1), +-e^(2)}  (so 6 options)
            # where e^(i) is the ith standard basis vector; note, for
            # ordering, we do all positives first, then negatives
            potion_effects = rng.integers(0, 6, size=(self.num_item_specs*MAX_POTIONS,))
            genotype.append(potion_effects)
        
        # (c) Stone chemistry
        if self.gt_scs[2] == 'd':
            # Designed for simplicity and so that all values are valid
            # 1. Stone rotation as integer {0, 1, 2, 3}
            # 2. Stone reflection as {0, 1}^3 (applies to all stones)
            
            # Decide on stone rotation for all stones, where
            # S_rotate = U({I, Rx45, Ry45, Rz45)})
            stone_rotation = rng.integers(0, 4, size=(1,))
            genotype.append(stone_rotation)
            # Decide on stone reflection for each stone, where 
            # S_reflect = diag(s) for s ~ U({-1, 1}^3)
            # NOTE: 0, 1 -> -1, 1
            stone_reflection = rng.choice([0, 1], size=(3,))
            genotype.append(stone_reflection)
        
        # (d) Potion chemistry
        if self.gt_scs[3] == 'd':
            # Designed for simplicity and so that all values are valid
            # 1. Potion reflection as {0, 1}^3 (applies to all potions)
            # 2. Potion permutation as integer {0, 1, 2, 3, 4, 5}

            # Decide on potion reflection for all potions (same as S_reflect)
            # NOTE: 0, 1 -> -1, 1
            potion_reflection = rng.choice([0, 1], size=(3,))
            genotype.append(potion_reflection)

            # Decide on potion permutation matrix for each potion; 
            # note there are six possible permutations, so we can just use a 
            # single integer to represent the permutation:
            # 0 -> (0, 1, 2); 1 -> (0, 2, 1); 2 -> (1, 0, 2); 
            # 3 -> (1, 2, 0); 4 -> (2, 0, 1); 5 -> (2, 1, 0)
            potion_permutation = rng.integers(0, 6, size=(1,))
            genotype.append(potion_permutation)
        
        # (e) Graph topology
        if self.gt_scs[4] == 'd':
            # Designed for simplicity and so that all values are valid
            # NOTE: We are removing lots of structural knowledge with this
            # simplification---and making it easy to generate diversity
            # 1. Graph topology as integer {0, ..., 108}

            # Decide on graph adjacency matrix; there are 109 possible
            # graphs, so we use a single integer to represent the graph
            # topology
            graph_topology_idx = rng.integers(0, 109, size=(1,))
            genotype.append(graph_topology_idx)
        elif self.gt_scs[4] == 'c6':
            # Designed so that each 6 slots (non-diagonal entries in the 3x3
            # grid) specifying constraints are specified directly as {-1, 0, 1}.
            # NOTE: We can generate invalid graphs with this genotype---in the
            # sense that we will have more bottlenecks than in the downstream
            # distribution. 
            # 1. Graph topology constraints as {-1, 0, 1}^6

            # Decide on graph adjacency matrix constraints; there are 3^6
            # possible constraints, so we use a single integer to represent
            # the constraints
            constraints = rng.integers(-1, 2, size=(6,))
            genotype.append(constraints)
        elif self.gt_scs[4] == 'c9':
            # Designed so that each 9 slots. First three slots indicate
            # if we're using the constraint specification (binary), the next
            # three indicate the position of the constraint (index 0-5), and
            # the last three indicate the constraint value {-1, 0, 1}.
            # NOTE: All genotypes under this specification are valid, but
            # we introduce one-to-many (since unactive/redudant constraints are
            # ignored).
            # 1. Graph topology constraints as {0, 1}^3 x {0, ..., 5}^3 x {-1, 0, 1}^3
            active = rng.choice([0, 1], size=(3,))
            constraint_idx = rng.integers(0, 6, size=(3,))
            constraint_val = rng.integers(-1, 2, size=(3,))
            genotype.append(active)
            genotype.append(constraint_idx)
            genotype.append(constraint_val)

        # Create genotype from these components by concatenating them
        genotype = np.concatenate(genotype, axis=0)

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
        """
        Args:
            c_stone_map: StoneMap
            c_potion_map: PotionMap
            c_rotation: np.ndarray
            c_graph: Graph
            c_stones: List[List[LatentStone]] where outer list is trials (i.e.
                to handle dynamic case) and inner list is the list of stones
                either in that specific trial, or for static stone case, the
                stones used across all trials
            c_potions: List[List[LatentPotion]] " " for potions
        """

        # Convert these back to their genotype representations
        num_stones = [len(cs) for cs in c_stones] 
        num_potions = [len(cp) for cp in c_potions]
        
        assert len(num_stones) == len(num_potions) == self.num_item_specs

        stone_latent_states = []
        for cs in c_stones:
            sls = (np.array([s.latent for s in cs]).flatten() + 1) / 2
            pad_length = MAX_STONES * 3 - len(sls)
            sls = np.pad(sls, (0, pad_length), 'constant')
            stone_latent_states.append(sls)
        
        stone_rotation = next((i for i, rotation in enumerate(possible_rotations()) 
                               if np.array_equal(c_rotation, rotation)), None)
        stone_reflection = (np.array(c_stone_map.latent_pos_dir) + 1) / 2
        
        potion_effects = []
        for cp in c_potions:
            pe = np.array([LATENT_POTION_LIST.index(potion_to_latent_potion(potion)) 
                           for potion in cp])
            pad_length = MAX_POTIONS - len(pe)
            pe = np.pad(pe, (0, pad_length), 'constant')
            potion_effects.append(pe)
        
        potion_reflection = (np.array(c_potion_map.dir_map) + 1) / 2
        potion_permutation = PERMUTATION_TO_IDX[tuple(c_potion_map.dim_map)]
        graph_topology = GRAPH_LIST.index(c_graph)
        graph_constraints = constraint_from_graph(c_graph)

        # Now concatenate all these into a genotype

        genotype = []

        # (a) Stone specifications
        if self.gt_scs[0] == 'd':
            genotype.append(num_stones)

            for sls in stone_latent_states:
                genotype.append(sls)
        elif 's' in self.gt_scs[0]:
            n = int(self.gt_scs[0][1:])
            genotype.append(num_stones)

            stone_latent_states = np.reshape(stone_latent_states, (self.num_item_specs * MAX_STONES, 3))
            raw_stone_latent_states = np.tile(stone_latent_states, n)  # shape (self.num_item_specs * MAX_STONES, n * 3)
            raw_stone_latent_states = np.reshape(raw_stone_latent_states, (self.num_item_specs * MAX_STONES, n, 3))
            # Swap axes to get shape (self.num_item_specs * MAX_STONES, 3, n)
            raw_stone_latent_states = np.swapaxes(raw_stone_latent_states, 1, 2)
            raw_stone_latent_states = raw_stone_latent_states.flatten()
            # NOTE: As-is, this is a valid representation (since zeros will be
            # all zeros, and ones will be all ones)
            genotype.append(raw_stone_latent_states)

        # (b) Potion specifications
        if self.gt_scs[1] == 'd':
            genotype.append(num_potions)
            for pe in potion_effects:
                genotype.append(pe)
        
        # (c) Stone chemistry
        if self.gt_scs[2] == 'd':
            genotype.append((stone_rotation,))
            genotype.append(stone_reflection)
        
        # (d) Potion chemistry
        if self.gt_scs[3] == 'd':
            genotype.append(potion_reflection)
            genotype.append((potion_permutation,))

        # (e) Graph topology
        if self.gt_scs[4] == 'd':
            genotype.append((graph_topology,))
        elif self.gt_scs[4] == 'c6':
            # Convert graph constraints to genotype integer values
            # Flatten constraints 2d list
            constraints = np.array(graph_constraints).flatten()
            indices = [1, 2,   3, 5,   6, 7]
            # Use indices to get genotype from constraints
            gt = []
            for i in indices:
                gt.append(CONSTRAINT_STR_TO_INT[constraints[i]])
            genotype.append(gt)
        elif self.gt_scs[4] == 'c9':
            # Convert graph constraints to genotype integer values
            # Flatten constraints 2d list
            constraints = np.array(graph_constraints).flatten()
            # Get indices of constraints where the string contains a 1 substring
            constraint_indices = [i for i, c in enumerate(constraints) if '1' in c]
            assert 0 <= len(constraint_indices) <= 3
            indices = [1, 2,   3, 5,   6, 7]
            # Use indices to get genotype from constraints
            gt = []
            for i in range(3):
                # Get binary indicator for whether constraint is active
                active = len(constraint_indices) >= i + 1
                gt.append(active)
                if not active:
                    gt.append(0)  # We just pad with zeros if not active, 
                    gt.append(0)  # since these are made-up values that don't exist
                    continue
                # Get index of constraint
                constraint_index = constraint_indices[i]
                constraint_index = indices.index(constraint_index)  # Get corrected index
                gt.append(constraint_index)
                # Get constraint value
                constraint_value = CONSTRAINT_STR_TO_INT[constraint_index]
                gt.append(constraint_value)
            genotype.append(gt)
             
        # Create genotype from these components by concatenating them
        genotype = np.concatenate(genotype, axis=0).astype(int)

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