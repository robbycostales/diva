import itertools

import numpy as np

from dm_alchemy.types.graphs import constraint_from_graph
from diva.components.qd.measures.measure_selection import MeasureInfo


class AlchemyMeasures:
    
    @staticmethod
    def _manhattan_distance(a, b):
        """ Helper method to compute the manhattan distance between two points. """
        return sum(abs(a_i - b_i) for a_i, b_i in zip(a, b))

    @staticmethod
    def _euclidean_distance(a, b):
        """ Helper method to compute the euclidean distance between two points. """
        return np.linalg.norm(np.array(a) - np.array(b))
    
    @staticmethod
    def _reflection_to_int(reflection):
        """ Compute the reflection of the potion. """
        # We have shape {0, 1}^3; we can convert to an integer
        return reflection[0] * 4 + reflection[1] * 2 + reflection[2]
    
    @staticmethod
    def _stone_latent_to_int(latent):
        """ Convert latent state to integer. """
        # We have shape {0, 1}^3; we can convert to an integer
        return latent[0] * 4 + latent[1] * 2 + latent[2]
    
    ###########################################################################
    #                                MEASURES                                 #  
    ###########################################################################

    """
    stone_rotation
    
    Stone rotation.
    """

    @staticmethod
    def i_stone_rotation():
        return MeasureInfo(
            full_range      = (-0.5, 3.5),
            sample_range    = (0, 3),
            normal_params   = None,
            max_cells       = 4,
            sample_dist     = 'uniform',
            fn              = AlchemyMeasures.m_stone_rotation,
            args            = ('stone_rotation',)
        )
    
    @staticmethod
    def m_stone_rotation(stone_rotation):
        return stone_rotation[0]

    ###########################################################################

    """
    stone_reflection

    Stone reflection.
    """

    @staticmethod
    def i_stone_reflection():
        return MeasureInfo(
            full_range      = (-0.5, 7.5),
            sample_range    = (0, 7),
            normal_params   = None,
            max_cells       = 8,
            sample_dist     = 'uniform',
            fn              = AlchemyMeasures.m_stone_reflection,
            args            = ('stone_reflection',)
        )
    
    @staticmethod
    def m_stone_reflection(stone_reflection):
        return AlchemyMeasures._reflection_to_int(stone_reflection)
    
    ###########################################################################

    """
    potion_reflection

    Potion reflection.
    """

    @staticmethod
    def i_potion_reflection():
        return MeasureInfo(
            full_range      = (-0.5, 7.5),
            sample_range    = (0, 7),
            normal_params   = None,
            max_cells       = 8,
            sample_dist     = 'uniform',
            fn              = AlchemyMeasures.m_potion_reflection,
            args            = ('potion_reflection',)
        )
    
    @staticmethod
    def m_potion_reflection(potion_reflection):
        return AlchemyMeasures._reflection_to_int(potion_reflection)
    
    ###########################################################################

    """
    potion_permutation

    Potion permutation.
    """

    @staticmethod
    def i_potion_permutation():
        return MeasureInfo(
            full_range      = (-0.5, 5.5),
            sample_range    = (0, 5),
            normal_params   = None,
            max_cells       = 6,
            sample_dist     = 'uniform',
            fn              = AlchemyMeasures.m_potion_permutation,
            args            = ('potion_permutation',)
        )
    
    @staticmethod
    def m_potion_permutation(potion_permutation):
        return potion_permutation

    ###########################################################################

    """
    average_manhattan_to_optimal 
    
    Average manhattan distance between all stones (across all trials) to
    the optimal state.
    """

    @staticmethod
    def i_average_manhattan_to_optimal():
        return MeasureInfo(
            full_range      = (-0.01, 2.05),
            sample_range    = (0.99, 2.00), 
            normal_params   = None,
            max_cells       = 10,
            sample_dist     = 'normal', 
            fn              = AlchemyMeasures.m_average_manhattan_to_optimal,
            args            = ('stone_latent_states',)
        )

    @staticmethod
    def m_average_manhattan_to_optimal(stone_latent_states):
        optimal = (1, 1, 1)
        distances = []
        for sls in stone_latent_states:
            distances += [AlchemyMeasures._manhattan_distance(stone, optimal) for stone in sls]
        return sum(distances) / len(distances)

    ###########################################################################

    """
    average_stone_to_stone_distance 
    
    Average euclidean distance between all pairs of stones (across all trials).
    """

    @staticmethod
    def i_average_stone_to_stone_distance():
        return MeasureInfo(
            full_range      = (-0.01, 1.5),
            sample_range    = (0.75, 1.5), 
            normal_params   = None,
            max_cells       = 10,
            sample_dist     = 'normal', 
            fn              = AlchemyMeasures.m_average_stone_to_stone_distance,
            args            = ('stone_latent_states',)
        )

    @staticmethod
    def m_average_stone_to_stone_distance(stone_latent_states):
        distances = []
        for sls in stone_latent_states:
            pairs = itertools.combinations(sls, 2)
            distances += [AlchemyMeasures._euclidean_distance(a, b) for a, b in pairs]
        # If there is only one stone, then we will not get any pairs, hence a divide by zero error
        return 0 if len(distances) == 0 else sum(distances) / len(distances)

    ###########################################################################
    
    """
    stone_to_stone_distance_variance

    Variance of the distances between stones (across all trials).
    """

    @staticmethod
    def i_stone_to_stone_distance_variance():
        return MeasureInfo(
            full_range      = (-0.01, 0.6),
            sample_range    = (0.1, 0.6), # NOTE: We want to exclude points 
                                     # exactly at 0; need high enough resolution 
                                     # for this to be possible (since we err
                                     # on the inclusive side for sample_range)
            normal_params   = None,
            max_cells       = 8,
            sample_dist     = 'uniform',  # It's actually a skewed normal distribution
            fn              = AlchemyMeasures.m_stone_to_stone_distance_variance,
            args            = ('stone_latent_states',)
        )

    @staticmethod
    def m_stone_to_stone_distance_variance(stone_latent_states):
        distances = []
        for sls in stone_latent_states:
            pairs = itertools.combinations(sls, 2)
            distances += [AlchemyMeasures._euclidean_distance(a, b) for a, b in pairs]
        # If there is only one stone, then we will not get any pairs, hence a divide by zero error
        return 0 if len(distances) == 0 else np.var(distances)

    ###########################################################################

    """
    latent_state_diversity

    The 'diversity' of the latent stone states (across all trials). 
    Diversity is calculated as the standard deviation of each latent
    state coordinate across all stones.
    """

    @staticmethod
    def i_latent_state_diversity():
        return MeasureInfo(
            full_range      = (-0.01, 0.33),
            sample_range    = (0.05, 0.32),
            normal_params   = None,
            max_cells       = 9,
            sample_dist     = 'normal',
            fn              = AlchemyMeasures.m_latent_state_diversity,
            args            = ('stone_latent_states',)
        )

    @staticmethod
    def m_latent_state_diversity(stone_latent_states):
        std_devs = []
        for sls in stone_latent_states:
            std_devs.append(np.std(np.array(sls), axis=0))
        # Use mean of standard deviations as a measure of overall diversity
        return np.mean(std_devs)

    ###########################################################################

    """
    potion_effect_diversity

    The 'diversity' of the potion effects (across all trials).
    Diversity is calculated as the standard deviation of each potion
    effect coordinate across all potions.
    """

    @staticmethod
    def i_potion_effect_diversity():
        return MeasureInfo(
            full_range      = (0.75, 1.8),
            sample_range    = (0.75, 1.8),
            normal_params   = None,
            max_cells       = 10,
            sample_dist     = 'normal',  # Slightly skewed
            fn              = AlchemyMeasures.m_potion_effect_diversity,
            args            = ('potion_effects',)
        )

    @staticmethod
    def m_potion_effect_diversity(potion_effects):
        # TODO: Below is incorrect because ints are passed in, not raw potion effects
        std_devs = []
        for pe in potion_effects:
            std_devs.append(np.std(np.array(pe), axis=0))
        return np.mean(std_devs)

    ###########################################################################

    """
    graph_num_bottlenecks

    The number of bottlenecks in the graph topology.
    """

    @staticmethod
    def i_graph_num_bottlenecks():
        return MeasureInfo(
            full_range      = (-0.5, 3.5),
            sample_range    = (0, 3),
            normal_params   = None,
            max_cells       = 4,
            sample_dist     = 'uniform',
            fn              = AlchemyMeasures.m_graph_num_bottlenecks,
            args            = ('graph_topology',)
        )

    @staticmethod
    def m_graph_num_bottlenecks(graph_topology):
        constraint = constraint_from_graph(graph_topology)
        # flatten constraint 2D list
        constraint = [item for sublist in constraint for item in sublist]
        # count number of bottlenecks (strings with 1 in a substring)
        return sum([1 for c in constraint if '1' in c])

    ###########################################################################
    
    """
    parity_first_stone

    First stone location (first trial, first stone), as parity measure.
    """

    @staticmethod
    def i_parity_first_stone():
        return MeasureInfo(
            full_range      = (-0.5, 7.5),
            sample_range    = (-0.5, 7.5),
            normal_params   = None,
            max_cells       = 8,
            sample_dist     = 'uniform',
            fn              = AlchemyMeasures.m_parity_first_stone,
            args            = ('stone_latent_states',)
        )
    
    @staticmethod
    def m_parity_first_stone(stone_latent_states):
        # convert from {0, 1}^3 to int
        # first_latent = (stone_latent_states[0][0] + 1.0)/2  # Convert from {-1, 1} to {0, 1}
        first_latent = stone_latent_states[0][0]  # Convert from {0, 1}
        int_value = AlchemyMeasures._stone_latent_to_int(first_latent)
        return int_value

    ###########################################################################
    
    """
    parity_first_potion

    First potion location (first trial, first potion), as parity measure.
    """

    @staticmethod
    def i_parity_first_potion():
        return MeasureInfo(
            full_range      = (-0.5, 5.5),
            sample_range    = (0, 5),  
            normal_params   = None,
            max_cells       = 6,
            sample_dist     = 'uniform',
            fn              = AlchemyMeasures.m_parity_first_potion,
            args            = ('potion_effects',)
        )
    
    @staticmethod
    def m_parity_first_potion(potion_effects):
        # Already an int
        return potion_effects[0][0]
        
    
    ###########################################################################

    """
    <measure_name>

    Copy and define a new measure here.
    """
    
    ###########################################################################

    @staticmethod
    def get_all_measures():
        """ Return a list of all available measures. """
        return MEASURES
    
    @staticmethod
    def get_measures_info(env_name): 
        del env_name  # Right now, all values are the same across all Alchemy envs
        return MEASURE_INFO

    @staticmethod
    def compute_measures(stone_latent_states=None, 
                         stone_reflection=None, 
                         stone_rotation=None,
                         potion_effects=None,
                         potion_reflection=None,
                         potion_permutation=None, 
                         graph_topology=None, 
                         measures=None):
        """ Compute the specified measures for the alchemy environment.
        
        Args:
        - stone_latent_states (array-like): An array of latent states for stones.
        - graph_topology (Graph object): Graph object containing graph topology.
        - potion_effects (array-like): An array of the effects of each potion.
        - measures (list): List of measures to compute. If None, all measures are computed.

        Returns:
        A dictionary with the computed measures.
        """
        # If measures not provided, compute all measures
        if measures is None:
            measures = AlchemyMeasures.get_all_measures()

        # All args
        all_args = {
            'stone_latent_states': stone_latent_states,
            'stone_reflection': stone_reflection,
            'stone_rotation': stone_rotation,
            'potion_effects': potion_effects,
            'potion_reflection': potion_reflection,
            'potion_permutation': potion_permutation,
            'graph_topology': graph_topology
        }

        # Collect all measures
        measures_dict = dict()
        for measure in measures:
            # Find measure function
            mi = MEASURE_INFO[measure]
            # Assert that all arguments are not none
            for k in mi.args:
                assert all_args[k] is not None, f"Missing argument {k} for measure {measure}"
            # Call function to get measure value
            val = mi.fn(**{k: all_args[k] for k in mi.args})
            measures_dict[measure] = val

        # Ensure no NaNs or Infs
        for k, v in measures_dict.items():
            if np.isnan(v) or np.isinf(v):
                measures_dict[k] = -1

        return measures_dict
    

METHOD_NAMES = [name for name in dir(AlchemyMeasures) if name.startswith("m_") and callable(getattr(AlchemyMeasures, name))]
MEASURES = [name[2:] for name in METHOD_NAMES]

MEASURE_INFO = dict()
for measure in MEASURES:
    info_fn = getattr(AlchemyMeasures, 'i_' + measure, None)
    assert info_fn is not None, f"Missing info function for measure {measure}"
    MEASURE_INFO[measure] = info_fn()


if __name__ == '__main__':
    # Print all measures
    for i, measure in enumerate(MEASURES):
        print(i, measure)