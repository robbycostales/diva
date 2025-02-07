import logging
from argparse import Namespace
from typing import Tuple

import gym
import gym_minigrid.minigrid as minigrid
import numpy as np

from diva.components.qd.measures.toygrid_measures import ToyGridMeasures
from diva.environments.toygrid.multigrid import Grid, MultiGridEnv

logger = logging.getLogger(__name__)


def sample_grid_location(
        size=None,
        region=None,
        sample_mode='uniform',
        border_size=0,
        seed=None,
        prng=None, 
        prevs=[],
):
    """
    Sample maze location given various specifications and constraints.

    NOTE: Either size or region must be provided.

    Args:
        size: Number of cells in (presumed square) maze.
        region: Region within maze to sample from (all inclusive). The region
            is defined by (i_1, j_1, i_2, j_2), where 1 is top left point and
            2 is bottom right.
        sample_mode: Method for sampling points within region.
        border_size: How many cells inwards to exclude from the boundary.
        seed: Seed used for RNG.
        prng: Random number generator.
        prevs: List of previously sampled points to avoid.
    Returns (tuple): (i, j) sampled location; (y, x).
    """
    if prng is None:
        prng = np.random.default_rng(seed)
    if size is None and region is None:
        raise ValueError('Need to provide either region or size.')

    # Define region if only size passed in
    if region is None:
        # If no region provided, allow sampling over full size of maze
        region = (0, 0, size-1, size-1)
    else:
        assert len(region) == 4
    
    # Potentially shrink region to account for maze border
    r = region
    bs = border_size
    r = (r[0]+bs, r[1]+bs, r[2]-bs, r[3]-bs)
    if sample_mode == 'uniform':
        while True:
            i = prng.integers(r[0], r[2]+1)
            j = prng.integers(r[1], r[3]+1)
            if (i, j) in prevs:
                continue
            else:
                return (i, j), prng
    else:
        raise NotImplementedError(f'Unknown sample mode: {sample_mode}')


class ToyGrid(MultiGridEnv):
    """Single-agent maze environment specified via a bit map."""

    def __init__(
        self,                       #
        size:                       int = 11,
        minigrid_mode:              bool = True,
        max_steps:                  int = None,
        reward_type:                str = 'sparse',
        initial_goal_visibility:    str = 'visible',
        goal_pos:                   Tuple[int] = None,
        goal_sampler:               str = 'uniform',
        goal_sampler_region:        Tuple[int] = None,
        variable_episode_lengths:     bool = False,
        seed:                       int = None,
        distribution_type:          str = 'SB',
        gt_type:                    str = 'a16s11',
        visualize:                  bool = False,
        dense_rewards:              bool = False,
    ):
        """ Maze environment initialization.
        
        Args:
        - size (int): Size of maze (odd number)--includes outer walls.
        - agent_view_size (int): Size of agent's view.
        - minigrid_mode (bool): Whether to use minigrid mode (not multigrid).
        - max_steps (int): Maximum number of steps in episode.
        - reward_type (str): Type of reward. Options are:
            - 'sparse': 1 if agent reaches goal, 0 otherwise.
            - 'dense': (proximity-based reward)
        - initial_goal_visibility (str): Whether goal is visible at start.
          Options are:
            - 'visible': Goal is visible at start.
            - 'invisible': Goal is not visible at start.
        - goal_pos (Tuple[int]): Goal position of agent.
        - goal_sampler (str): How to sample goal position. Options are:
            - 'uniform': Uniformly sample from all locations.
            - 'edges': Sample from edges of maze.
        - goal_sampler_region (Tuple[int]): Region to sample goal position
            from. If None, sample from entire maze.
        - variable_episode_lengths (bool): If we don't care about trajectories being the
          same length, we can set this to True. If False, when agent 
          reaches goal or dies, we keep episode running.
        - seed (int): Seed for RNG.
        - distribution_type: How to generate maze. Options are:
            - 'SB': Use seeded backtracking algorithm
                        and location specifications to generate maze.
            - 'QD': Use QD genotype to generate maze.
        - gt_type: How to interpret genotype. Options are:
            - 'aN': genotype is X, and N locations summed for Y
            - 'natural': genotype is X and Y
        - visualize (bool): Whether to visualize environment.
        - dense_rewards (bool): Whether to use dense rewards.
        """
        del visualize  # Unused for this environmnent
        self.size = size
        assert size == 11  # Hardcoded for now
        self.reward_type = reward_type
        self.rng_seed = seed
        self.distribution_type = distribution_type
        self.gt_type = gt_type
        self.dense_rewards = dense_rewards

        self.start_pos = (0, 0)
        self.goal_pos = goal_pos
        self._init_goal_pos = goal_pos
        self.goal_sampler = goal_sampler
        self.goal_sampler_region = goal_sampler_region

        if type(self.goal_sampler_region) is str:
            if self.goal_sampler_region == 'left':
                # i1 (y1), j1 (x1), i2 (y2), j2 (x2)
                # 'i' is top left point and 'j' is bottom right
                self.goal_sampler_region = (0, 0, size-1, size//2)
            elif self.goal_sampler_region == 'right':
                self.goal_sampler_region = (0, size//2, size-1, size-1)
            else:
                raise NotImplementedError
        
        self.goal_visibility = initial_goal_visibility

        self.bit_map = np.zeros((size - 2, size - 2))
        self.bit_map_padded = np.pad(self.bit_map, pad_width=1, mode='constant', constant_values=1)
        self.bit_map_shape = self.bit_map.shape
        self.bit_map_size = self.bit_map.size

        # Generate level from specifications
        if self.distribution_type == 'SB':
            self.generate_level_from_seed()
        elif self.distribution_type == 'QD':
            self.genotype_set = False
            self.generate_level_from_genotype(
                genotype=None, gt_type=self.gt_type)
        else:
            raise NotImplementedError(
                f'Unknown distribution type: {self.distribution_type}')

        # Set max_steps as function of size of maze, by default
        if max_steps is None:
            max_steps = 2 * size * size

        # For QD+MetaRL
        self._max_episode_steps = max_steps
        self.num_states = (size - 2)**2
        self.variable_episode_lengths = variable_episode_lengths

        super().__init__(
            n_agents=1,
            grid_size=size,
            agent_view_size=3,  # NOTE: doesn't matter
            max_steps=max_steps,
            see_through_walls=True,  # Set this to True for maximum speed
            minigrid_mode=minigrid_mode,
            seed=seed
        )

        self.grid_size = size
        
        # Observation space is just agent position
        self.coords_obs_space = gym.spaces.Box(
            low=0,
            high=self.grid_size,
            shape=(3,),  # also hardcoded
            dtype="uint8")

        self.observation_space = self.coords_obs_space

        # For QD+MetaRL
        self.genotype = None
        self.set_genotype_info()
        self.success = False

    def step(self, action):
        """ Step function. """

        success_prev = self.success

        obs, reward, done, info = super().step(action)

        if reward == 0:
            reward -= 0.05  # Whenever we have zero reward, we instead penalize
        
        if self.dense_rewards:
            # Compute dense reward
            agent_pos = np.array(self.agent_pos[0])
            goal_pos = np.array(self.goal_pos)
            dist = np.linalg.norm(agent_pos - goal_pos)
            
            # Normalize distance by the diagonal of the maze
            max_distance = np.sqrt(self.size**2 + self.size**2)
            normalized_dist = dist / max_distance
            
            # Scale reward with respect to max_steps
            scale_factor = 0.1 / self.max_steps
            dense_reward = -scale_factor * self.max_steps * (np.exp(normalized_dist) - 1)
            
            # Add dense reward
            reward += dense_reward

        if self.success:
            info['success'] = True
        else:
            info['success'] = False

        # If we've just succeeded, log as event
        if self.success and not success_prev:
            info['event'] = 'goal reached'

        return obs, reward, done, info
        
    def seed(self, seed=None):
        """ Set seed. """
        if seed is not None: 
            seed = int(seed)
        super().seed(seed=seed)
        self.rng_seed = seed

    def _gen_grid(self, width, height):
        """Generate grid from start/goal locations."""
        if self.distribution_type == 'QD' and not self.genotype_set:
            return
        # Create an empty grid
        self.grid = Grid(width, height)
        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)
        # Goal
        self.put_obj(minigrid.Goal(), self.goal_pos[0], self.goal_pos[1])
        # Agent
        self.place_agent_at_pos(0, self.start_pos)
    
    def set_genotype_info(self):
        """ Extract information from genotype and genotype type. """

        self.measures = ToyGridMeasures.get_all_measures()

        if self.gt_type[0] == 'a':
            # Parse format of gt_type: 'a<num_terms>s<size>'
            _parts = self.gt_type[1:].split('s')
            num_y_pos_terms = int(_parts[0])
            assert len(_parts) == 2 and len(_parts[1]) > 0, f'Invalid gt_type: {self.gt_type}'
            self.size = int(_parts[1])

            # Genotype is goal location, with unnecessary y position terms
            # Length is [ 1    +    num_y_terms ] 
            #           [ x pos  +  y pos terms ]
            self.genotype_size = 1 + num_y_pos_terms
            # Lower bound for goal in x position is 1, since 0 is wall; 
            # For y positions, each term is either -1, 0, or 1, and we just
            # clip the sum to be within the appropriate range
            self.genotype_lower_bounds = np.array(
                [1]             + [-1] * num_y_pos_terms)
            self.genotype_upper_bounds = np.array(
                [self.size - 2] + [+1] * num_y_pos_terms)                                    

        elif self.gt_type.startswith('natural'):
            # Parse format of gt_type: 'natural<size>'
            _parts = self.gt_type.split('s')
            assert len(_parts) == 2 and len(_parts[1]) > 0, f'Invalid gt_type: {self.gt_type}'
            self.size = int(_parts[1])

            # Genotype is just direct goal location
            # Length is [ 1      +      1 ] 
            #           [  x pos + y pos  ]
            self.genotype_size = 2
            self.genotype_lower_bounds = np.array([1, 1])
            self.genotype_upper_bounds = np.array([self.size - 2, self.size - 2])
        else:
            raise ValueError('Unknown genotype type: {}'.format(self.gt_type))
        
        self.genotype_bounds = [(l, u) for l, u in  # noqa: E741
                                zip(list(self.genotype_lower_bounds), 
                                    list(self.genotype_upper_bounds))]

    def pos_to_flat_pos(self, pos):
        """ Convert position tuple to flattened index. """
        return (pos[0]-1) * (self.size - 2) + (pos[1]-1)
    
    def get_measures_info(self, env_name):
        """ Get info about measures. """
        return ToyGridMeasures.get_measures_info(env_name)

    def compute_measures(self, genotype=None, measures=None, return_pg=False):
        """" Compute the measures for the given maze. """
        # Use current maze measures and genotype if none provided
        if measures is None:
            measures = self.measures
        if genotype is None:
            genotype = self.genotype

        # Extract useful properties of genotype
        pg = self._process_genotype(genotype)

        # Compute measures
        meas = ToyGridMeasures.compute_measures(
            genotype=genotype,
            goal_pos = pg.goal_pos,
            measures=measures
        )

        if return_pg:
            return meas, pg
        else:
            return meas
    
    @staticmethod
    def compute_measures_static(genotype=None, size=None, measures=None, gt_type=None, return_pg=False):
        """" Compute the measures for the given maze. """

        # Extract useful properties of genotype
        pg = ToyGrid.process_genotype(genotype, gt_type=gt_type)

        # Compute measures
        meas = ToyGridMeasures.compute_measures(
            genotype=genotype,
            goal_pos = pg.goal_pos,
            measures=measures
        )

        if return_pg:
            return meas, pg
        else:
            return meas

    def genotype_from_seed(self, seed, level_store=None):
        """ Generate or retrieve (if already generated) level from seed. """
        # First, check if seed in level store
        if level_store is not None and seed in level_store.seed2level:
            # NOTE: For now, we are ignoring the whole encoding thing, as we are
            # not using it (see level_store.get_level for detailes)
            sol = level_store.seed2level[seed]
        # Otherwise, generate level    
        else:
            num_attempts = 0
            rng = np.random.default_rng(seed)
            while True:
                # Keep using this seed to generate environments until one is valid
                num_attempts += 1
                sol = rng.integers(
                    low=self.genotype_lower_bounds,
                    high=self.genotype_upper_bounds + 1,
                    size=(self.genotype_size))

                # Check if solution is valid
                pg = self._process_genotype(sol)
                valid, reason = self.is_valid_genotype(
                    pg, gt_type=self.gt_type)
                del reason
                if valid:
                    break
                if num_attempts == 100:
                    print('WARNING: Could not sample a valid solution after 100 attempts')
                if num_attempts > 100_000:
                    raise RuntimeError("Could not sample a valid solution")
        return sol
    
    @staticmethod
    def genotype_from_seed_static(
            seed, gt_type='a16', genotype_lower_bounds=None, 
            genotype_upper_bounds=None, genotype_size=None):
        num_attempts = 0
        rng = np.random.default_rng(seed)
        while True:
            # Keep using this seed to generate environments until one is valid
            num_attempts += 1
            sol = rng.integers(
                low=genotype_lower_bounds,
                high=genotype_upper_bounds + 1,
                size=(genotype_size))

            # Check if solution is valid
            pg = ToyGrid.process_genotype(sol, gt_type=gt_type)  # Hardcoded for now!
            valid, reason = ToyGrid.is_valid_genotype(
                pg, gt_type=gt_type)
            del reason
            if valid:
                break
            if num_attempts == 100:
                print('WARNING: Could not sample a valid solution after 100 attempts')
            if num_attempts > 100_000:
                raise RuntimeError("Could not sample a valid solution")
        return sol


    def _process_genotype(self, genotype):
        """Extract information from genotype and genotype type."""
        gisn = genotype is None

        assert gisn or len(genotype) == self.genotype_size, \
            f'Genotype length {len(genotype)} != {self.genotype_size}'

        if self.gt_type[0] == 'a':
            # Parse format of gt_type: 'a<num_terms>s<size>'
            _parts = self.gt_type[1:].split('s')
            num_y_pos_terms = int(_parts[0])
            assert len(_parts) == 2 and len(_parts[1]) > 0, f'Invalid gt_type: {self.gt_type}'
            self.size = int(_parts[1])

            # Genotype is goal location, with unnecessary y position terms
            # Length is [ 1    +    num_y_terms ]
            #           [ x pos  +  y pos terms ]
            # E.g. if size = 11, then we get 4; this is how many blank cells
            # above and below goal position
            y_num = (self.size - 2 - 1) // 2 
            # Ensure that the number of terms we're using for y position is a
            # multiple of y_num, so division is clean
            assert num_y_pos_terms % y_num == 0
            # y_div is what we will divide our sum by to get final y position
            y_div = num_y_pos_terms // y_num
            # Get raw x position and y position terms
            x_pos = None if gisn else genotype[0]
            y_pos_terms = None if gisn else genotype[1:]
            # Get scaled y sum
            y_unshifted = None if gisn else np.sum(y_pos_terms) // y_div
            # For e.g. size = 11 for, we will get values between -4 and 4, so 
            # we shift by 5 to get values between 1 and 9 (but do for 
            # general case)
            y_shift = (self.size - 2 - 1) // 2 + 1
            # Get final y position
            y_pos = None if gisn else y_unshifted + y_shift

        elif self.gt_type.startswith('natural'):
            # Parse format of gt_type: 'natural<size>'
            _parts = self.gt_type.split('s')
            assert len(_parts) == 2 and len(_parts[1]) > 0, f'Invalid gt_type: {self.gt_type}'
            self.size = int(_parts[1])

            # Genotype is just direct goal location
            # Length is [ 1      +      1 ] 
            #           [  x pos + y pos  ]
            x_pos = None if gisn else genotype[0]
            y_pos = None if gisn else genotype[1]

        goal_pos = None if gisn else (y_pos, x_pos)

        # Return processed genotype
        processed_genotype = {
            'grid_size': self.size,
            'start_pos': (self.size // 2, self.size // 2),
            'goal_pos': goal_pos,
            'genotype': genotype,
            'genotype_bounds': self.genotype_bounds,
            'genotype_size': self.genotype_size,
            'genotype_lower_bounds': self.genotype_lower_bounds,
            'genotype_upper_bounds': self.genotype_upper_bounds,
        }
        
        return Namespace(**processed_genotype)

    @staticmethod
    def process_genotype(genotype, gt_type='a64s11'):
        """Extract information from genotype and genotype type.
        
        Args:
            genotype: The genotype to process
            gt_type: String specifying genotype type and parameters.
                    Format for 'a' type: 'a<num_terms>s<size>' (e.g. 'a64s11')
                    Format for 'natural' type: 'naturals<size>' (e.g. 'naturals11')
        """
        gisn = genotype is None

        if gt_type[0] == 'a':
            # Parse format of gt_type: 'a<num_terms>s<size>'
            _parts = gt_type[1:].split('s')
            num_y_pos_terms = int(_parts[0])
            assert len(_parts) == 2 and len(_parts[1]) > 0, f'Invalid gt_type: {gt_type}'
            size = int(_parts[1])

            # Genotype is goal location, with unnecessary y position terms
            # Length is [ 1    +    num_y_terms ]
            #           [ x pos  +  y pos terms ]
            # E.g. if size = 11, then we get 4; this is how many blank cells
            # above and below goal position
            y_num = (size - 2 - 1) // 2  # For 11, this is 8//2 = 4
            genotype_size = 1 + num_y_pos_terms    # a128: = 1 + 128
            # Ensure that the number of terms we're using for y position is a
            # multiple of y_num, so division is clean
            assert num_y_pos_terms % y_num == 0    # 128 % 4 = 0
            # y_div is what we will divide our sum by to get final y position
            y_div = num_y_pos_terms // y_num       # 128 // 4 = 32
            # Get raw x position and y position terms
            x_pos = None if gisn else genotype[0]
            y_pos_terms = None if gisn else genotype[1:]
            # Get scaled y sum
            y_unshifted = None if gisn else np.sum(y_pos_terms) // y_div    
            # E.g. min = (-1 * 128)//32 = 4; max = (1 * 128)//32 = 34

            # For e.g. size = 11 for, we will get values between -4 and 4, so
            # we shift by 5 to get values between 1 and 9 (but do for
            # general case)
            y_shift = (size - 2 - 1) // 2 + 1
            # Get final y position
            y_pos = None if gisn else y_unshifted + y_shift

            # Bounds
            genotype_lower_bounds = np.array(
                [1]        + [-1] * num_y_pos_terms)
            genotype_upper_bounds = np.array(
                [size - 2] + [+1] * num_y_pos_terms)
            
        elif gt_type.startswith('natural'):
            _parts = gt_type.split('s')
            assert len(_parts) == 2 and len(_parts[1]) > 0, f'Invalid gt_type: {gt_type}'
            size = int(_parts[1])

            # Genotype is just direct goal location
            # Length is [ 1      +      1 ] 
            #           [  x pos + y pos  ]
            genotype_size = 2
            x_pos = None if gisn else genotype[0]
            y_pos = None if gisn else genotype[1]

            # Bounds
            genotype_lower_bounds = np.array([1, 1])
            genotype_upper_bounds = np.array([size - 2, size - 2])
        else:
            raise NotImplementedError(
                'Unknown genotype type: {}'.format(gt_type))

        goal_pos = None if gisn else (y_pos, x_pos)

        genotype_bounds = [(l, u) for l, u in  # noqa: E741
                            zip(list(genotype_lower_bounds),
                                list(genotype_upper_bounds))]

        # Return processed genotype
        processed_genotype = {
            'grid_size': size,
            'start_pos': (size // 2, size // 2),
            'goal_pos': goal_pos,
            'genotype': genotype,
            'genotype_bounds': genotype_bounds,
            'genotype_size': genotype_size,
            'genotype_lower_bounds': genotype_lower_bounds,
            'genotype_upper_bounds': genotype_upper_bounds,
        }
        
        return Namespace(**processed_genotype)
    
    @staticmethod
    def is_valid_genotype(processed_genotype, gt_type=None):
        """Check if genotype is valid"""
        del gt_type
        pg = processed_genotype
        # Check if genotype within bounds
        if pg.genotype is None:
            return False, 'none_genotype'
        if np.any(pg.genotype < pg.genotype_lower_bounds):
            return False, 'lower_bound_violation'
        if np.any(pg.genotype > pg.genotype_upper_bounds):
            return False, 'upper_bound_violation'
        # Check goal does not overlap with agent start position
        # Start position is at center of maze (size // 2, size // 2)
        start_pos = (pg.grid_size // 2, pg.grid_size // 2)
        if pg.goal_pos == start_pos:
            return False, 'goal_start_overlap'
        
        return True, None
    
    def set_genotype_from_current_grid(self):
        """Set genotype from current grid.
        
        This method is called after a level is generated by a generator
        (i.e. not a genotype), and we want to set the genotype from the current
        grid information.

        Returns:
            Namespace: Processed genotype.
        """

        if self.gt_type[0] == 'a':
            # Parse format of gt_type: 'a<num_terms>s<size>'
            _parts = self.gt_type[1:].split('s')
            num_y_pos_terms = int(_parts[0])
            assert len(_parts) == 2 and len(_parts[1]) > 0, f'Invalid gt_type: {self.gt_type}'
            self.size = int(_parts[1])

            # There is not a deterministic mapping from grid to genotype, so
            # we just set values to one possible genotype
            # Goal pos of form (y, x); that is (i, j)
            x_pos = self.goal_pos[1]
            y_pos = self.goal_pos[0]
            
            # From y_pos, we first want to compute shifted y_pos, which is
            # y_pos - y_shift
            y_shift = (self.size - 2 - 1) // 2 + 1
            y_unshifted = y_pos - y_shift
            # Now, we we set the magnitude of y_unshifted number of y_pos_terms
            # to the sign of y_unshifted
            y_pos_terms = np.zeros(num_y_pos_terms)
            num_terms = np.abs(y_unshifted) * 4
            y_pos_terms[:num_terms] = np.sign(y_unshifted)
            genotype = np.concatenate(([x_pos], y_pos_terms))

        elif self.gt_type.startswith('natural'):
            # Parse format of gt_type: 'natural<size>'
            _parts = self.gt_type.split('s')
            assert len(_parts) == 2 and len(_parts[1]) > 0, f'Invalid gt_type: {self.gt_type}'
            self.size = int(_parts[1])

            # goal_pos of form (y, x); that is (i, j)
            genotype = np.array([self.goal_pos[1], self.goal_pos[0]])
        else:
            raise NotImplementedError(
                'Unknown genotype type: {}'.format(self.gt_type))
        
        self.genotype = genotype.astype(int)

        return self._process_genotype(genotype)

    def generate_level_from_genotype(self, genotype, gt_type='a16s11'):
        """Generate level from genotype, which is a sequence of ints"""
        if genotype is not None:
            genotype = np.array(genotype).astype(int)
            self.genotype = genotype
        
        if genotype is None and self.genotype_set:
            self._gen_grid(self.size, self.size)
            return

        # Process genotype
        pg = self.process_genotype(
            genotype, gt_type=gt_type)
        
        # Set common variables
        self.genotype_size = pg.genotype_size
        self.genotype_lower_bounds = pg.genotype_lower_bounds
        self.genotype_upper_bounds = pg.genotype_upper_bounds
        self.genotype_bounds = pg.genotype_bounds
        self.genotype = pg.genotype
        self.start_pos = pg.start_pos
        self.goal_pos = pg.goal_pos

        # Indicate that genotype is set
        if genotype is None:
            # NOTE: This is important because we might pass in a None genotype
            # here after we've already set one
            self.genotype_set = False
        else:
            self.genotype_set = True
            self._gen_grid(self.size, self.size)

    def generate_level_from_seed(self, seed=None):
        """
        We assume self.rng_seed is already set from __init__, so no need to
        pass in seed here, unless we're changing the task. We assume self.seed
        has already been called.
        """
        if seed is not None:
            self.seed(seed=seed)

        prevs = []
        self.start_pos = np.array((self.size // 2, self.size // 2))
        prevs.append(tuple(self.start_pos))

        # Set goal location
        if self._init_goal_pos is None:
            # Sample goal location
            # goal_pos is of form (x, y); that is (j, i)
            self.goal_pos, _ = sample_grid_location(
                size=self.size,
                region=self.goal_sampler_region,
                sample_mode=self.goal_sampler,
                border_size=1,
                seed=self.rng_seed,
                prevs=prevs
            )
            self.goal_pos = np.array(self.goal_pos)
        else:
            self.goal_pos = np.array(self.goal_pos)


    def gen_obs(self):
        """Add goal loc to observations"""
        # Observation is just agent current position plus its direction as an int
        pos = np.array(self.agent_pos[0])
        dr =  self.agent_dir[0]
        obs = np.concatenate((pos, [dr]))
        return obs
    
    def _reward(self):
        """
        Fixing the reward to make it markovian... commented-out version is not
        """
        # return 1 - 0.9 * (self.step_count / self.max_steps)
        return 1

    def agent_is_done(self, agent_id):
        """
        Overwriting MultiGridEnv functionality so that we can choose if
        episode ends when agent reaches a goal state.
        """
        # If we want all trajectories to be the same length, we do not allow
        # ending the episode early
        if not self.variable_episode_lengths:
            self.success = True
            return
        else:
            # Otherwise, we use the parent method
            self.success = True
            super().agent_is_done(agent_id)

    def reset(self):
        """ Copied from MultiGridEnv.reset() and modified for QD """
        if self.distribution_type == 'QD' and not self.genotype_set:
            # NOTE: If we're using QD, genotype needs to be set before we
            # generate the grid, etc. For now, return placeholder obs by
            # generating maze from seed
            return None
        
        if self.fixed_environment:
            self.seed(self.seed_value)

        # Current position and direction of the agent
        self.agent_pos = [None] * self.n_agents
        self.agent_dir = [None] * self.n_agents
        self.done = [False] * self.n_agents

        # Generate the grid. Will be random by default, or same environment if
        # 'fixed_environment' is True.
        self._gen_grid(self.width, self.height)

        # These fields should be defined by _gen_grid
        for a in range(self.n_agents):
            assert self.agent_pos[a] is not None
            assert self.agent_dir[a] is not None

            # Check that the agent doesn't overlap with an object
            start_cell = self.grid.get(*self.agent_pos[a])
            assert (start_cell.type == "agent" or start_cell is None or
                    start_cell.can_overlap()), \
                   "Invalid starting position for agent: {}\n".format(start_cell) + \
                   "Agent pos: {}\n".format(self.agent_pos[a]) + \
                   "Agent dir: {}\n".format(self.agent_dir[a]) + \
                   "Genotype: {}\n".format(None if not self.distribution_type 
                                         == 'QD' else self.genotype) + \
                   "Is valid genotype: {}, {}\n".format(*self.is_valid_genotype(
                        self.process_genotype(self.genotype, self.gt_type),
                        self.gt_type))

        # Item picked up, being carried, initially nothing
        self.carrying = [None] * self.n_agents

        # Step count since episode start
        self.step_count = 0

        # Return first observation
        obs = self.gen_obs()

        self.success = False

        return obs

    def reset_task(self, task=None) -> None:
        """
        Reset current task (i.e. seed, genotype, etc.).

        Reset the task, either at random (if task=None) or the given task.
        Should *not* reset the environment. Returns the coordinates of a new
        goal state. 
        """
        # Generate level from specifications
        if self.distribution_type == 'SB':
            # If we're using a seed-based distribution, we need to generate
            # a new seed and then generate the level from that seed
            self.seed(task)
            self.generate_level_from_seed(seed=task)
        elif self.distribution_type == 'QD':
            # Convert genotype to all int array
            self.generate_level_from_genotype(genotype=task, gt_type=self.gt_type)
        else:
            raise ValueError(
                f'Unknown distribution type: {self.distribution_type}')
        self._gen_grid(self.width, self.height)

        if self.distribution_type == 'SB':
            # Set genotype from current task (generated from seed)
            _ = self.set_genotype_from_current_grid()

        if self.distribution_type == 'QD':
            return self.genotype
        else:
            return self.goal_pos
    
    def get_task(self):
        """ Return the ground truth task. """
        # TODO: more thoughtful implementation
        if hasattr(self, 'genotype') and self.genotype is not None:
            return np.asarray(self.genotype).copy()
        else:
            return np.array((0.0,))

    def task_to_id(self, goals):
        """
        MazeEnv can be enumerated as easily as VariBAD's gridworld environment, 
        so instead of using a separate head for each state in reward prediction,
        we pass it in as input. Thus, we do not need this function (I think).
        """
        raise NotImplementedError
    
    def id_to_task(self, classes):
        """ Undefined for same reason as `task_to_id` (see docstring). """
        raise NotImplementedError

    def goal_to_onehot_id(self, pos):
        """ Undefined for same reason as `task_to_id` (see docstring). """
        raise NotImplementedError

    def onehot_id_to_goal(self, pos):
        """ Undefined for same reason as `task_to_id` (see docstring). """
        raise NotImplementedError
    
    def _reset_belief(self) -> np.ndarray:
        raise NotImplementedError('Oracle not implemented for MazeEnv.')

    def update_belief(self, state, action) -> np.ndarray:
        raise NotImplementedError('Oracle not implemented for MazeEnv.')

    def get_belief(self):
        raise NotImplementedError('Oracle not implemented for MazeEnv.')

    @property
    def level_rendering(self):
        """Render high-level view of level"""
        return self.render(mode='human')