# Copyright (c) OpenAI
# 
# Licensed under the MIT License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/licenses/MIT
#
# This file is an extended version of
# https://github.com/openai/gym/blob/master/gym/envs/box2d/car_racing.py
import argparse
import math
import os
import time
from argparse import Namespace

import Box2D
import gym
import numpy as np
import pyglet
from Box2D.b2 import contactListener, fixtureDef, polygonShape
from gym import spaces

# from gym.envs.box2d.car_dynamics import Car
from gym.utils import EzPickle, seeding
from PIL import Image

from diva.components.qd.measures.car_racing_measures import CarRacingMeasures
from diva.environments.box2d import geo_complexity
from diva.environments.box2d.car_dynamics import Car

os.environ['PYGLET_SHADOW_WINDOW'] = '0'

pyglet.options["debug_gl"] = False
from pyglet import gl  # noqa: E402

from . import bezier  # noqa: E402

# STATE_W = 96
# STATE_H = 96
VIDEO_W = 600
VIDEO_H = 400
WINDOW_W = 1000
WINDOW_H = 800

SCALE = 6.0  # Track scale
TRACK_RAD = 900 / SCALE  # Track is heavily morphed circle with this radius
PLAYFIELD = 2000 / SCALE  # Game over boundary
# FPS = 50  # Frames per second
ZOOM = 2.7  # Camera zoom
ZOOM_FOLLOW = True  # Set to False for fixed view (don't use zoom)


TRACK_DETAIL_STEP = 21 / SCALE
TRACK_TURN_RATE = 0.31
TRACK_WIDTH = 40 / SCALE
BORDER = 8 / SCALE
BORDER_MIN_COUNT = 4

ROAD_COLOR = [0.4, 0.4, 0.4]

N_SEGS_FOR_1D_OBS = 20




class FrictionDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        self._contact(contact, True)

    def EndContact(self, contact):
        self._contact(contact, False)

    def _contact(self, contact, begin):
        tile = None
        obj = None
        u1 = contact.fixtureA.body.userData
        u2 = contact.fixtureB.body.userData
        index = -1
        if u1 and "tile" in u1:
            if "road_friction" in u1['tile'].__dict__:
                tile = u1['tile']
                index = u1['index']
                obj = u2
        if u2 and "tile" in u2:
            if "road_friction" in u2['tile'].__dict__:
                tile = u2['tile']
                index = u2['index']
                obj = u1
        if not tile:
            return

        tile.color[0] = ROAD_COLOR[0]
        tile.color[1] = ROAD_COLOR[1]
        tile.color[2] = ROAD_COLOR[2]

        if not obj or "tiles" not in obj.__dict__:
            return
        if begin:
            obj.tiles.add(tile)
            if not tile.road_visited:
                tile.road_visited = True
                self.env.reward += 1000.0 / len(self.env.track)
                self.env.tile_visited_count += 1

            if self.env.sparse_rewards and index >= 0:
                self._eval_tile_index(index)
        else:
            obj.tiles.remove(tile)

    def _eval_tile_index(self, index):
        goal_bin = self.env.goal_bin
        track_len = len(self.env.track)
        goal_step = track_len/(self.env.num_goal_bins)

        MIN_DISTANCE_TO_GO = 10
        distance = track_len - index
        tile_bin = np.floor(distance/goal_step)

        # print('in tile bin, index', tile_bin, index, flush=True)
        if goal_bin == 0 and distance < MIN_DISTANCE_TO_GO:
            self.env.goal_reached = False
        elif goal_bin == self.env.num_goal_bins - 1 \
            and index < MIN_DISTANCE_TO_GO:
            self.env.goal_reached = False
        elif tile_bin == goal_bin:
            self.env.goal_reached = True
            # print(f'goal bin {goal_bin} reached!', flush=True)


N_CONTROL_POINTS = 12
GENOTYPE_SIZE = N_CONTROL_POINTS * 2

class CarRacingBezier(gym.Env, EzPickle):
    """ Car racing environment with Beziér curves. """
    
    # metadata = {
    #     "render.modes": ["human", "rgb_array", "state_pixels"],
    #     "video.frames_per_second": FPS,
    # }

    def __init__(self,
        n_control_points:           int = 12,
        track_name:                 str = None,
        use_bezier:                 bool = True, 
        show_borders:               bool = True, 
        show_indicators:            bool = True,
        birdseye:                   bool = False, 
        seed:                       int = None,
        fixed_environment:          bool = False,
        animate_zoom:               bool = False,
        min_rad_ratio:              float = 0.333333333,
        max_rad_ratio:              float = 1.0,
        sparse_rewards:             bool = False,
        clip_reward:                float = None,
        num_goal_bins:              int = 24,
        verbose:                    int = 0,
        max_steps:                  int = None,
        variable_episode_lengths:     bool = False,
        distribution_type:          str = 'SB',
        gt_type:                    str = 'CP-1',
        visualize:                  bool = False,
        dense_rewards:              bool = False,
        fps:                        int = 50,
        obs_type:                   str = 'image',
        state_w:                    int = 96,
        state_h:                    int = 96,
        use_f1_tracks:              bool = False,
    ):
        """ Car racing environment initialization. 
        
        Args:
        - n_control_points (int): Number of control points for the Beziér curve.
        - track_name (str): Name of the track to use. If None, a random track is generated.
        - use_bezier (bool): Whether to use Beziér curves or polar coordinates.
        - show_borders (bool): Whether to show borders on the track.
        - show_indicators (bool): Whether to show indicators on the track.
        - birdseye (bool): Whether to use a bird's eye view.
        - seed (int): Seed for the random number generator.
        - fixed_environment (bool): Whether to use a fixed environment.
        - animate_zoom (bool): Whether to animate the zoom.
        - min_rad_ratio (float): Minimum radius ratio for the Beziér curve.
        - max_rad_ratio (float): Maximum radius ratio for the Beziér curve.
        - sparse_rewards (bool): Whether to use sparse rewards.
        - clip_reward (float): Reward clipping value.
        - num_goal_bins (int): Number of goal bins for sparse rewards.
        - verbose (int): Verbosity level.
        - max_steps (int): Maximum number of steps per episode.
        - variable_episode_lengths (bool): Whether to end the episode early.
        - distribution_type (str): How to generate environment. Options are:
            - 'SB': Use seed to generate control_points etc.
            - 'QD': Use QD genotype to generate control_points etc.
        - gt_type (str): How to interpret genotype. Options are:
            - 'CP-T': Use control points with temperature T. E.g. CP-10, CP-3.
            - Default is 'CP-1'---no stretch.
        - use_image_state (bool): Whether to use image state or 1d obs.
        - visualize (bool): Whether to visualize the environment.
        - dense_rewards (bool): Whether to use dense rewards.
        - fps (int): Frames per second.
        - obs_type (str): Observation type. Options are:
            - 'image': Image observation.
            - '1d': 1d observation.
        """
        del visualize  # Unused for this environment
        EzPickle.__init__(self)

        self.level_seed = seed
        self.dense_rewards = dense_rewards
        self.seed(seed)

        self.use_f1_tracks = use_f1_tracks
        
        self.n_control_points = n_control_points
        # We do not take variable control points into account in gt_type
        assert self.n_control_points == N_CONTROL_POINTS  
        self.bezier = use_bezier
        self.fixed_environment = fixed_environment
        self.animate_zoom = animate_zoom
        self.min_rad_ratio = min_rad_ratio
        self.max_rad_ratio = max_rad_ratio

        self.steps = 0
        self.max_steps = max_steps
        self.fps = fps
        self.obs_type = obs_type
        if self.obs_type == '1d-v1':
            obs_length = N_SEGS_FOR_1D_OBS*3 + 7
        elif self.obs_type[0] == 'f':
            res = int(self.obs_type[1:])
            obs_length = (res)**2 + 4  # We have 4 other features currently
            state_h = res*10  # NOTE: we're overwriting the defaults here
            state_w = res*10
        else:
            assert self.obs_type == 'image'

        # For QD-MetaRL 
        self._max_episode_steps = max_steps
        self.variable_episode_lengths = variable_episode_lengths
        self.distribution_type = distribution_type
        if 'CP' in gt_type:
            self.gt_temp = int(gt_type[3:])
        else:
            raise ValueError(f'Unknown genotype type: {gt_type}')
        self.gt_type = gt_type
        self.genotype_size = self.n_control_points * 2
        assert self.genotype_size == GENOTYPE_SIZE  # See above; self.n_control_points

        self.contactListener_keepref = FrictionDetector(self)
        self.world = Box2D.b2World((0, 0), contactListener=self.contactListener_keepref)
        self.viewer = None
        self.invisible_state_window = None
        self.invisible_video_window = None
        self.road = None
        self.car = None
        self.reward = 0.0
        self.prev_reward = 0.0

        # Returns None if track_name is None
        if track_name is None:
            self.preloaded_track = None
        else:
            # global racetracks; 
            from . import racetracks
            self.preloaded_track = racetracks.get_track(track_name)

        if self.use_f1_tracks:
            # global racetracks; 
            from diva.environments.box2d.car_racing_f1 import RACETRACKS

            from . import racetracks
            self.preloaded_tracks = [racetracks.get_track(name) for name in RACETRACKS.keys()]
            self.preloaded_track = self.preloaded_tracks[0]
        else:
            self.preloaded_tracks = None
            self.preloaded_track = None
        
        self.show_borders = show_borders
        self.show_indicators = show_indicators
        self.birdseye = birdseye
        self.verbose = verbose

        self.track_data = None
        self.complexity_info = None

        self.window_h = WINDOW_H
        self.window_w = WINDOW_W
        self.state_h = state_h
        self.state_w = state_w
        self.track_rad = TRACK_RAD
        self.track_width = TRACK_WIDTH
        if self.preloaded_track:
            self.playfield = self.preloaded_track.bounds / SCALE
            self.full_zoom = self.preloaded_track.full_zoom
        else:
            self.playfield = PLAYFIELD
            self.full_zoom = 0.25
        
        self.fd_tile = fixtureDef(
            shape=polygonShape(vertices=[(0, 0), (1, 0), (1, -1), (0, -1)])
        )

        self.action_space = spaces.Box(
            np.array([-1, 0, 0]), np.array([+1, +1, +1]), dtype=np.float32
        )  # steer, gas, brake

        self.use_image_state = (self.obs_type == 'image')
        if self.use_image_state:
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(self.state_h, self.state_w, 3), dtype=np.float32
            )
        else:
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_length,), dtype=np.float32
            )

        self.clip_reward = clip_reward

        # Create goal for sparse rewards
        self.sparse_rewards = sparse_rewards
        self.num_goal_bins = num_goal_bins # 0-indexed
        self.goal_bin = None
        if sparse_rewards:
            self.set_goal()
            self.accumulated_rewards = 0.0

        # For QD+metaRL
        self.genotype = None
        self.control_points = None
        self.set_genotype_info()
        self.size = self.genotype_size

        if self.distribution_type == 'QD':
            self.genotype_set = False

        self.prev_done = False
        self.prev_done_to_return = False

        self.bps = bezier.bernstein_polynomials(degree=3, num_points=40)

    def seed(self, seed=None):
        """ Set seed. """
        if seed is not None:
            seed = int(seed)
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_complexity_info(self):
        if self.complexity_info is None:
            # recompute
            points = ((x,y) for _,_,x,y in self.track)
            return geo_complexity.complexity(points)

        return self.complexity_info

    def set_goal(self, goal_bin=None):
        if goal_bin is None:
            goal_bin = self.goal_bin

        if goal_bin is None:
            self.goal_bin = self.np_random.randint(1,self.num_goal_bins)
        else:
            self.goal_bin = goal_bin

        self.goal_reached = False

    def set_genotype_info(self):
        """ Extract information from genotype and genotype type. """
        
        self.measures = CarRacingMeasures.get_all_measures()

        if 'CP' in self.gt_type:
            # Genotype is simply control point coordinates
            self.genotype_size = GENOTYPE_SIZE
        else:
            raise ValueError(f'Unknown genotype type: {self.gt_type}')
        
        self.genotype_lower_bounds = np.array([0.0] * GENOTYPE_SIZE)
        self.genotype_upper_bounds = np.array([1.0] * GENOTYPE_SIZE)
        self.genotype_bounds = [(l, u) for l, u in  # noqa: E741
                                zip(list(self.genotype_lower_bounds), 
                                    list(self.genotype_upper_bounds))]

    def _destroy(self):
        if not self.road:
            return

        for t in self.road:
            t.userData = t.userData['tile']
            self.world.DestroyBody(t)

        self.world = Box2D.b2World((0, 0), contactListener=self.contactListener_keepref)

        self.road = []
        self.car.destroy()
        self.car = None

    def reset_sparse_state(self):
        if self.sparse_rewards:
            self.accumulated_rewards = 0.0
            self.set_goal()

    def get_measures_info(self, env_name):
        """ Get information on environment measures. """
        return CarRacingMeasures.get_measures_info(env_name)
    
    @staticmethod
    def compute_measures_static(genotype=None, measures=None, gt_type=None, return_pg=False):
        """ Compute measures for a given genotype. """
        # Extract useful properties of genotype
        pg = CarRacingBezier.process_genotype(genotype, gt_type)
        curve, segments = pg.curve, pg.segments

        meas = CarRacingMeasures.compute_measures(
            curve=curve, 
            segments=segments,
            measures=measures,
        )
        
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

        if isinstance(self.genotype, str) and self.genotype == 'f1':
            genotype = None

        # Extract useful properties of genotype
        pg = self._process_genotype(genotype)
        curve, segments = pg.curve, pg.segments


        if isinstance(self.genotype, str) and self.genotype == 'f1':
            curve = self.curve  # (5000, 2); normally curve is (480, 2) 
            # Take every 1 in 10 points on the curv 
            curve = curve[::10]
            segments = None
            # print('curve.shape: {}'.format(curve.shape))

        meas = CarRacingMeasures.compute_measures(
            curve=curve, 
            segments=segments,
            measures=measures,
        )
        
        if return_pg:
            return meas, pg
        else:
            return meas

    def generate_level_from_seed(self, seed):
        """ Generate a level from a given seed. """
        # Set the seed
        if seed is not None:
            self.seed(seed=seed)

        if self.use_f1_tracks:
            idx = self.np_random.randint(0, len(self.preloaded_tracks))
            # print('F1 sampled index: {}'.format(idx))
            self.preloaded_track = self.preloaded_tracks[idx]
            self.playfield = self.preloaded_track.bounds / SCALE
            self.full_zoom = self.preloaded_track.full_zoom
        else:
            # Set control points
            self.control_points = bezier.get_random_points(n=self.n_control_points, 
                                                        scale=self.playfield, 
                                                        np_random=self.np_random)
        # Simply call reset, which will generate environment from the seed
        self.reset()  
    
    @staticmethod
    def genotype_from_seed_static(
            seed, 
            gt_type,
            genotype_lower_bounds,
            genotype_upper_bounds,
            genotype_size):
        """ Generate level from seed. """
        num_attempts = 0
        rng = np.random.default_rng(seed)
        while True:
            # Keep using this seed to generate environments until one is valid
            num_attempts += 1
            sol = rng.uniform(
                low=genotype_lower_bounds,
                high=genotype_upper_bounds,
                size=(genotype_size))
            # Check if solution is valid
            pg = CarRacingBezier.process_genotype(sol, gt_type=gt_type)
            genotype = pg.genotype
            valid, reason = CarRacingBezier.is_valid_genotype(
                pg, gt_type=gt_type)
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
                
                sol = rng.uniform(
                    low=self.genotype_lower_bounds,
                    high=self.genotype_upper_bounds,
                    size=(GENOTYPE_SIZE))

                # Check if solution is valid
                pg = self._process_genotype(sol)
                genotype = pg.genotype

                valid, reason = self.is_valid_genotype(
                    pg, gt_type=self.gt_type)
                del reason 
                if valid:
                    break
                if num_attempts == 100:
                    print('WARNING: Could not sample a valid solution after 100 attempts')
                if num_attempts > 1000:
                    raise RuntimeError("Could not sample a valid solution")        
        return genotype
    
    def _process_genotype(self, genotype):
        """ Extract infoormation from genotype and genotype type. """
        gisn = genotype is None

        assert gisn or len(genotype) == GENOTYPE_SIZE, \
            f'Genotype length {len(genotype)} != {GENOTYPE_SIZE}'
        
        if 'CP' in self.gt_type:
            if gisn:
                control_points = None
                x, y, segments, curve = None, None, None, None
                points = None
            else:
                # Assume unit points
                control_points = np.array(genotype).reshape(-1, 2).astype(np.float32)
                projected_points = self.project_control_points(
                    control_points, scale=1.0, factor=self.gt_temp)
                control_points = bezier.scale_unit_points(projected_points, scale=self.playfield)
                x, y, _, segments, curve = bezier.get_bezier_curve(a=control_points, rad=0.2, edgy=0.2, numpoints=40, return_all=True, bernstein_poly=self.bps)
                points = list(zip(x, y))
        # Return processed genotype
        processed_genotype = {
            'control_points': control_points,
            'points': points,
            'x': x,
            'y': y,
            'segments': segments,
            'curve': curve,
            'genotype': genotype,
            'genotype_lower_bounds': self.genotype_lower_bounds,
            'genotype_upper_bounds': self.genotype_upper_bounds,
            'genotype_bounds': self.genotype_bounds,
            'genotype_size': GENOTYPE_SIZE
        }

        return Namespace(**processed_genotype)
    
    @staticmethod
    def process_genotype(genotype, gt_type='CP-1'):
        """ Extract information from genotype and genotype type. """
        gisn = genotype is None
        size = GENOTYPE_SIZE

        if 'CP' in gt_type:
            gt_temp = int(gt_type[3:])
            if gisn:
                control_points = None
                x, y, segments, curve = None, None, None, None
                points = None
                genotype_lower_bounds = None
                genotype_upper_bounds = None
                genotype_bounds = None
            else:
                # Assume unit points
                try:
                    control_points = np.array(genotype).reshape(-1, 2).astype(np.float32)
                    projected_points = CarRacingBezier.project_control_points(
                        control_points, scale=1.0, factor=gt_temp)
                    control_points = bezier.scale_unit_points(projected_points, scale=PLAYFIELD)
                    x, y, _, segments, curve = bezier.get_bezier_curve(a=control_points, rad=0.2, edgy=0.2, numpoints=40, return_all=True)
                    points = list(zip(x, y))
                    genotype_lower_bounds = np.array([0.0] * size)
                    genotype_upper_bounds = np.array([1.0] * size)
                    genotype_bounds = [(l, u) for l, u in  # noqa: E741
                                        zip(list(genotype_lower_bounds),
                                            list(genotype_upper_bounds))]
                except Exception as exc:
                    print('genotype:', genotype)
                    print('control_points:', control_points)
                    print('projected_points:', projected_points)
                    raise exc
        else:
            raise ValueError(f'Unknown genotype type: {gt_type}')
        
        # Return processed genotype
        processed_genotype = {
            'control_points': control_points,
            'x': x,
            'y': y,
            'points': points,
            'segments': segments,
            'curve': curve,
            'genotype': genotype,
            'genotype_lower_bounds': genotype_lower_bounds,
            'genotype_upper_bounds': genotype_upper_bounds,
            'genotype_bounds': genotype_bounds,
            'genotype_size': size
        }

        return Namespace(**processed_genotype)
        
    @staticmethod
    def is_valid_genotype(processed_genotype, gt_type='CP-1'): 
        """ Check if genotype is valid. """
        genotype = processed_genotype.genotype
        if genotype is None:
            return False, 'genotype is None'
        if len(genotype) != processed_genotype.genotype_size:
            return False, f'genotype length {len(genotype)} != {processed_genotype.genotype_size}'
        if not np.all(np.isfinite(genotype)):
            return False, 'genotype contains non-finite values'
        if not np.all(np.logical_and(
            processed_genotype.genotype_lower_bounds <= genotype,
            genotype <= processed_genotype.genotype_upper_bounds)):
            return False, 'genotype out of bounds'
        return True, None

    def generate_level_from_genotype(self, genotype, gt_type='CP-1'):
        """ Generate a level from a given genotype. """
        if genotype is not None:
            self.genotype = np.array(genotype)
        
        # Process genotype
        pg = self.process_genotype(genotype, gt_type=gt_type)
        self.control_points = pg.control_points

        if genotype is None and self.genotype_set:
            self.reset()
        
        # Set common variables
        self.genotype_size = pg.genotype_size
        self.genotype_lower_bounds = pg.genotype_lower_bounds
        self.genotype_upper_bounds = pg.genotype_upper_bounds
        self.genotype_bounds = pg.genotype_bounds
        self.genotype = pg.genotype
        self.x = pg.x
        self.y = pg.y
        self.segments = pg.segments
        self.curve = pg.curve
        self.points = pg.points

        # Indicate that genotype is set
        if genotype is None:
            # NOTE: This is important because we might pass in a None genotype
            # here after we've already set one
            self.genotype_set = False
        else:
            self.genotype_set = True
            self.reset()
    
    def set_genotype_from_current_task(self):
        """ Set the genotype from the current task. """
        if self.use_f1_tracks:
            self.genotype = 'f1'
            return 'f1'  # No genotype for F1 tracks

        if 'CP' in self.gt_type:
            if self.control_points is not None:
                # Assume control points is at scale of self.playfield
                unprojected_points = self.unproject_control_points(
                    self.control_points, scale=self.playfield, 
                    factor=self.gt_temp)
                self.genotype = bezier.unscale_unit_points(
                    unprojected_points.reshape(-1), scale=self.playfield)
            else:
                print('WARNING: control_points is None, so cannot set genotype from current task')
        else:
            raise ValueError(f'Unknown genotype type: {self.gt_type}')
        return self._process_genotype(self.genotype)

    def reset(self):
        if self.fixed_environment:
            self.seed(self.level_seed)

        self._destroy()
        self.reward = 0.0
        self.prev_reward = 0.0
        self.tile_visited_count = 0
        self.prev_tile_visited_count = 0
        self.t = 0.0
        self.road_poly = []
        self.track_data = None

        self.steps = 0

        # NOTE: we assume reset_task has already been called if need-be, 
        # and that self.control_points has been set to the desired value;
        # which will be used by self._create_track().
        self._create_track()

        beta0, x0, y0 = self.track[0][1:4]
        x0 -= self.x_offset
        y0 -= self.y_offset
        self.car = Car(self.world, beta0, x0, y0)
        self.x0 = x0
        self.y0 = y0

        self.goal_bin = None
        self.reset_sparse_state()
        self.prev_done = False
        self.prev_done_to_return = False

        return self.step(None)[0]

    def reset_task(self, task=None) -> None:
        """
        Reset current task (i.e. genotype, seed, etc.).

        Reset the task, either at random (if task=None) or the given task.
        Should *not* reset the environment.
        """
        # Generate level from specifications
        if self.distribution_type == 'SB':
            # If we're using a seed-based distribution, we need to generate
            # a new seed and then generate the level from that seed
            self.seed(task)
            self.generate_level_from_seed(seed=task)
        elif self.distribution_type == 'QD':
            self.generate_level_from_genotype(genotype=task, gt_type=self.gt_type)
        else:
            raise NotImplementedError(
                f'Unknown distribution type: {self.distribution_type}')
        
        # Create the track, given the current seed
        self._create_track()

        if self.distribution_type == 'SB':
            # Set genotype from current task (generated from seed)
            _ = self.set_genotype_from_current_task()
        
        # Return genotype
        return self.genotype
    
    def get_task(self):
        """ Return the ground truth task. """
        # TODO: more thoughtful implementation
        if hasattr(self, 'genotype') and self.genotype is not None:
            return np.asarray(self.genotype).copy()
        else:
            return np.array((0.0,))

    def _reset_belief(self) -> np.ndarray:
        raise NotImplementedError('Oracle not implemented for MazeEnv.')

    def update_belief(self, state, action) -> np.ndarray:
        raise NotImplementedError('Oracle not implemented for MazeEnv.')

    def get_belief(self):
        raise NotImplementedError('Oracle not implemented for MazeEnv.')

    def step(self, action):
        info = {}

        if self.steps == 1:
            if self.control_points is not None:
                info['env/control_points_mean'] = np.mean(self.control_points)
                info['env/control_points_std'] = np.std(self.control_points)
                info['env/control_points_min'] = np.min(self.control_points)
                info['env/control_points_max'] = np.max(self.control_points)

        st = time.time()
        if action is not None:
            self.car.steer(-action[0])
            self.car.gas(action[1])
            self.car.brake(action[2])
        et = time.time()
        info['time/ES-CRB.step;stear,gas,break'] = et - st

        
        st0 = time.time()
        self.car.step(1.0 / self.fps)
        et0 = time.time()
        info['time/ES-CRB.step;car.step'] = et0 - st0
        
        st0 = time.time()
        self.world.Step(1.0 / self.fps, 6 * 30, 2 * 30)
        et0 = time.time()
        info['time/ES-CRB.step;world.step'] = et0 - st0
        
        self.t += 1.0 / self.fps

        self.steps += 1
        st = time.time()
        if self.use_image_state:
            self.state = self.render("state_pixels")
        else:
            self.state = self.generate_1d_observation()
        et = time.time()
        info['time/ES-CRB.step;render,generate'] = et - st

        st = time.time()
        step_reward = 0
        done = False
        left_playfield = False
        completed_track = False
        if action is not None:  # First step without action, called from reset()
            self.reward -= 0.1

            # Reward for each new segment visted
            self.reward += (self.tile_visited_count - self.prev_tile_visited_count)

            # We actually don't want to count fuel spent, we want car to be faster.
            # self.reward -=  10 * self.car.fuel_spent / ENGINE_POWER
            self.car.fuel_spent = 0.0
            step_reward = self.reward - self.prev_reward
            self.prev_reward = self.reward

            # Provide a large positive reward if car completes track
            if self.tile_visited_count == len(self.track):
                done = True
                completed_track = True
                step_reward = 100
            
            # Provide large negative reward if car leaves playfield
            x, y = self.car.hull.position
            if abs(x) > self.playfield or abs(y) > self.playfield:
                done = True
                step_reward = -100
                left_playfield = True

        # For sparse reward, we provide the accumlated rewards
        if self.sparse_rewards:
            self.accumulated_rewards += step_reward
            revealed_reward = 0
            if self.goal_reached:
                revealed_reward = self.accumulated_rewards
                self.accumulated_rewards = 0.0
                done = True
        else:
            revealed_reward = step_reward

        if self.clip_reward:
            revealed_reward = min(max(revealed_reward, -self.clip_reward), self.clip_reward)

        # End episiode if beyond max_steps, regardless of not self.variable_episode_lengths
        # (i.e. we always return done if we're past max steps)
        done_to_return = done
        if self.max_steps is not None and self.steps > self.max_steps:
            done_to_return = True
        # Otherwise, if we've completed or crashed...
        elif done and self.variable_episode_lengths:
            done_to_return = True
        elif done and not self.variable_episode_lengths:
            done_to_return = False
            # Stop providing rewards after first "done"
            if self.prev_done:
                revealed_reward = 0
            # Otherwise, our revealed reward can be
        
        # If done, we set prev_done flag, so that we know for future steps,
        # if we continue the rollout despite this done, we know that we've 
        # previously finished
        if done:
            self.prev_done = True
        if done_to_return:
            self.prev_done_to_return = True

        # Stuff to log
        vel_x, vel_y = self.car.hull.linearVelocity
        velocity = (vel_x**2 + vel_y**2)**0.5
        info['env/avg_velocity'] = velocity
        distance_from_start = np.linalg.norm(np.array(self.car.hull.position) - np.array([self.x0, self.y0]))
        info['env/avg_distance_from_start'] = distance_from_start

        if len(self.track) * 0.99 <= self.tile_visited_count <= len(self.track) * 1.0:
            info['env/end_100_completion_num_steps'] = self.steps
        if len(self.track) * 0.97 <= self.tile_visited_count <= len(self.track) * 0.99:
            info['env/end_99_completion_num_steps'] = self.steps
        if len(self.track) * 0.93 <= self.tile_visited_count <= len(self.track) * 0.95:
            info['env/end_95_completion_num_steps'] = self.steps
        if len(self.track) * 0.88 <= self.tile_visited_count <= len(self.track) * 0.9:
            info['env/end_90_completion_num_steps'] = self.steps
        if len(self.track) * 0.78 <= self.tile_visited_count <= len(self.track) * 0.8:
            info['env/end_80_completion_num_steps'] = self.steps
        if len(self.track) * 0.48 <= self.tile_visited_count <= len(self.track) * 0.5:
            info['env/end_50_completion_num_steps'] = self.steps
        if len(self.track) * 0.27 <= self.tile_visited_count <= len(self.track) * 0.25:
            info['env/end_25_completion_num_steps'] = self.steps

        if done_to_return:
            info['env/end_segments_reached'] = self.tile_visited_count
            info['env/end_perc_segments_reached'] = self.tile_visited_count / len(self.track)
            info['env/end_velocity'] = velocity
            info['env/end_distance_from_start'] = distance_from_start
            info['env/end_100_completion_perc'] = self.tile_visited_count >= len(self.track) - 1
            info['env/end_99_completion_perc'] = self.tile_visited_count >= len(self.track) * 0.99
            info['env/end_95_completion_perc'] = self.tile_visited_count >= len(self.track) * 0.95
            info['env/end_90_completion_perc'] = self.tile_visited_count >= len(self.track) * 0.9
            info['env/end_80_completion_perc'] = self.tile_visited_count >= len(self.track) * 0.8
            info['env/end_50_completion_perc'] = self.tile_visited_count >= len(self.track) * 0.5
            info['env/end_25_completion_perc'] = self.tile_visited_count >= len(self.track) * 0.25

        self.prev_tile_visited_count = self.tile_visited_count

        et = time.time()
        info['time/ES-CRB.step;REST'] = et - st
        return self.state, revealed_reward, done_to_return, info
    
    def obs_flattened_image(self):
        img = self.render("state_pixels")
        img = img[:,:,1]  # Only get green!
        img = Image.fromarray(img)
        assert img.size == (self.state_h, self.state_h)
        wh = (self.state_w + self.state_h) // 2
        img = img.resize((wh // 10, wh // 10), Image.BILINEAR)
        img = np.array(img).astype(np.float32)
        img = img.flatten()

        # 1. Position of the car
        x, y = self.car.hull.position
        # 2. Velocity of the car
        vel_x, vel_y = self.car.hull.linearVelocity
        velocity = (vel_x**2 + vel_y**2)**0.5
        # 3. Angle of the car
        angle = self.car.hull.angle
        extras = np.array([x, y, velocity, angle])
        
        # concatenate and keep 1d
        img = np.concatenate([img, extras])
        img = np.array(img)
        return img
    
    def obs_1d_v1(self):

        track_loop = self.track_loop

        # 1. Position of the car
        x, y = self.car.hull.position

        # 2. Velocity of the car
        vel_x, vel_y = self.car.hull.linearVelocity
        velocity = (vel_x**2 + vel_y**2)**0.5

        # 3. Angle of the car
        angle = self.car.hull.angle

        # 4. Track ahead (relative position between car and next 10 segments)
        # Find the index of the track segment closest to the car's current position
        pos = np.array([x, y])
        # compute distance between track_loop (track_length, 2) and pos (2,) for each track segment
        distances = np.linalg.norm(track_loop - pos, axis=1)
        # find argmin distance
        posi = np.argmin(distances)
        # get the next N_SEGS_FOR_1D_OBS segments
        track_ahead = track_loop[posi:posi+N_SEGS_FOR_1D_OBS].flatten().tolist()

        # 4.5 Track ahead (relative angle between car and next segments)
        angles_diff = self.segment_angles_relative_to_car(pos, posi)

        # 5. Steps taken
        steps_taken = self.steps

        # 6. Tile visited count
        tile_visited = self.tile_visited_count

        # 7. Distance from track center
        distance_center = self.distance_from_track_center(pos, posi)

        # Combine all observations into a single 1D vector
        observation = [x, y, velocity, angle, *track_ahead, *angles_diff, 
                       steps_taken, tile_visited, distance_center]
        
        return np.array(observation)

    def generate_1d_observation(self):
        if self.obs_type[0] == 'f':
            return self.obs_flattened_image()
        elif self.obs_type == '1d-v1':
            return self.obs_1d_v1()
        else:
            raise ValueError(f'Unrecognized value for self.obs_type: {self.obs_type}')
        
    @staticmethod
    def project_control_points(points, scale=1.0, factor=1.0):
        midpoint = scale / 2.0

        # Calculate coordinates relative to the center
        x_rel = points[:, 0] - midpoint
        y_rel = points[:, 1] - midpoint

        # Apply the scaling factor
        x_scaled = midpoint + factor * x_rel
        y_scaled = midpoint + factor * y_rel

        # Clamp the coordinates to the square boundaries
        x_clamped = np.clip(x_scaled, 0, scale)
        y_clamped = np.clip(y_scaled, 0, scale)

        # Return the transformed points
        transformed_points = np.column_stack((x_clamped, y_clamped))
        return transformed_points

    @staticmethod
    def unproject_control_points(points, scale=1.0, factor=1.0):
        midpoint = scale / 2.0

        # Calculate coordinates relative to the center
        x_rel = points[:, 0] - midpoint
        y_rel = points[:, 1] - midpoint

        # Apply the inverse scaling factor
        x_original = midpoint + x_rel / factor
        y_original = midpoint + y_rel / factor

        # Return the original points
        original_points = np.column_stack((x_original, y_original))
        return original_points

    def segment_angles_relative_to_car(self, pos, closest_index):
        car_angle = self.car.hull.angle  # Angle of the car in radians
        track_loop = self.track_loop   # All points along the track

        # Compute the angle of each segment relative to the car's angle
        angles_diff = []
        for i in range(closest_index, closest_index + N_SEGS_FOR_1D_OBS):
            p1 = track_loop[i % len(track_loop)]
            p2 = track_loop[(i + 1) % len(track_loop)]

            # Calculate the direction of the segment
            segment_direction = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])

            # Calculate the difference and normalize it
            angle_diff = segment_direction - car_angle
            angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi  # Normalize to -π to π

            angles_diff.append(angle_diff)

        return angles_diff

    def distance_from_track_center(self, pos, closest_index):
        # Identify the points around the closest track point for interpolation
        p_closest = self.track_loop[closest_index]
        p_next = self.track_loop[closest_index + 1]

        # Calculate the perpendicular distance from point (x, y) to the line segment (p_closest-p_next)
        def point_to_line_distance(p1, p2, p0):
            """ Calculate the perpendicular distance from point p0 to the line segment p1-p2 """
            return np.abs(np.cross(p2-p1, p1-p0)) / np.linalg.norm(p2-p1)

        # Calculate the distance from the car to the closest segment
        distance_center = point_to_line_distance(p_closest, p_next, pos)

        return distance_center
    
    @property
    def level_rendering(self):
        """ Render the level. """
        val = self.render(mode='level')
        return val

    def _create_track(self, control_points=None, show_borders=None):
        if self.control_points is not None:
            control_points = self.control_points

        if self.bezier:
            return self._create_track_bezier(
                control_points=control_points, 
                show_borders=show_borders)
        else:
            t = 0
            reset_random = False
            while True:
                t += 1
                if t > 10:
                    reset_random = True
                    break

                success = self._create_track_polar(
                    control_points=control_points,
                    show_borders=show_borders)
                if success:
                    return success

        if reset_random:
            t = 0
            while True:
                t += 1
                success = self._create_track_polar(
                    show_borders=show_borders)
                if success:
                    return success

    def _create_track_bezier(self, control_points=None, show_borders=None):
        if show_borders is None:
            show_borders = self.show_borders
        else:
            show_borders = show_borders

        # Create random bezier curve
        track = []
        self.road = []

        if self.preloaded_track is not None:
            points = self.preloaded_track.xy  
            x,y = zip(*points)
            # Each row is a point on the curve and the two columns are the 
            # x and y coordinates respectively.
            self.curve = np.vstack((x, y)).T
        elif control_points is not None:
            a = np.array(control_points)
            x, y, _, self.segments, self.curve = bezier.get_bezier_curve(a=a, rad=0.2, edgy=0.2, numpoints=40, return_all=True)
            self.edgy = 0.2
            self.track_data = a
        else:
            a = bezier.get_random_points(n=self.n_control_points, scale=self.playfield, np_random=self.np_random)
            self.control_points = a
            self.edgy = 0.2
            x, y, _, self.segments, self.curve = bezier.get_bezier_curve(a=a, rad=0.2, edgy=0.2, numpoints=40, return_all=True)
            self.track_data = a

        min_x, max_x = x[-1], x[-1]
        min_y, max_y = y[-1], y[-1]

        points = list(zip(x,y))
        self.points = points
        betas = []
        for i, p in enumerate(points[:-1]):
            x1, y1 = points[i]
            x2, y2 = points[i+1]
            dx = x2 - x1
            dy = y2 - y1
            if (dx == dy == 0):
                continue

            # alpha = math.atan(dy/(dx+1e-5))
            alpha = np.arctan2(dy, dx)
            beta = math.pi/2 + alpha

            track.append((alpha, beta, x1, y1))
            betas.append(beta)

            min_x = min(x1, min_x)
            min_y = min(y1, min_y)
            max_x = max(x1, max_x)
            max_y = max(y1, max_y)

        x_offset = min_x + (max_x - min_x)/2
        y_offset = min_y + (max_y - min_y)/2
        self.x_offset = x_offset
        self.y_offset = y_offset

        betas = np.array(betas)
        abs_dbeta = abs(betas[1:] - betas[0:-1])
        mean_abs_dbeta = abs_dbeta.mean()
        std_abs_dbeta = abs_dbeta.std()
        one_dev_dbeta = mean_abs_dbeta + std_abs_dbeta/2

        # Red-white border on hard turns
        border = [False] * len(track)
        if show_borders:
            for i in range(len(track)):
                good = True
                oneside = 0
                for neg in range(BORDER_MIN_COUNT):
                    beta1 = track[i - neg - 0][1]
                    beta2 = track[i - neg - 1][1]
                    good &= abs(beta1 - beta2) > mean_abs_dbeta
                    oneside += np.sign(beta1 - beta2)
                good &= abs(oneside) == BORDER_MIN_COUNT
                border[i] = good
            for i in range(len(track)):
                for neg in range(BORDER_MIN_COUNT):
                    border[i - neg] |= border[i]

        # Create tiles
        for i in range(len(track)):
            alpha1, beta1, x1, y1 = track[i]

            alpha2, beta2, x2, y2 = track[i - 1]

            road1_l = (
                x1 - TRACK_WIDTH * math.cos(beta1) - x_offset,
                y1 - TRACK_WIDTH * math.sin(beta1) - y_offset,
            )
            road1_r = (
                x1 + TRACK_WIDTH * math.cos(beta1) - x_offset,
                y1 + TRACK_WIDTH * math.sin(beta1) - y_offset,
            )
            road2_l = (
                x2 - TRACK_WIDTH * math.cos(beta2) - x_offset,
                y2 - TRACK_WIDTH * math.sin(beta2) - y_offset,
            )
            road2_r = (
                x2 + TRACK_WIDTH * math.cos(beta2) - x_offset,
                y2 + TRACK_WIDTH * math.sin(beta2) - y_offset,
            )
            vertices = [road1_l, road1_r, road2_r, road2_l]

            try:
                self.fd_tile.shape.vertices = vertices
            except:
                pass
            t = self.world.CreateStaticBody(fixtures=self.fd_tile)
            # t.userData = t
            t.userData = {
                'tile': t,
                'index': i
            }
            c = 0.01 * (i % 3)
            t.color = [ROAD_COLOR[0] + c, ROAD_COLOR[1] + c, ROAD_COLOR[2] + c]
            t.road_visited = False
            t.road_friction = 1.0
            t.fixtures[0].sensor = True
            self.road_poly.append(([road1_l, road1_r, road2_r, road2_l], t.color))
            self.road.append(t)

            if self.show_borders and border[i]:
                side = np.sign(beta2 - beta1)
                b1_l = (
                    x1 + side * TRACK_WIDTH * math.cos(beta1) - x_offset,
                    y1 + side * TRACK_WIDTH * math.sin(beta1) - y_offset,
                )
                b1_r = (
                    x1 + side * (TRACK_WIDTH + BORDER) * math.cos(beta1) - x_offset,
                    y1 + side * (TRACK_WIDTH + BORDER) * math.sin(beta1) - y_offset,
                )
                b2_l = (
                    x2 + side * TRACK_WIDTH * math.cos(beta2) - x_offset,
                    y2 + side * TRACK_WIDTH * math.sin(beta2) - y_offset,
                )
                b2_r = (
                    x2 + side * (TRACK_WIDTH + BORDER) * math.cos(beta2) - x_offset,
                    y2 + side * (TRACK_WIDTH + BORDER) * math.sin(beta2) - y_offset,
                )
                self.road_poly.append(
                    ([b1_l, b1_r, b2_r, b2_l], (1, 1, 1) if i % 2 == 0 else (1, 0, 0))
                )
        self.track = track

        self.complexity_info = geo_complexity.complexity(points)

        self.track_numpy = np.array(self.track)[:, 2:]

        # concatenate two of them together
        self.track_loop = np.concatenate((self.track_numpy, self.track_numpy[:N_SEGS_FOR_1D_OBS+2]), axis=0)

        return True

    def _create_track_polar(self, control_points=None, show_borders=None):
        if show_borders is None:
            show_borders = self.show_borders
        else:
            show_borders = show_borders

        CHECKPOINTS = self.n_control_points

        self.x_offset = 0
        self.y_offset = 0

        min_rad = TRACK_RAD*self.min_rad_ratio
        max_rad = TRACK_RAD*self.max_rad_ratio

        # Create checkpoints
        if control_points is not None:
            checkpoints = control_points
            self.start_alpha = 2 * math.pi * (-0.5) / self.n_control_points
        else:
            checkpoints = []
            for c in range(CHECKPOINTS):
                noise = self.np_random.uniform(0, 2 * math.pi * 1 / CHECKPOINTS)
                alpha = 2 * math.pi * c / CHECKPOINTS + noise
                rad = self.np_random.uniform(min_rad, max_rad)

                if c == 0:
                    alpha = 0
                    rad = 1.5 * TRACK_RAD
                if c == CHECKPOINTS - 1:
                    alpha = 2 * math.pi * c / CHECKPOINTS
                    self.start_alpha = 2 * math.pi * (-0.5) / CHECKPOINTS
                    rad = 1.5 * TRACK_RAD

                checkpoints.append((alpha, rad * math.cos(alpha), rad * math.sin(alpha)))

        self.track_data = checkpoints

        self.road = []

        # Go from one checkpoint to another to create track
        # x, y, beta = 1.5 * TRACK_RAD, 0, 0
        _,x,y = checkpoints[0]
        beta = 0
        dest_i = 0
        laps = 0
        track = []
        no_freeze = 2500
        visited_other_side = False
        while True:
            alpha = math.atan2(y, x)
            if visited_other_side and alpha > 0:
                laps += 1
                visited_other_side = False
            if alpha < 0:
                visited_other_side = True
                alpha += 2 * math.pi

            while True:  # Find destination from checkpoints
                failed = True

                while True:
                    dest_alpha, dest_x, dest_y = checkpoints[dest_i % len(checkpoints)]
                    if alpha <= dest_alpha:
                        failed = False
                        break
                    dest_i += 1
                    if dest_i % len(checkpoints) == 0:
                        break

                if not failed:
                    break

                alpha -= 2 * math.pi
                continue

            r1x = math.cos(beta)
            r1y = math.sin(beta)
            p1x = -r1y
            p1y = r1x
            dest_dx = dest_x - x  # vector towards destination
            dest_dy = dest_y - y
            # destination vector projected on rad:
            proj = r1x * dest_dx + r1y * dest_dy
            while beta - alpha > 1.5 * math.pi:
                beta -= 2 * math.pi
            while beta - alpha < -1.5 * math.pi:
                beta += 2 * math.pi
            prev_beta = beta
            proj *= SCALE
            if proj > 0.3:
                beta -= min(TRACK_TURN_RATE, abs(0.001 * proj))
            if proj < -0.3:
                beta += min(TRACK_TURN_RATE, abs(0.001 * proj))
            x += p1x * TRACK_DETAIL_STEP
            y += p1y * TRACK_DETAIL_STEP
            track.append((alpha, prev_beta * 0.5 + beta * 0.5, x, y))
            if laps > 4:
                break
            no_freeze -= 1
            if no_freeze == 0:
                break

        # Find closed loop range i1..i2, first loop should be ignored, second is OK
        i1, i2 = -1, -1
        i = len(track)
        while True:
            i -= 1
            if i == 0:
                return False  # Failed
            pass_through_start = (
                track[i][0] > self.start_alpha and track[i - 1][0] <= self.start_alpha
            )
            if pass_through_start and i2 == -1:
                i2 = i
            elif pass_through_start and i1 == -1:
                i1 = i
                break
        assert i1 != -1
        assert i2 != -1

        track = track[i1 : i2 - 1]

        # Red-white border on hard turns
        border = [False] * len(track)
        if show_borders:
            for i in range(len(track)):
                good = True
                oneside = 0
                for neg in range(BORDER_MIN_COUNT):
                    beta1 = track[i - neg - 0][1]
                    beta2 = track[i - neg - 1][1]
                    good &= abs(beta1 - beta2) > TRACK_TURN_RATE * 0.2
                    oneside += np.sign(beta1 - beta2)
                good &= abs(oneside) == BORDER_MIN_COUNT
                border[i] = good
            for i in range(len(track)):
                for neg in range(BORDER_MIN_COUNT):
                    border[i - neg] |= border[i]

        # Create tiles
        for i in range(len(track)):
            alpha1, beta1, x1, y1 = track[i]
            alpha2, beta2, x2, y2 = track[i - 1]
            road1_l = (
                x1 - TRACK_WIDTH * math.cos(beta1),
                y1 - TRACK_WIDTH * math.sin(beta1),
            )
            road1_r = (
                x1 + TRACK_WIDTH * math.cos(beta1),
                y1 + TRACK_WIDTH * math.sin(beta1),
            )
            road2_l = (
                x2 - TRACK_WIDTH * math.cos(beta2),
                y2 - TRACK_WIDTH * math.sin(beta2),
            )
            road2_r = (
                x2 + TRACK_WIDTH * math.cos(beta2),
                y2 + TRACK_WIDTH * math.sin(beta2),
            )
            vertices = [road1_l, road1_r, road2_r, road2_l]
            self.fd_tile.shape.vertices = vertices
            t = self.world.CreateStaticBody(fixtures=self.fd_tile)
            t.userData = t
            c = 0.01 * (i % 3)
            t.color = [ROAD_COLOR[0] + c, ROAD_COLOR[1] + c, ROAD_COLOR[2] + c]
            t.road_visited = False
            t.road_friction = 1.0
            t.fixtures[0].sensor = True
            self.road_poly.append(([road1_l, road1_r, road2_r, road2_l], t.color))
            self.road.append(t)
            if border[i]:
                side = np.sign(beta2 - beta1)
                b1_l = (
                    x1 + side * TRACK_WIDTH * math.cos(beta1),
                    y1 + side * TRACK_WIDTH * math.sin(beta1),
                )
                b1_r = (
                    x1 + side * (TRACK_WIDTH + BORDER) * math.cos(beta1),
                    y1 + side * (TRACK_WIDTH + BORDER) * math.sin(beta1),
                )
                b2_l = (
                    x2 + side * TRACK_WIDTH * math.cos(beta2),
                    y2 + side * TRACK_WIDTH * math.sin(beta2),
                )
                b2_r = (
                    x2 + side * (TRACK_WIDTH + BORDER) * math.cos(beta2),
                    y2 + side * (TRACK_WIDTH + BORDER) * math.sin(beta2),
                )
                self.road_poly.append(
                    ([b1_l, b1_r, b2_r, b2_l], (1, 1, 1) if i % 2 == 0 else (1, 0, 0))
                )
        self.track = track

        return True

    def render(self, mode="human"):
        assert mode in ["human", "state_pixels", "rgb_array", "level", "sketch"]
        if self.viewer is None:
            from diva.environments.box2d import rendering

            self.viewer = rendering.Viewer(WINDOW_W, WINDOW_H)
            self.score_label = pyglet.text.Label(
                "0000",
                font_size=36,
                x=20,
                y=WINDOW_H * 2.5 / 40.00,
                anchor_x="left",
                anchor_y="center",
                color=(255, 255, 255, 255),
            )
            self.transform = rendering.Transform()

        if "t" not in self.__dict__:
            return  # reset() not called yet

        # Animate zoom first second:
        if self.birdseye or mode in ['level', 'sketch']:
            zoom_coef = self.full_zoom
        else:
            zoom_coef = ZOOM
        if self.animate_zoom:
            zoom = 0.1 * SCALE * max(1 - self.t, 0) + zoom_coef * SCALE * min(self.t, 1)
        else:
            zoom = zoom_coef * SCALE

        scroll_x = self.car.hull.position[0]
        scroll_y = self.car.hull.position[1]
        angle = -self.car.hull.angle
        vel = self.car.hull.linearVelocity
        if np.linalg.norm(vel) > 0.5:
            angle = math.atan2(vel[0], vel[1])
        self.transform.set_scale(zoom, zoom)

        if self.birdseye or mode in ['level', 'sketch']:
            self.transform.set_translation(
                WINDOW_W / 2,
                WINDOW_H / 2,
            )
            self.transform.set_rotation(0)
        else:
            self.transform.set_translation(
                WINDOW_W / 2
                - (scroll_x * zoom * math.cos(angle) - scroll_y * zoom * math.sin(angle)),
                WINDOW_H / 4
                - (scroll_x * zoom * math.sin(angle) + scroll_y * zoom * math.cos(angle)),
            )
            self.transform.set_rotation(angle)

        self.car.draw(self.viewer, mode != "state_pixels") # , translation=(0,0), angle=0)

        arr = None
        win = self.viewer.window
        win.switch_to()
        win.dispatch_events()

        win.clear()
        t = self.transform
        if mode == "rgb_array":
            VP_W = VIDEO_W
            VP_H = VIDEO_H
        elif mode in ["state_pixels", "sketch"]:
            VP_W = self.state_w
            VP_H = self.state_h
        else:
            pixel_scale = 1
            if hasattr(win.context, "_nscontext"):
                pixel_scale = (
                    win.context._nscontext.view().backingScaleFactor()
                )  # pylint: disable=protected-access
            VP_W = int(pixel_scale * WINDOW_W)
            VP_H = int(pixel_scale * WINDOW_H)

        gl.glViewport(0, 0, VP_W, VP_H)
        t.enable()
        self.render_road()
        for geom in self.viewer.onetime_geoms:
            geom.render()
        self.viewer.onetime_geoms = []
        t.disable()

        if mode not in ['level', 'sketch'] and self.show_indicators:
            self.render_indicators(WINDOW_W, WINDOW_H)

        if mode == "human":
            win.flip()
            return self.viewer.isopen

        image_data = (
            pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
        )
        # arr = np.fromstring(image_data.get_data(), dtype=np.uint8, sep="")
        arr = np.frombuffer(image_data.get_data(), dtype=np.uint8)
        arr = arr.reshape(VP_H, VP_W, 4)
        arr = arr[::-1, :, 0:3]

        return arr

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def render_road(self):
        colors = [0.4, 0.8, 0.4, 1.0] * 4
        polygons_ = [
            +self.playfield,
            +self.playfield,
            0,
            +self.playfield,
            -self.playfield,
            0,
            -self.playfield,
            -self.playfield,
            0,
            -self.playfield,
            +self.playfield,
            0,
        ]

        k = self.playfield / 20.0
        colors.extend([0.4, 0.9, 0.4, 1.0] * 4 * 20 * 20)
        for x in range(-20, 20, 2):
            for y in range(-20, 20, 2):
                polygons_.extend(
                    [
                        k * x + k,
                        k * y + 0,
                        0,
                        k * x + 0,
                        k * y + 0,
                        0,
                        k * x + 0,
                        k * y + k,
                        0,
                        k * x + k,
                        k * y + k,
                        0,
                    ]
                )

        for poly, color in self.road_poly:
            colors.extend([color[0], color[1], color[2], 1] * len(poly))
            for p in poly:
                polygons_.extend([p[0], p[1], 0])

        vl = pyglet.graphics.vertex_list(
            len(polygons_) // 3, ("v3f", polygons_), ("c4f", colors)  # gl.GL_QUADS,
        )
        vl.draw(gl.GL_QUADS)
        vl.delete()

    def render_indicators(self, W, H):
        s = W / 40.0
        h = H / 40.0
        colors = [0, 0, 0, 1] * 4
        polygons = [W, 0, 0, W, 5 * h, 0, 0, 5 * h, 0, 0, 0, 0]

        def vertical_ind(place, val, color):
            colors.extend([color[0], color[1], color[2], 1] * 4)
            polygons.extend(
                [
                    place * s,
                    h + h * val,
                    0,
                    (place + 1) * s,
                    h + h * val,
                    0,
                    (place + 1) * s,
                    h,
                    0,
                    (place + 0) * s,
                    h,
                    0,
                ]
            )

        def horiz_ind(place, val, color):
            colors.extend([color[0], color[1], color[2], 1] * 4)
            polygons.extend(
                [
                    (place + 0) * s,
                    4 * h,
                    0,
                    (place + val) * s,
                    4 * h,
                    0,
                    (place + val) * s,
                    2 * h,
                    0,
                    (place + 0) * s,
                    2 * h,
                    0,
                ]
            )

        true_speed = np.sqrt(
            np.square(self.car.hull.linearVelocity[0])
            + np.square(self.car.hull.linearVelocity[1])
        )

        vertical_ind(5, 0.02 * true_speed, (1, 1, 1))
        vertical_ind(7, 0.01 * self.car.wheels[0].omega, (0.0, 0, 1))  # ABS sensors
        vertical_ind(8, 0.01 * self.car.wheels[1].omega, (0.0, 0, 1))
        vertical_ind(9, 0.01 * self.car.wheels[2].omega, (0.2, 0, 1))
        vertical_ind(10, 0.01 * self.car.wheels[3].omega, (0.2, 0, 1))
        horiz_ind(20, -10.0 * self.car.wheels[0].joint.angle, (0, 1, 0))
        horiz_ind(30, -0.8 * self.car.hull.angularVelocity, (1, 0, 0))
        vl = pyglet.graphics.vertex_list(
            len(polygons) // 3, ("v3f", polygons), ("c4f", colors)  # gl.GL_QUADS,
        )
        vl.draw(gl.GL_QUADS)
        vl.delete()
        self.score_label.text = "%04i" % self.reward
        self.score_label.draw()


if __name__ == "__main__":
    from pyglet.window import key

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--track-name', 
        type=str, 
        default=None, help='Name of preexisting track.')
    parser.add_argument(
        '--birdseye', 
        action='store_true',
        default=False, help='Show a fixed birdseye view of track.')
    parser.add_argument(
        '--seed', 
        type=int,
        default=None, help='PRNG seed.')
    args = parser.parse_args()

    a = np.array([0.0, 0.0, 0.0])

    def key_press(k, mod):
        global restart
        if k == 0xFF0D:
            restart = True
        if k == key.LEFT:
            a[0] = -1.0
        if k == key.RIGHT:
            a[0] = +1.0
        if k == key.UP:
            a[1] = +1.0
        if k == key.DOWN:
            a[2] = +0.8  # set 1.0 for wheels to block to zero rotation

    def key_release(k, mod):
        if k == key.LEFT and a[0] == -1.0:
            a[0] = 0
        if k == key.RIGHT and a[0] == +1.0:
            a[0] = 0
        if k == key.UP:
            a[1] = 0
        if k == key.DOWN:
            a[2] = 0

    env = CarRacingBezier(track_name=args.track_name, birdseye=args.birdseye, seed=args.seed)
    env.render()
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release
    record_video = False
    if record_video:
        from gym.wrappers.monitor import Monitor

        env = Monitor(env, "videos/", force=True)
    isopen = True
    while isopen:
        env.reset()
        total_reward = 0.0
        steps = 0
        restart = False
        while True:
            s, r, done, info = env.step(a)
            total_reward += r
            if steps % 200 == 0 or done:
                print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
                print("step {} total_reward {:+0.2f}".format(steps, total_reward))
            steps += 1
            isopen = env.render()
            if done or restart or isopen == False:
                break
    env.close()
