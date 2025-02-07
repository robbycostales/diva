import warnings

from gym.envs.registration import register
from loguru import logger

from .box2d.car_racing_bezier import CarRacingBezier
from .box2d.car_racing_f1 import RACETRACKS

warnings.filterwarnings('ignore', category=UserWarning, module='gym.envs.registration')

logger.info('Registering environments!')



# DeepMind Alchemy
# -----------------------------------------------------------------------------

register(
    id='AlchemyRandom-v0',
    entry_point='environments.alchemy.alchemy_qd:ExtendedSymbolicAlchemy',
    kwargs={'env_type': 'random',
	        'distribution_type': 'SB'}
)

# QD

register(
    id='AlchemyRandomQD-v0',
    entry_point='environments.alchemy.alchemy_qd:ExtendedSymbolicAlchemy',
    kwargs={'env_type': 'random',
            'distribution_type': 'QD'}
)

# CarRacing (from DCD)
# -----------------------------------------------------------------------------

register(
    'CarRacing-Bezier-v0',
    entry_point='environments.box2d.car_racing_bezier:CarRacingBezier',
    kwargs={
	    'distribution_type': 'SB',
    },
)

register(
    'CarRacing-F1-v0',
    entry_point='environments.box2d.car_racing_bezier:CarRacingBezier',
    kwargs={
	    'distribution_type': 'SB',
        'use_f1_tracks': True,
    },
)
	
# QD

register(
    'CarRacing-BezierQD-v0',
    entry_point='environments.box2d.car_racing_bezier:CarRacingBezier',
    kwargs={
	    'distribution_type': 'QD',
    },
)

# -----------------------------------------------------------------------------
# ToyGrid envs
# -----------------------------------------------------------------------------

register(
    'ToyGrid-v0',
    entry_point='environments.toygrid.toygrid:ToyGrid',
	kwargs={
		'max_steps': 30,
		'distribution_type': 'SB',
		'goal_sampler': 'uniform',
		'goal_sampler_region': 'left',
    },
)

# QD

register(
    'ToyGridQD-v0',
    entry_point='environments.toygrid.toygrid:ToyGrid',
	kwargs={
		'max_steps': 30,
		'distribution_type': 'QD',
    },
)