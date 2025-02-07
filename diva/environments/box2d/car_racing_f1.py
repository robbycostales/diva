# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .car_racing_bezier import CarRacingBezier
from .racetracks import RaceTrack, formula1


def set_global(name, value):
    globals()[name] = value


RACETRACKS = dict([(name, cls) for name, cls in formula1.__dict__.items() if isinstance(cls, RaceTrack)])


def _create_constructor(track):
	def constructor(self, **kwargs):
		return CarRacingBezier.__init__(self, 
			track_name=track.name,
			**kwargs)
	return constructor


for name, track in RACETRACKS.items():
	class_name = f"CarRacingF1-{track.name}"
	env = type(class_name, (CarRacingBezier, ), {
	    "__init__": _create_constructor(track),
	})
	set_global(class_name, env)