from typing import Optional

import gin
import numpy as np
import ribs
from ribs.emitters import EvolutionStrategyEmitter

from diva.components.qd.emitters.cma_es import (
    CMAEvolutionStrategy as _CMAEvolutionStrategy,
)


@gin.configurable(denylist=["archive", "x0", "seed"])
class EvolutionStrategyCustomEmitter(EvolutionStrategyEmitter):
    """Implementation of MAP-Elites for env gen.
    Args:
        archive: Archive to store the solutions.
        x0: Initial solution. Only used for solution_dim.
        bounds: Bounds of the solution space. Pass None to
            indicate there are no bounds. Alternatively, pass an array-like to
            specify the bounds for each dim. Each element in this array-like can
            be None to indicate no bound, or a tuple of
            ``(lower_bound, upper_bound)``, where ``lowerbound`` or
            ``upper_bound`` may be None to indicate no bound. (default: None)
        seed: Random seed. (default None)
        batch_size: Number of solutions to return in :meth:`ask`.
        initial_population: Size of the initial population before starting to
            mutate elites from the archive.
    """

    def __init__(self,
                 archive: ribs.archives.ArchiveBase,
                 x0: np.ndarray,
                 bounds: Optional['array-like'] = None,  # noqa: F821 # type: ignore
                 seed: int = None,
                 sigma0: float = 1.0,
                 batch_size: int = gin.REQUIRED,
                 initial_population: int = gin.REQUIRED):
        super().__init__(
            archive, 
            x0=x0, 
            sigma0=sigma0,
            bounds=bounds, 
            batch_size=batch_size)
        self.rng = np.random.default_rng(seed)
        self.initial_population = initial_population
        self.sols_emitted = 0

        # Reset opt to use your custom cma_es opt
        opt_seed = None if seed is None else self.rng.integers(10_000)
        self._opt = _CMAEvolutionStrategy(
            sigma0=sigma0,
            solution_dim=self._solution_dim,
            batch_size=batch_size,
            seed=opt_seed,
            dtype=self.archive.dtype)
        self._opt.reset(self._x0)
        self._batch_size = self._opt.batch_size
        self._last_ask_random = False

    def ask(self):

        if self.sols_emitted < self.initial_population:
            # Still generating random solutions to fill initial_population...
            self._last_ask_random = True
            self.sols_emitted += self.batch_size
            # Generate random float solutions
            sols = self.rng.random(size=(self.batch_size, self.solution_dim))
            # Scale and shift values to fit within the specified bounds
            sols = self.lower_bounds + (self.upper_bounds - self.lower_bounds) * sols
            return sols

        self._last_ask_random = False
        self.sols_emitted += self.batch_size
        
        ##### NOTE: Below line is the only line in the original function ######
        raw_sols = self._opt.ask()
        #######################################################################

        # # Convert to integers by rounding  # TODO: make this functionality an argument
        # sols = np.rint(raw_sols).astype(int)

        # clip each row in raw_sols to the bounds
        sols = np.clip(raw_sols, self.lower_bounds, self.upper_bounds)

        return sols
    
    def tell(self, 
             solution,
             objective,
             measures,
             add_info,
             metadata=None):
        if self._last_ask_random:  # No update on random solutions from last `ask`
            return
        if metadata is not None:
            raise NotImplementedError("Metadata is not currently supported.")
        super().tell(solution, objective, measures, add_info)
