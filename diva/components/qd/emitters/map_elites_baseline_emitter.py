import copy
from typing import Optional

import gin
import numpy as np
import ribs
from ribs.emitters import EmitterBase


@gin.configurable(denylist=["archive", "x0", "seed"])
class MapElitesBaselineEmitter(EmitterBase):
    """Implementation of MAP-Elites which generates solutions corresponding to
    mazes.
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
        num_objects: Solutions will be generated as ints between
            [0, num_objects)
        batch_size: Number of solutions to return in :meth:`ask`.
        initial_population: Size of the initial population before starting to
            mutate elites from the archive.
        mutation_k: Number of positions in the solution to mutate. Should be
            less than solution_dim.
    """

    def __init__(self,
                 archive: ribs.archives.ArchiveBase,
                 x0: np.ndarray,
                 bounds: Optional['array-like'] = None,  # noqa: F821 # type: ignore
                 seed: int = None,
                 num_objects: int = gin.REQUIRED,
                 batch_size: int = gin.REQUIRED,
                 initial_population: int = gin.REQUIRED,
                 mutation_k: int = gin.REQUIRED):
        solution_dim = len(x0)
        super().__init__(archive, solution_dim=solution_dim, bounds=bounds)
        self.rng = np.random.default_rng(seed)
        self.batch_size = batch_size

        self.num_objects = num_objects
        self.initial_population = initial_population
        self.mutation_k = mutation_k
        assert solution_dim >= self.mutation_k

        self.sols_emitted = 0

    def ask(self):
        if self.sols_emitted < self.initial_population:
            self.sols_emitted += self.batch_size
            return self.rng.integers(self.num_objects,
                                     size=(self.batch_size, self.solution_dim))
        else:
            sols = []
            # select k spots randomly without replacement
            # and calculate the random replacement values
            idx_array = np.tile(np.arange(self.solution_dim),
                                (self.batch_size, 1))
            mutate_idxs = self.rng.permuted(idx_array,
                                            axis=1)[:, :self.mutation_k]
            mutate_vals = self.rng.integers(self.num_objects,
                                            size=(self.batch_size,
                                                  self.mutation_k))

            for i in range(self.batch_size):
                # MOD (had to replace get_random_elite, which no longer exists)
                elite_batch = self.archive.sample_elites(1)
                parent_sol = elite_batch.solution_batch[0]
                sol = copy.deepcopy(parent_sol.astype(int))
                # Replace with random values
                sol[mutate_idxs[i]] = mutate_vals[i]
                sols.append(sol)

            self.sols_emitted += self.batch_size
            return np.array(sols)
