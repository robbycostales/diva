import copy
from typing import Optional

import gin
import numpy as np
import ribs
from ribs.emitters import EmitterBase


@gin.configurable(denylist=["archive", "x0s", "seed"])
class MapElitesCustomEmitter(EmitterBase):
    """Implementation of MAP-Elites which generates solutions corresponding to
    mazes.
    Args:
        archive: Archive to store the solutions.
        x0s: Initial solutions. Only used for solution_dim.
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
        mutation_k: Number of positions in the solution to mutate. Should be
            less than solution_dim.
    """

    def __init__(self,
                 archive: ribs.archives.ArchiveBase,
                 x0s: np.ndarray,
                 use_x0s: bool = True,
                 bounds: Optional['array-like'] = None,  # noqa: F821 # type: ignore
                 seed: int = None,
                 batch_size: int = gin.REQUIRED,
                 initial_population: int = gin.REQUIRED,
                 mutation_k: int = gin.REQUIRED,
                 stepwise_mutations: bool = False):
        solution_dim = len(x0s[0])
        super().__init__(archive, solution_dim=solution_dim, bounds=bounds)
        self.rng = np.random.default_rng(seed)
        self.batch_size = batch_size

        self.initial_population = initial_population
        self.x0s = x0s
        self.use_x0s = use_x0s
        self.mutation_k = mutation_k
        self.stepwise_mutations = stepwise_mutations
        assert solution_dim >= self.mutation_k

        self.sols_emitted = 0
    
    def get_next_x0(self):
        if self.use_x0s:
            # Wrap around if self.sols_emitted > len(self.x0s)
            idx = self.sols_emitted % len(self.x0s)
            self.sols_emitted += 1
            return self.x0s[idx]
        else:
            raise NotImplementedError

    def ask(self):
        if self.sols_emitted < self.initial_population:
            if not self.use_x0s:
                self.sols_emitted += self.batch_size
                return self.rng.integers(
                    low=self.lower_bounds,
                    high=self.upper_bounds + 1,
                    size=(self.batch_size, self.solution_dim))
            else:
                return np.array([self.get_next_x0() for _ in range(self.batch_size)])
        else:
            sols = []
            # select k spots randomly without replacement
            # and calculate the random replacement values
            idx_array = np.tile(np.arange(self.solution_dim),
                                (self.batch_size, 1))
            mutate_idxs = self.rng.permuted(idx_array,
                                            axis=1)[:, :self.mutation_k]

            if self.stepwise_mutations:
                mutate_diffs = self.rng.integers(
                    low=-1,
                    high=+1,
                    endpoint=True,  # To make inclusive
                    size=(self.batch_size, self.mutation_k))
            else:
                mutate_vals = self.rng.integers(
                    low=self.lower_bounds[mutate_idxs],
                    high=self.upper_bounds[mutate_idxs] + 1,
                    size=(self.batch_size, self.mutation_k))

            for i in range(self.batch_size):
                # MOD (had to replace get_random_elite, which no longer exists)
                elite_batch = self.archive.sample_elites(1)
                parent_sol = elite_batch['solution'][0]
                sol = copy.deepcopy(parent_sol.astype(int))
                # Replace with random values

                if self.stepwise_mutations:
                    sol[mutate_idxs[i]] += mutate_diffs[i]
                    sol[mutate_idxs[i]] = np.clip(sol[mutate_idxs[i]],
                                                  self.lower_bounds[mutate_idxs[i]],
                                                  self.upper_bounds[mutate_idxs[i]])
                else:
                    sol[mutate_idxs[i]] = mutate_vals[i]
                sols.append(sol)

            self.sols_emitted += self.batch_size
            return np.array(sols)
