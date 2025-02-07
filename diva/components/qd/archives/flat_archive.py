"""Custom (flat) GridArchive."""
from typing import Any

import gin
import numpy as np
import ribs.archives
from ribs.archives._archive_stats import ArchiveStats


@gin.configurable
class FlatArchive(ribs.archives.ArchiveBase):
    """ Flat version of GridArchive. """

    def __init__(self,
                 solution_dim,
                 cells: int,
                 seed: int = None,
                 dtype: Any = np.float64,
                 qd_score_offset: float = 0.0,
                 threshold_min: float = -0.00001):
        ribs.archives.ArchiveBase.__init__(
            self,
            solution_dim=solution_dim,
            cells=cells,
            measure_dim=1,  # Placeholder value
            learning_rate=1.0,
            seed=seed,
            dtype=dtype,
            qd_score_offset=qd_score_offset)
        self._threshold_min = threshold_min  
        
        # Ribs does not have sample_weights functionality
        self._sample_weights = np.zeros(self._cells, dtype=self.dtype)

        # Base archive does not define this
        self._occupied_indices = np.array(list(range(self._cells))).astype(np.int32)
    
        # Keep track of solution set to ensure no duplicates
        self._solution_set = set()

        # Handling of stale solutions
        self._steps_since_last_updated = dict()
        self._stale_solutions_to_update = None

    @property
    def occupied_indices(self):
        return self._store.occupied_list[:len(self._store)]
    
    @property
    def unoccupied_indices(self):
        # Create an array of all possible indices from 0 to self._cells-1
        all_indices = np.arange(self._cells)
        # Use np.setdiff1d to find indices that are in all_indices but not in self._occupied_indices
        # It doesn't require the input arrays to be sorted but will sort them internally for efficiency
        return np.setdiff1d(all_indices, self._occupied_indices[:self._store._props['n_occupied']], assume_unique=True)

    def get_matrices(self):
        """ Returns a matrix of the objective values (and occupied) for all cells. """
        indices = self.occupied_indices
        
        # Assert that there are no inf values in indices
        assert not np.any(np.isinf(indices))
        
        matrices = {
            'objective': self._store._fields['objective'][indices],
            'occupied_indices': indices,
            'solutions': self._store._fields['solution'][indices],         
        }
        
        return matrices

    def sample_elites(self, n):
        """ Samples elites from the archive, supporting nonuniform sampling.

        Args:
            n (int): Number of elites to sample.
        Returns:
            elites (dict): A batch of elites randomly selected from the archive.
        Raises:
            IndexError: The archive is empty.
        """
        if self.empty:
            raise IndexError("No elements in archive.")
        if len(np.nonzero(self._sample_weights)) <= 1:
            # Sample uniformly.
            random_indices = self._rng.integers(self._store._props['n_occupied'], size=n)
            selected_indices = self._occupied_indices[random_indices]
        else:
            # Use self._sample_weights to sample:
            sample_weights = self._sample_weights / np.sum(self._sample_weights)
            selected_indices = self._rng.choice(
                list(range(len(self._store._fields['solution']))), size=n, p=sample_weights)

        _, elites = self._store.retrieve(selected_indices)
        return elites
    
    def update_sample_weights(self, seed_weights, seed2index, weight_sum_to_check):
        """ Updates the sample weights for the archive.

        Args:
            seed_weights (dict): A dictionary mapping seeds to their weights.
            seed2index (dict): A dictionary mapping seeds to their indices in
                the archive.
        """
        # Reset the sample weights.
        self._sample_weights = np.zeros(self._cells, dtype=self.dtype)
        # Update the sample weights.
        for seed, weight in seed_weights.items():
            self._sample_weights[seed2index[seed]] = weight

        if not np.isclose(np.sum(self._sample_weights), weight_sum_to_check):
            print('WARNING: Sample weights sums not close: ', np.sum(self._sample_weights), weight_sum_to_check)

    def retrieve(self, measures_batch):
        del measures_batch
        raise NotImplementedError(
            "FlatArchive does not support retrieval.")

    def retrieve_single(self, measures):
        del measures
        raise NotImplementedError(
            "FlatArchive does not support retrieval.")
    
    def index_of(self, measures_batch):
        del measures_batch
        raise NotImplementedError(
            "FlatArchive does not support indexing by measures.")
    
    def index_of_single(self, measures):
        del measures
        raise NotImplementedError(
            "FlatArchive does not support indexing by measures.")
    
    def add(self,
            solution,
            objective,
            measures):  # User should input dummy values for consistency
        """ Inserts a batch of solutions into the archive. 
        
        We keep the archive sorted by objective value, so we need to merge the
        new solutions with the old solutions. We do this by creating a temporary
        array to store the merged solutions, and then taking the top `self._cells`
        solutions from the merged list.

        NOTE: Archive sorted in descending order of objective value.
        """
        # Sort batch in descending order of objective value.
        sort_indices = np.argsort(-objective)
        solution = solution[sort_indices]
        objective = objective[sort_indices]
        if measures is not None:
            measures = measures[sort_indices]
        
        ## Step 0: Preprocess input. ##
        solution = np.asarray(solution)
        objective = np.asarray(objective)
        measures = np.asarray(measures)
        batch_size = solution.shape[0]

        # Create temporary arrays to store merged solutions
        temp_size = self._cells + batch_size
        merged_solution = np.empty((temp_size, self._solution_dim), dtype=self.dtype)
        merged_objective = np.empty(temp_size, dtype=self.dtype)
        merged_measures = np.empty((temp_size, self._measure_dim), dtype=self.dtype)

        # Initialize value_batch and status_batch
        value_batch = np.zeros(batch_size, dtype=self.dtype)
        status_batch = np.zeros(batch_size, dtype=np.int32)
        is_new = np.zeros(batch_size, dtype=bool)
        improve_existing = np.zeros(batch_size, dtype=bool)
        
        i, j, k = 0, 0, 0  # indices for i=old solutions, j=new solutions, and k=merged list, respectively
        while i < self._store._props['n_occupied'] and j < len(solution):
            if self._store._fields['objective'][i] > objective[j]:
                merged_solution[k] = self._store._fields['solution'][i]
                merged_objective[k] = self._store._fields['objective'][i]
                merged_measures[k] = self._store._fields['measures'][i]
                i += 1
                k += 1
            else:
                if (tuple(solution[j].tolist()) not in self._solution_set
                    and objective[j] > self._threshold_min):
                    merged_solution[k] = solution[j]
                    merged_objective[k] = objective[j]
                    if measures is not None:
                        merged_measures[k] = measures[j]
                    if k < self._cells:
                        # Track improvement
                        value_batch[j] = objective[j] - self._store._fields['objective'][i]
                        improve_existing[j] = True
                        self._solution_set.add(tuple(solution[j].tolist()))
                        self._steps_since_last_updated[tuple(solution[j].tolist())] = 0
                    k += 1
                j += 1
            

        # If there are leftover solutions in old solutions
        while i < self._store._props['n_occupied']:
            merged_solution[k] = self._store._fields['solution'][i]
            merged_objective[k] = self._store._fields['objective'][i]
            merged_measures[k] = self._store._fields['measures'][i]
            if k >= self._cells:
                self._solution_set.remove(tuple(self._store._fields['solution'][i].tolist()))
                self._steps_since_last_updated.pop(tuple(self._store._fields['solution'][i].tolist()))
            i += 1
            k += 1

        # If there are leftover solutions in new solutions
        while j < len(objective):
            if (tuple(solution[j].tolist()) not in self._solution_set
                and objective[j] > self._threshold_min):
                merged_solution[k] = solution[j]
                merged_objective[k] = objective[j]
                if measures is not None:
                    merged_measures[k] = measures[j]
                if k < self._cells:
                    is_new[j] = True
                    value_batch[j] = objective[j] - self._threshold_min
                    self._solution_set.add(tuple(solution[j].tolist()))
                    self._steps_since_last_updated[tuple(solution[j].tolist())] = 0
                k += 1
            j += 1

        self._store._props['n_occupied'] = min(self._cells, k)
        
        # Update status_batch
        status_batch[is_new] = 2
        status_batch[improve_existing] = 1

        # Now, we simply take the top `self._cells` solutions from the merged list
        self._store._fields['solution'] = merged_solution[:self._cells]
        self._store._fields['objective'] = merged_objective[:self._cells]
        self._store._fields['measures'] = merged_measures[:self._cells]
        # Get num non-empty cells in self._store._fields['objective']
        self._store._props['occupied'].fill(False)
        self._store._props['occupied'][:self._store._props['n_occupied']] = True

        # Get indices where occupied is True
        self._store._props['occupied_list'][:self._store._props['n_occupied']] = (
            np.where(self._store._props['occupied'])[0])

        # Return early if we cannot insert any solutions
        can_insert = is_new | improve_existing
        if not np.any(can_insert):
            add_info = {'value': value_batch, 'status': status_batch}
            return add_info

        # Update the thresholds
        pass  # Nothing to do

        ## Update archive stats. ##
        self._objective_sum = np.sum(self._store._fields['objective'][:self._store._props['n_occupied']])
        new_qd_score = (self._objective_sum - self.dtype(len(self))
                        * self._qd_score_offset)
        # Get first nonzero element of status_batch
        max_obj_insert = objective[0]

        if self._stats.obj_max is None or max_obj_insert > self._stats.obj_max:
            new_obj_max = max_obj_insert
        else:
            new_obj_max = self._stats.obj_max
        
        norm_qd_score = self.dtype(new_qd_score / self.cells),

        self._stats = ArchiveStats(
            num_elites=len(self),
            coverage=self.dtype(len(self) / self.cells),
            qd_score=new_qd_score,
            norm_qd_score=norm_qd_score,
            obj_max=new_obj_max,
            obj_mean=self._objective_sum / self.dtype(len(self)),
        )

        # Swap indices to match the original order
        status_batch[sort_indices] = status_batch
        value_batch[sort_indices] = value_batch

        add_info = {
            'status': status_batch,
            'value': value_batch,
        }

        return add_info
    
    def add_single(self, solution, objective, measures):
        """ Inserts a single solution into the archive. """
        solution = np.asarray([solution]).astype(self.dtype)
        objective = np.asarray([objective]).astype(self.dtype)
        measures = np.asarray([measures]).astype(self.dtype)
        
        add_info = self.add(solution, objective, measures)
        
        return add_info
    
    def reset_sslu(self):
        """ Resets the steps since last updated for all cells. """
        for k in self._steps_since_last_updated.keys():
            self._steps_since_last_updated[k] = -1
    
    def increment_sslu(self, val=1): 
        """ Increments the steps since last updated for all cells. """
        for k in self._steps_since_last_updated.keys():
            self._steps_since_last_updated[k] += val
    
    def get_max_sslu(self):
        if len(self._steps_since_last_updated) == 0:
            return 0
        return np.max(list(self._steps_since_last_updated.values()))
    
    def get_stale_solutions(self, sslu_threshold=100):
        """ Return solutions with sslu > sslu_threshold. """
        # First, remove all values in self._steps_since_last_updated that are no longer in the archive
        for k in list(self._steps_since_last_updated.keys()):
            # Use numpy to check if k is in self._store._fields['solution']
            karr = np.array(k)
            if len(karr.shape) == 1:
                karr = karr.reshape(1, -1)
            if not np.any((self._store._fields['solution'] == karr).all(axis=1)):
                self._steps_since_last_updated.pop(k)

        solutions = []
        for k, v in self._steps_since_last_updated.items():
            if v > sslu_threshold or v == -1:
                solutions.append(np.array(k))
        self._stale_solutions_to_update = np.array(solutions).astype(self._store._fields['solution'].dtype)
        return self._stale_solutions_to_update
    
    def update_stale_solutions(self, objective_batch):
        """ Updates the stale solutions with the given objective_batch. """
        objective_batch = objective_batch[:len(self._stale_solutions_to_update)]

        for k in self._steps_since_last_updated.keys():
            self._steps_since_last_updated[k] = 0
        
        # Identify the indices of stale solutions in your archive.
        stale_indices = [np.where((self._store._fields['solution'] == k).all(axis=1))[0][0] 
                         for k in self._stale_solutions_to_update]

        # Update the values at these indices with new objectives.
        for idx, new_objective in zip(stale_indices, objective_batch):
            self._store._fields['objective'][idx] = new_objective
        
        # Be sure to preserve order of the archive.
        self.sort_by_objective()
        self._stale_solutions_to_update = None

    def sort_by_objective(self):
        """ Sorts the archive by objective value in descending order. """
        # Get indices that would sort the objective array in descending order
        sorted_indices = np.argsort(self._store._fields['objective'][:self._store._props['n_occupied']])[::-1]
        
        # Use these indices to rearrange the objective array
        self._store._fields['objective'] = np.array(self._store._fields['objective'])[sorted_indices]
        self._store._fields['solution'] = np.array(self._store._fields['solution'])[sorted_indices]
        self._store._fields['measures'] = np.array(self._store._fields['measures'])[sorted_indices]

        # If self._steps_since_last_updated is a dictionary, you would update it like:
        keys = list(self._steps_since_last_updated.keys())
        values = list(self._steps_since_last_updated.values())

        keys_sorted = [keys[i] for i in sorted_indices]
        values_sorted = [values[i] for i in sorted_indices]

        self._steps_since_last_updated = dict(zip(keys_sorted, values_sorted))

    def to_grid_archive(self, grid_archive_fn, compute_measures_fn, measures, measure_selector=None, gt_type=None):
        """ Converts the archive to a GridArchive. """
        grid_archive = grid_archive_fn()
        new_measures = []
        for s in self._store._fields['solution'][:self._store._props['n_occupied']]:
            meas = compute_measures_fn(s, measures, gt_type=gt_type)
            new_measures.append([meas[k] for k in measures])
        if measure_selector is not None:
            new_measures, _ = measure_selector.transform_data(new_measures)
        # Add solutions
        grid_archive.add(self._store._fields['solution'][:self._store._props['n_occupied']],
                         self._store._fields['objective'][:self._store._props['n_occupied']], 
                         new_measures)
        return grid_archive