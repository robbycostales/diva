import itertools
from collections import defaultdict
from multiprocessing import Pool
from typing import Any, Callable

import gym
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from scipy import stats
from scipy.stats import norm


def _init_worker(env_id, env_kwargs):
    global env  # Declare a global variable to hold the environment for each worker
    env = gym.make(env_id, **env_kwargs)

def _worker(idx, seed, measures_to_collect, use_gen):
    level = env.genotype_from_seed(idx + seed * 10000) if use_gen else idx + seed * 10000
    env.reset_task(task=level)
    env.reset()
    
    genotype = env.genotype
    measures = env.compute_measures(genotype=genotype, measures=measures_to_collect)
    
    return measures, genotype


class MeasureInfo:
    def __init__(self, 
                 full_range: tuple[float, float], 
                 sample_range: tuple[float, float], 
                 normal_params: tuple[float, float],
                 max_cells: int, 
                 sample_dist: Any,
                 fn: callable, 
                 args: tuple[str]):
        self.full_range = full_range
        self.sample_range = sample_range
        self.normal_params = normal_params
        self.max_cells = max_cells
        self.sample_dist = sample_dist
        self.fn = fn
        self.args = args

    def __str__(self):
        return f'Full range: {self.full_range}, Sample range: {self.sample_range}, Normal params: {self.normal_params}, Max cells: {self.max_cells}, Sample dist: {self.sample_dist}, Fn: {self.fn}, Args: {self.args}'

    @property
    def __dict__(self):
        return {
            'full_range': self.full_range,
            'sample_range': self.sample_range,
            'normal_params': self.normal_params,
            'max_cells': self.max_cells,
            'sample_dist': self.sample_dist,
            # 'fn': self.fn,  # Not JSON serializable
            'args': self.args
        }

class MeasureSelector:
    def __init__(self, 
                 env_id: str,
                 neg_env_id: str = None,
                 og_measures: list[str,] = None,
                 seed: int = None,
                 neg_env_scheduler_fn: Callable = None,
                 process_genotype_fn: Callable = None,
                 is_valid_genotype_fn: Callable = None,
                 standardize_genotype_fn: Callable = None):
        """
        Args:
            env_ids (str): Environment ID to use to compute transforms etc.
        """
        self.env_id = env_id
        self.neg_env_id = neg_env_id
        self.transform = None
        self.pca = None
        self.rng = np.random.default_rng(seed)
        self.og_measures = og_measures
        self.neg_env_scheduler_fn = neg_env_scheduler_fn
        self.process_genotype_fn = process_genotype_fn
        self.is_valid_genotype_fn = is_valid_genotype_fn
        self.standardize_genotype_fn = standardize_genotype_fn

        self.dim_red_method = None
    
    @staticmethod
    def measure_dict_to_array(measure_samples: list[dict,]):
        """
        Converts a dictionary of measure samples to a numpy array.
        
        Args:
            measure_samples (dict): List of measure samples.
        
        Returns:
            measure_array (np.ndarray): Numpy array of measure samples.
        """
        measure_array = np.array(list(measure_samples.values())).T
        return measure_array

    @staticmethod
    def get_measure_samples(env_ids: list[str,],
                            num_samples: int = 1000,
                            seed=None,
                            return_genotypes=False, 
                            env_kwargs_list=None):
        if env_kwargs_list is None:
            env_kwargs_list = [{} for _ in range(len(env_ids))]
        return [MeasureSelector._sample_measures(
                    env_id, 
                    num_samples, 
                    seed=seed, 
                    return_genotypes=return_genotypes, 
                    env_kwargs=env_kwargs_list[i])
                for i, env_id in enumerate(env_ids)]

    @staticmethod
    def _sample_measures(env_id: str, 
                        num_samples: int = 1000, 
                        measures_to_collect: list[str] = None,
                        seed=None,
                        return_genotypes=False, 
                        env_kwargs=None):
        """
        Samples measures from the given environment in separate processes.

        Args:
            env_id (str): Environment ID to use to compute measures.
            num_samples (int): Number of samples to collect from the environment.
            measures_to_collect (list[str]): List of measures to collect. If None, all
                measures will be collected.
        """
        if seed is None:    
            seed = np.random.randint(99)
        if env_kwargs is None:
            env_kwargs = {}

        use_gen = 'QD' in env_id  # Environment type identifier

        # Define arguments for each sample
        args = [(i, seed, measures_to_collect, use_gen) for i in range(num_samples)]
        
        # Create a pool of processes, initializing each with its own environment
        with Pool(processes=10, initializer=_init_worker, initargs=(env_id, env_kwargs)) as pool:
            results = pool.starmap(_worker, args)

        # Collect results
        measure_values = defaultdict(list)
        genotypes = []
        for measures, genotype in results:
            for k, v in measures.items():
                measure_values[k].append(v)
            genotypes.append(genotype)

        if return_genotypes:
            return dict(measure_values), genotypes
        else:
            return dict(measure_values)

        
    @staticmethod
    def are_like_integers(data, tolerance=1e-8):
        rounded_data = np.round(data)
        return np.all(np.isclose(data, rounded_data, atol=tolerance))
        
    @staticmethod
    def compute_measures_info(dr_env_id: str, 
                              ds_env_id: str,
                              measures: list[str,],
                              num_samples: int,
                              base_measures_info: dict[str, MeasureInfo],
                              dr_perc: float,
                              seed=None,
                              env_kwargs=None):
        """ Compute measures info from samples. 
        
        Args:
            ds_env_id (str): Downstream environment ID.
            dr_env_id (str): Domain randomization environment ID.
            measures (list[str,]): List of measures to collect. If None, all
                measures will be collected.
            num_samples (int): Number of samples to collect from the environment.
            base_measures_info (dict[str, MeasureInfo]): Base measures info to compare against.
            dr_perc (float): Percentage of DR data that should fall within the full range.
            seed (int): Seed to use for sampling.
        
        """
        dr_measures = MeasureSelector._sample_measures(  # We assume these are cheap, hence 1000 samples
            dr_env_id, 1000, measures_to_collect=measures, seed=seed, env_kwargs=env_kwargs)

        ds_measures = MeasureSelector._sample_measures(
            ds_env_id, num_samples, measures_to_collect=measures, seed=seed, env_kwargs=env_kwargs)

        measures_infos = dict()
        measures_info_debug = dict()

        for measure in measures:
            ds_measure_values = np.array(ds_measures[measure])
            dr_measure_values = np.array(dr_measures[measure])
            # Perform a test on the measure samples to see if the 
            # distribution is uniform or normal

            if MeasureSelector.are_like_integers(ds_measure_values):
                # Discrete measure; we'll test discrete normal and discrete uniform
                data = np.round(ds_measure_values).astype(int)
                min_val, max_val = np.min(data), np.max(data)
                observed_counts = np.bincount(data - min_val)  # Shift indices for bincount
                # print('observed_counts:', observed_counts)

                # (1) Uniform
                expected_uniform_counts = np.full(
                    shape=observed_counts.shape, 
                    fill_value=len(data) / len(observed_counts))
                # print('expected_uniform_counts:', expected_uniform_counts)
                # Chi-squared test for uniform distribution
                _, p_value_uniform = stats.chisquare(
                    f_obs=observed_counts, f_exp=expected_uniform_counts)
                # Ranges for archive bounds
                ranges_uniform = (min_val-0.5, max_val+0.5)
                num_cells_uniform = max_val - min_val + 1
    
                # (2) Normal
                # Expected frequencies for a discretized normal distribution
                mean, std = np.mean(data), np.std(data)
                norm_prob = stats.norm.cdf(np.arange(min_val, max_val+1), loc=mean, scale=std)
                expected_normal_counts = len(data) * np.diff(np.append(norm_prob, 1))  # Probabilities for each integer
                expected_normal_counts *= observed_counts.sum() / expected_normal_counts.sum()
                # print('expected_normal_counts:', expected_normal_counts)
                # Chi-squared test for normal distribution
                _, p_value_normal = stats.chisquare(f_obs=observed_counts, f_exp=expected_normal_counts)
                # Ranges for archive bounds
                ranges_normal = (min_val-0.5, max_val+0.5)
                num_cells_normal = max_val - min_val + 1
                normal_params = (mean, std)
            else: 
                # Continuous measure; we'll test continuous normal and continuous uniform
                
                # (1) Uniform
                try:
                    # Need to normalize the data to [0, 1] for this test
                    data = ds_measure_values
                    data_normalized = (data - np.min(data)) / (np.max(data) - np.min(data))
                    _, p_value_uniform = stats.kstest(data_normalized, 'uniform')
                except FloatingPointError:  # E.g. underflow if highly unlikely
                    p_value_uniform = 0
                # Ranges for archive bounds
                ranges_uniform = (np.min(data), np.max(data))
                
                # (2) Normal
                try:
                    # Need to standardize the data to ~N(0, 1) for this test
                    data = ds_measure_values
                    data_standardized = (data - np.mean(data)) / np.std(data)
                    _, p_value_normal = stats.kstest(data_standardized, 'norm')
                except FloatingPointError:  # E.g. underflow if highly unlikely
                    p_value_normal = 0
                # Ranges for archive bounds
                ds_mu, ds_std = norm.fit(ds_measure_values)
                ranges_normal = norm.interval(0.99, loc=ds_mu, scale=ds_std)
                normal_params = (ds_mu, ds_std)

            # Determine which distribution it likely belongs to
            if p_value_uniform > p_value_normal:
                likely_distribution = "uniform"
                normal_params = None
                ds_lower_bound, ds_upper_bound = ranges_uniform
                sample_range = ranges_uniform
            else:
                likely_distribution = "normal"
                normal_params = normal_params
                ds_lower_bound, ds_upper_bound = ranges_normal
                sample_range = ranges_normal

            # We want to ensure at least <percentage> of the DR data falls within the full range
            dr_lower_bound = np.percentile(dr_measure_values, 100-dr_perc*100)  # for dr_perc=0.8, this is the 20th percentile
            dr_upper_bound = np.percentile(dr_measure_values, dr_perc*100)  # for dr_perc=0.8, this is the 80th percentile

            full_range = (min(ds_lower_bound, dr_lower_bound), max(ds_upper_bound, dr_upper_bound))
        
            mi = MeasureInfo(
                full_range=full_range,
                sample_range=sample_range,
                normal_params=normal_params,
                max_cells=base_measures_info[measure].max_cells,
                sample_dist=likely_distribution,
                fn=base_measures_info[measure].fn,
                args=base_measures_info[measure].args)

            measures_infos[measure] = mi

        
        return measures_infos
    
    @staticmethod
    def create_measure_histograms(
        env_ids: list[str,], 
        num_samples: int = 1000,
        names: list[str,] = None,
        dims: list[int, int] = (2, 6),
        figsize: tuple[int, int] = (22, 7),
        env_kwargs_list: list[dict,] = None, 
        return_data=False,
        data = None,
        filter_outliers=True,
        only_plot_first_bounds=False):
        """
        Creates histograms of the measures for the given environments.
        
        Args:
            env_ids (list[str,]): List of environment IDs to use to compute measures.
            num_samples (int): Number of samples to collect from each environment.
        
        Returns:
            fig (matplotlib.figure.Figure): Figure containing the histograms.
        """
        if names is None:
            names = env_ids

        if data is not None:
            all_measure_values = data
        else:
            # Sample from the environment to collect the measures
            all_measure_values = MeasureSelector.get_measure_samples(
                env_ids, num_samples, env_kwargs_list=env_kwargs_list)

        # Create a grid of subplots
        fig, axes = plt.subplots(*dims, figsize=figsize)
        axes = axes.flatten()  # Flatten the grid into a vector, for easier iteration

        colors =       ['cyan', 'magenta', 'yellow',  'green', 'red', 'yellow']
        bound_colors = ['blue', 'red',     'orange',  'green', 'red', 'orange']  # Slightly darker versions
        proxy_artists = []

        # Iterate through the keys and values, and plot the histograms
        for ax, key in zip(axes, sorted(list(all_measure_values[0].keys()))):
            for idx, measure_values in enumerate(all_measure_values):
                values = np.array(measure_values[key])
                values = values[np.isfinite(values)]  # Filter out NaN and infinite values

                # Fit a Gaussian distribution to the data
                mu, std = norm.fit(values)
                print('measure', key)
                print('mean', mu, 'std', std)
                
                # Calculate the 99% confidence bound
                try:
                    confidence_bound = norm.interval(0.99, loc=mu, scale=std)
                    lower_bound, upper_bound = confidence_bound
                    if filter_outliers:
                        values = values[(values > lower_bound) & (values < upper_bound)]
                except:
                    lower_bound, upper_bound = mu, mu

                ax.hist(values, edgecolor='black', color=colors[idx], alpha=0.5)  # Plot histogram
                ax.set_title(key)  # Use the key as the title for each subplot
                
                # Plot vertical lines at the ends of the confidence bound
                if not(only_plot_first_bounds and idx > 0):
                    ax.axvline(lower_bound, color=bound_colors[idx], linestyle='--')
                    ax.axvline(upper_bound, color=bound_colors[idx], linestyle='--')
                    ax.axvline(mu, color=bound_colors[idx], linestyle='--')
                    ax.text(mu, ax.get_ylim()[1]*0.7, f'N({mu:.2e},{std:.2e})', color=bound_colors[idx])
                    # Label each line with text indicating the confidence bound
                    ax.text(lower_bound, ax.get_ylim()[1]*0.8, f'{lower_bound:.2e}', color=bound_colors[idx])
                    ax.text(upper_bound, ax.get_ylim()[1]*0.9, f'{upper_bound:.2e}', color=bound_colors[idx])

                # Create proxy artists for legend
                proxy_artists.append(Patch(color=colors[idx], label=names[idx]))

        # Add common labels
        fig.text(0.5, 0.051, 'Value', ha='center')
        fig.text(0.09, 0.5, 'Frequency', va='center', rotation='vertical')

        # Add some space between
        fig.subplots_adjust(hspace=0.7, wspace=0.7)

        # Add legend for the environments
        fig.legend(handles=proxy_artists[:len(names)], loc='lower right')
        
        # Return figure to plot
        if return_data:
            return fig, axes, all_measure_values
        else:
            return fig, axes

    @staticmethod
    def plot_covariance_matrices(
            env_ids: list[str,], 
            num_samples: int = 1000, 
            names: list[str,] = None,
            env_kwargs_list: list[dict,] = None,
            return_data=False,
            measures_to_exclude: list[str,] = None,
            data = None):
        """
        Computes and plots covariance matrices of the measures for the given environments.

        Args:
            env_ids (list[str,]): List of environment IDs to use to compute measures.
            num_samples (int): Number of samples to collect from each environment.

        Returns:
            fig (matplotlib.figure.Figure): Figure containing the covariance matrix plots.
        """
        if names is None:
            names = env_ids

        if data is not None:
            all_measure_values = data
        else:
            # Sample from the environment to collect the measures
            all_measure_values = MeasureSelector.get_measure_samples(
                env_ids, num_samples, env_kwargs_list=env_kwargs_list)

        # Initialize the plot
        fig, axes = plt.subplots(1, len(env_ids), figsize=(5 * len(env_ids), 5))
        if len(env_ids) == 1:  # Ensure axes is always iterable
            axes = [axes]

        # Define the custom colormap
        colors = ["red", "white", "green"]  # Red for negative, white for zero, green for positive
        cmap = LinearSegmentedColormap.from_list("rg", colors, N=256)

        normalized_data_list = []  # Store normalized data for each environment
        global_max = 0

        # Calculate the normalized covariance matrix for each environment and find global maximum
        for idx, (env_id, name, ax) in enumerate(zip(env_ids, names, axes)):
            measure_values = all_measure_values[idx]
            # Create a list of numpy arrays for each measure
            data = [np.array(measure_values[key])[np.isfinite(np.array(measure_values[key]))] for key in measure_values]
            # Normalize each measure
            normalized_data = [(d - np.mean(d)) / (np.std(d) if np.std(d) > 0 else 1) for d in data]
            normalized_data_list.append(normalized_data)
            # Compute covariance matrix
            matrix = np.cov(normalized_data)
            # Find the maximum absolute value in the matrix to use for color scaling
            max_val = np.max(np.abs(matrix))
            global_max = max(global_max, max_val)

        # Plot each matrix with the same color scaling
        for idx, (env_id, name, ax, normalized_data) in enumerate(zip(env_ids, names, axes, normalized_data_list)):
            matrix = np.cov(normalized_data)
            # Mask lower triangle
            mask = np.triu(np.ones_like(matrix, dtype=bool))
            matrix = np.ma.masked_where(~mask, matrix)
            # Plotting the covariance matrix with the custom colormap
            cax = ax.matshow(matrix, interpolation='nearest', cmap=cmap, vmin=-global_max, vmax=global_max)
            fig.colorbar(cax, ax=ax)
            ax.set_title(f"Covariance: {name}")
            # Process the keys to create abbreviated labels
            measure_values = all_measure_values[idx]
            abbreviated_labels = [''.join(word[0] for word in key.split('_')) for key in measure_values.keys()]
            # Set tick labels with abbreviated names
            ax.set_xticks(range(len(measure_values.keys())))
            ax.set_yticks(range(len(measure_values.keys())))
            ax.set_xticklabels(abbreviated_labels, fontsize=10)
            ax.set_yticklabels(abbreviated_labels, fontsize=10)

        # Print out combinations of measures with low covariance
        measure_indices = list(range(12))  # Indices for the measures
        combinations_with_covariances = []
        # Check combinations of 3 and 4 measures
        measure_names = list(measure_values.keys())
        matrix = np.cov(normalized_data_list[0])  # Downstream env
        # Exclude some
        # Measures to exclude (by name)
        if measures_to_exclude is None:
            measures_to_exclude = []

        # Get indices of measures to exclude
        excluded_indices = [measure_names.index(name) for name in measures_to_exclude]
        # Get indices of measures to consider
        measure_indices = [i for i in range(len(measure_names)) if i not in excluded_indices]

        # Check combinations of 3 and 4 measures
        for r in [3, 4]:
            for combination in itertools.combinations(measure_indices, r):
                total_covariance = 0
                for i, j in itertools.combinations(combination, 2):
                    total_covariance += abs(matrix[i, j])
                
                combinations_with_covariances.append((combination, total_covariance))
        # Sort combinations by total sum of absolute covariances
        combinations_with_covariances.sort(key=lambda x: x[1])
        # Get the top 20 combinations
        top_combinations = combinations_with_covariances[:20]
        # Print the top combinations with measure names
        for idx, (combination, covariance) in enumerate(top_combinations):
            combination_names = [measure_names[i] for i in combination]
            print(f"Rank {idx+1}: Combination {combination_names} with total covariance {covariance}")

        plt.tight_layout()
        # Return figure to plot
        if return_data:
            return fig, all_measure_values
        else:
            return fig

