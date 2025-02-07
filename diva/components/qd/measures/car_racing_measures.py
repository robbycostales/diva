import numpy as np

from diva.components.qd.measures.measure_selection import MeasureInfo


class CarRacingMeasures:

    @staticmethod
    def derivatives_of_bezier_full_curve(curve):
        """ Calculate the first and second derivatives of a Bezier curve at each point using finite differences.
        curve_points should be an array of shape (num_points, 2), representing the curve's points.
        """
        # Ensure curve_points is a numpy array
        curve = np.asarray(curve)
        # First derivative (finite difference)
        # Central differences for interior points
        B_prime = np.zeros_like(curve)
        B_prime[1:-1] = (curve[2:] - curve[:-2]) / 2
        # Forward difference for the first point
        B_prime[0] = (curve[1] - curve[0])
        # Backward difference for the last point
        B_prime[-1] = (curve[-1] - curve[-2])
        # Second derivative (finite difference)
        B_double_prime = np.zeros_like(curve)
        # Central difference for the second derivative
        B_double_prime[1:-1] = (curve[2:] - 2 * curve[1:-1] + curve[:-2])
        # Approximate endpoints assuming zero curvature change beyond the boundary
        B_double_prime[0] = (curve[2] - 2 * curve[1] + curve[0])
        B_double_prime[-1] = (curve[-1] - 2 * curve[-2] + curve[-3])
        return B_prime, B_double_prime
    
    @staticmethod
    def compute_curvatures(curve, use_abs=False):
        """ Calculate the curvature at the midpoint of each Bezier curve segment in points_array.
        points_array.shape should be (num_segments, 4, 2)
        """
        B_prime, B_double_prime = CarRacingMeasures.derivatives_of_bezier_full_curve(curve)
        curvature = B_prime[:, 0] * B_double_prime[:, 1] - B_prime[:, 1] * B_double_prime[:, 0]
        if use_abs:
            curvature = np.abs(curvature)
        curvature /= (1e-8 + (B_prime[:, 0]**2 + B_prime[:, 1]**2)**1.5)
        return curvature

    @staticmethod
    def curve_angle_changes(curve, min_angle_change=np.radians(2), max_angle_change=np.radians(45), do_abs=True):  # Default threshold of 5 degrees
        """
        Calculate the angle change along each point in the curve using derivatives, filtering out changes below a threshold.
        
        Parameters:
            curve (np.array): Array of shape (num_points, 2), representing the curve's points.
            angle_threshold (float): Threshold in radians to filter minor angle changes.
        
        Returns:
            np.array: Angle changes along the curve that exceed the threshold.
        """
        B_prime, B_double_prime = CarRacingMeasures.derivatives_of_bezier_full_curve(curve)

        # Calculate angles using arctan2 for each derivative vector
        angles = np.arctan2(B_prime[:, 1], B_prime[:, 0])

        # Calculate the absolute differences in angles between consecutive points
        if do_abs:
            angle_diffs = np.abs(np.diff(angles))
        else:
            angle_diffs = np.diff(angles)

        # Filter out angle changes that are below the threshold
        significant_angle_changes = angle_diffs[angle_diffs > min_angle_change]
        significant_angle_changes = significant_angle_changes[significant_angle_changes < max_angle_change]
        return significant_angle_changes

    ###########################################################################
    #                                MEASURES                                 #  
    ###########################################################################

    @staticmethod
    def i_area_to_length_ratio():
        return MeasureInfo(
            full_range      = (1.48e+1, 9.00e+1),
            sample_range    = (1.48e+1, 6.40e+1),
            normal_params   = (3.94e+1, 9.56e+0),
            max_cells       = np.inf,
            sample_dist     = 'normal',
            fn              = CarRacingMeasures.m_area_to_length_ratio,
            args            = ('curve',)
        )

    @staticmethod
    def m_area_to_length_ratio(curve):
        """ Calculate the ratio of enclosed area to curve length.
        
        Computes the ratio of the area enclosed by the Beziér curve to its 
        total length.
        """
        area = CarRacingMeasures.m_enclosed_area(curve)
        length = CarRacingMeasures.m_curve_length(curve)
        if length == 0:
            return 10e-10
        return area / length
    
    ###########################################################################

    @staticmethod
    def i_average_curvature():
        return MeasureInfo(
            full_range      = (-1.60e-2, +5.57e-5),
            sample_range    = (-1.60e-2, +5.57e-5),
            normal_params   = (-7.98e-3, +3.12e-3),
            max_cells       = np.inf,
            sample_dist     = 'normal',
            fn              = CarRacingMeasures.m_average_curvature,
            args            = ('curve',)
        )
    
    @staticmethod
    def m_average_curvature(curve):
        """ Calculate the average curvature at midpoints of Beziér segments.
        
        Uses the control points of each segment to determine the curvature 
        at the midpoint.
        """
        curvatures = CarRacingMeasures.compute_curvatures(curve, use_abs=False)
        return np.mean(curvatures)    
            
    ###########################################################################

    @staticmethod
    def i_curve_length():
        return MeasureInfo(
            full_range      = (7.92e+2, 1.51e+3),
            sample_range    = (7.92e+2, 1.51e+3),
            normal_params   = (1.15e+3, 1.38e+2), 
            max_cells       = np.inf,
            sample_dist     = 'normal',
            fn              = CarRacingMeasures.m_curve_length,
            args            = ('curve',)
        )

    @staticmethod
    def m_curve_length(curve):
        """ Calculate the total length of the Beziér curve.
        
        Computes the Euclidean distance between successive points on the 
        Beziér curve and sums them up.
        """
        return np.sum(np.sqrt(np.sum(np.diff(curve, axis=0)**2, axis=1)))
    
    ###########################################################################

    @staticmethod
    def i_curve_distances_variance():
        return MeasureInfo(
            full_range      = (7.02e-1, 4.5),
            sample_range    = (7.02e-1, 2.17e+0),
            normal_params   = (1.44e+0, 2.85e-1),
            max_cells       = np.inf,
            sample_dist     = 'normal',
            fn              = CarRacingMeasures.m_curve_distances_variance,
            args            = ('curve',)
        )

    @staticmethod
    def m_curve_distances_variance(curve):
        """ Quantify the variability in distances between successive points.
        """
        distances = np.sqrt(np.sum(np.diff(curve, axis=0)**2, axis=1))
        return np.std(distances)
    
    ###########################################################################

    @staticmethod
    def i_enclosed_area():
        return MeasureInfo(
            full_range      = (1.48e+4, 1.20e+5),
            sample_range    = (1.48e+4, 7.55e+4),
            normal_params   = (4.52e+4, 1.18e+4),
            max_cells       = np.inf,
            sample_dist     = 'normal',
            fn              = CarRacingMeasures.m_enclosed_area,
            args            = ('curve',)
        )

    @staticmethod
    def m_enclosed_area(curve):
        """ Calculate the area enclosed by the Beziér curve.
        
        Uses the shoelace formula to compute the area enclosed by the curve.
        """
        x, y = curve.T
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    
    ###########################################################################

    @staticmethod
    def i_sig_angle_changes():
        return MeasureInfo(
            full_range      = (55, 2.22e+2),  
            sample_range    = (9.36e+1, 2.22e+2),
            normal_params   = (1.58e+2, 2.50e+1),
            max_cells       = np.inf,
            sample_dist     = 'normal',
            fn              = CarRacingMeasures.m_sig_angle_changes,
            args            = ('curve',)
        )
    
    @staticmethod
    def m_sig_angle_changes(curve):
        """ Calculate the total change in angle across the curve, considering only significant changes. """
        significant_angle_changes = CarRacingMeasures.curve_angle_changes(curve)
        return len(significant_angle_changes)
    

    ###########################################################################

    @staticmethod
    def i_total_angle_changes():
        return MeasureInfo(
            full_range      = (5.52e+0, 2.38e+1),  
            sample_range    = (5.52e+0, 2.38e+1),
            normal_params   = (1.47e+1, 3.55e+0),
            max_cells       = np.inf,
            sample_dist     = 'normal',
            fn              = CarRacingMeasures.m_total_angle_changes,
            args            = ('curve',)
        )
    
    @staticmethod
    def m_total_angle_changes(curve):
        """ Calculate the total change in angle across the curve, considering only significant changes. """
        significant_angle_changes = CarRacingMeasures.curve_angle_changes(curve)
        return sum(significant_angle_changes)
                
    ###########################################################################

    @staticmethod
    def i_total_curvature():
        return MeasureInfo(
            full_range      = (2.79e+0, 2.06e+1),
            sample_range    = (4.79e+0, 2.06e+1),
            normal_params   = (1.27e+1, 3.07e+0),
            max_cells       = np.inf,
            sample_dist     = 'normal',
            fn              = CarRacingMeasures.m_total_curvature,
            args            = ('curve',)
        )
    
    @staticmethod
    def m_total_curvature(curve):
        """ Calculate the total curvature over each segment and sum them up. """
        curvatures = CarRacingMeasures.compute_curvatures(curve, use_abs=True)
        return np.sum(curvatures)

    ###########################################################################

    @staticmethod
    def i_com_x():
        return MeasureInfo(
            full_range      = (9.53e+1, 2.32e+2),
            sample_range    = (9.53e+1, 2.32e+2),
            normal_params   = (1.64e+2, 2.65e+1),
            max_cells       = np.inf,
            sample_dist     = 'normal',
            fn              = CarRacingMeasures.m_com_x,
            args            = ('curve',)
        )
    
    @staticmethod
    def m_com_x(curve):
        """ Calculate the center of mass x position over the curve. """
        return np.mean(curve[:, 0])

    ###########################################################################

    @staticmethod
    def i_com_y():
        return MeasureInfo(
            full_range      = (9.72e+1, 2.32e+2),
            sample_range    = (9.72e+1, 2.32e+2),
            normal_params   = (1.65e+2, 2.62e+1),
            max_cells       = np.inf,
            sample_dist     = 'normal',
            fn              = CarRacingMeasures.m_com_y,
            args            = ('curve',)
        )
    
    @staticmethod
    def m_com_y(curve):
        """ Calculate the center of mass x position over the curve. """
        return np.mean(curve[:, 1])
    

    ###########################################################################

    @staticmethod
    def i_var_x():
        return MeasureInfo(
            full_range      = (2.11e+3, 2.55e+4),
            sample_range    = (2.11e+3, 1.42e+4),
            normal_params   = (8.15e+3, 2.34e+3),
            max_cells       = np.inf,
            sample_dist     = 'normal',
            fn              = CarRacingMeasures.m_var_x,
            args            = ('curve',)
        )
    
    @staticmethod
    def m_var_x(curve):
        """ Calculate the center of mass x position over the curve. """
        return np.var(curve[:, 0])

    ###########################################################################

    @staticmethod
    def i_var_y():
        return MeasureInfo(
            full_range      = (2.58e+3, 2.55e+4),
            sample_range    = (2.58e+3, 1.40e+4),
            normal_params   = (8.31e+3, 2.23e+3),
            max_cells       = np.inf,
            sample_dist     = 'normal',
            fn              = CarRacingMeasures.m_var_y,
            args            = ('curve',)
        )
    
    @staticmethod
    def m_var_y(curve):
        """ Calculate the center of mass x position over the curve. """
        return np.var(curve[:, 1])
    
        ###########################################################################

    @staticmethod
    def i_med_x():
        return MeasureInfo(
            full_range      = (5.41e+1, 2.70e+2),
            sample_range    = (5.41e+1, 2.70e+2),
            normal_params   = (1.62e+2, 4.19e+1),
            max_cells       = np.inf,
            sample_dist     = 'normal',
            fn              = CarRacingMeasures.m_med_x,
            args            = ('curve',)
        )
    
    @staticmethod
    def m_med_x(curve):
        """ Calculate the center of mass x position over the curve. """
        return np.median(curve[:, 0])

    ###########################################################################

    @staticmethod
    def i_med_y():
        return MeasureInfo(
            full_range      = (5.72e+1, 2.65e+2),
            sample_range    = (5.72e+1, 2.65e+2),
            normal_params   = (1.61e+2, 4.03e+1),
            max_cells       = np.inf,
            sample_dist     = 'normal',
            fn              = CarRacingMeasures.m_med_y,
            args            = ('curve',)
        )
    
    @staticmethod
    def m_med_y(curve):
        """ Calculate the center of mass x position over the curve. """
        return np.median(curve[:, 1])

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
    def compute_measures(curve=None, 
                         segments=None, 
                         measures=None):
        """ Compute the specified measures for the car racing environment.

        Args:
        - curve (array-like): An array of coordinates representing the racing curve.
        - segments (array-like): Line segments making up the track or path.
        - measures (list): List of measures to compute. If None, all measures are computed.

        Returns:
        A dictionary with the computed measures.
        """
        # If measures not provided, compute all measures
        if measures is None:
            measures = CarRacingMeasures.get_all_measures()

        
        # All args
        all_args = {
            'segments': segments,
            'curve': curve,
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


METHOD_NAMES = [name for name in dir(CarRacingMeasures) if name.startswith("m_") and callable(getattr(CarRacingMeasures, name))]
MEASURES = [name[2:] for name in METHOD_NAMES]

MEASURE_INFO = dict()
for measure in MEASURES:
    info_fn = getattr(CarRacingMeasures, f'i_{measure}', None)
    assert info_fn is not None, f"Missing info function for measure {measure}"
    MEASURE_INFO[measure] = info_fn()


if __name__ == '__main__':
    # Print all measures
    for i, measure in enumerate(MEASURES):
        print(i, measure)