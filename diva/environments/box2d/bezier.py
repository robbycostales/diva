# Copyright (c) Stack Exchange, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the CC BY-SA 3.0 license.

import numpy as np
from scipy.special import binom as binomial_coefficient


def bernstein_polynomials(degree, num_points):
    """ Generate a matrix of Bernstein polynomials for a given degree and number of points. """
    t = np.linspace(0, 1, num_points)
    return np.array([binomial_coefficient(degree, k) * t**k * (1 - t)**(degree - k) for k in range(degree + 1)])


def bezier_curve(points, bernstein_poly):
    """ Compute Bezier curve given control points using a precomputed Bernstein polynomial matrix. """
    # Apply the precomputed Bernstein polynomials to the points
    curve = np.tensordot(bernstein_poly, points, axes=[0, 0])
    return curve
    # return curve.T  # Ensure the curve is transposed to match the expected output shape (num_points, 2)


def get_curve(points, r=0.3, numpoints=100, degree=3, bernstein_poly=None):
    """ Compute the concatenated Bezier curve for given control points including tangents,
        using a precomputed Bernstein polynomial matrix for efficiency. """
    if bernstein_poly is None:
        bernstein_poly = bernstein_polynomials(degree, numpoints)  # Precompute once
    num_segments = points.shape[0] - 1
    segments = []  # This will store the Segment objects
    curves = []  # This will store the curve points

    for i in range(num_segments):
        p1, p2 = points[i, :2], points[i + 1, :2]
        angle1, angle2 = points[i, 2], points[i + 1, 2]
        d = np.linalg.norm(p2 - p1)
        r_scaled = r * d
        p = np.array([
            p1,
            p1 + np.array([r_scaled * np.cos(angle1), r_scaled * np.sin(angle1)]),
            p2 + np.array([r_scaled * np.cos(angle2 + np.pi), r_scaled * np.sin(angle2 + np.pi)]),
            p2
        ])
        curve = bezier_curve(p, bernstein_poly)
        segments.append(p)  # Create and store the segment
        curves.append(curve)  # Store the curve points

    full_curve = np.concatenate(curves)  # Concatenate the curve points
    segments = np.array(segments)

    return segments, full_curve


# class Segment:
#     def __init__(self, p, curve):
#         self.p = p  # Control points including computed intermediate points if needed
#         self.curve = curve

#     def __repr__(self):
#         return f"Segment(Control Points Shape={self.p.shape}, Curve Shape={self.curve.shape})"

def ccw_sort(p):
    d = p-np.mean(p,axis=0)
    s = np.arctan2(d[:,0], d[:,1])
    return p[np.argsort(s),:]


# Optimized version
def get_bezier_curve(a=None, rad=0.2, edgy=0, return_all=False, bernstein_poly=None, **kw):
    if a is None:
        a = get_random_points(**kw)

    numpoints = kw.get('numpoints', 30)

    p = np.arctan(edgy) / np.pi + 0.5
    a = ccw_sort(a)
    a = np.vstack((a, a[0,:]))  # Use vstack for stacking arrays vertically
    d = np.diff(a, axis=0)
    ang = np.arctan2(d[:,1], d[:,0])

    # Vectorized operation for adjusting angles
    ang = np.where(ang >= 0, ang, ang + 2 * np.pi)

    ang1 = ang
    ang2 = np.roll(ang, 1)
    ang = p * ang1 + (1 - p) * ang2 + (np.abs(ang2 - ang1) > np.pi) * np.pi
    ang = np.concatenate((ang, [ang[0]]))

    a = np.hstack((a, ang[:, None]))  # Use hstack for stacking arrays horizontally

    segments, curve = get_curve(a, r=rad, numpoints=numpoints, bernstein_poly=bernstein_poly)
    # curve shape: (<length>, 2)
    x, y = curve.T

    if return_all:
        return x, y, a, segments, curve
    else:
        return x, y, a

def get_random_points(n=5, scale=0.8, mindst=None, rec=0, **kw):
    """Create n random points in the unit square, which are *mindst*
    apart, then scale them."""
    mindst = mindst or 0.7/n
    np_random = kw.get('np_random', np.random)
    a = np_random.rand(n,2)
    d = np.sqrt(np.sum(np.diff(ccw_sort(a), axis=0), axis=1)**2)
    if np.all(d >= mindst) or rec>=200:
        return a*scale
    else:
        return get_random_points(n=n, 
            scale=scale, mindst=mindst, rec=rec+1, np_random=np_random)


def get_random_points_unscaled(n=5, mindst=None, rec=0, **kw):
    """Create n random points in the unit square, which are *mindst*
    apart, then scale them."""
    mindst = mindst or 0.7/n
    np_random = kw.get('np_random', np.random)
    a = np_random.rand(n,2)
    d = np.sqrt(np.sum(np.diff(ccw_sort(a), axis=0), axis=1)**2)
    if np.all(d >= mindst) or rec>=200:
        return a
    else:
        return get_random_points_unscaled(n=n, 
            mindst=mindst, rec=rec+1, np_random=np_random)


def scale_unit_points(a, scale=0.8):
    """Take in an array of points from unit square, scale them, and return."""
    # If assertion is False, print the array
    assert np.max(a) <= 1.0 and np.min(a) >= 0.0, a
    a = a*scale
    return a


def unscale_unit_points(a, scale=0.8):
    """Take in an array of points not from unit square, unscale them, and return."""
    assert np.max(a) <= scale and np.min(a) >= 0.0, a
    a = a/scale
    return a


if __name__ == '__main__':
    global plt 
    import matplotlib.pyplot as plt  # noqa: I001

    fig, ax = plt.subplots()
    ax.set_aspect("equal")

    rad = 0.2
    edgy = 0.5

    for c in np.array([[0,0], [0,1], [1,0], [1,1]]):
        a = get_random_points(n=12, scale=1) + c
        x,y, _ = get_bezier_curve(a,rad=rad, edgy=edgy)
        plt.plot(x,y)

    plt.show()
