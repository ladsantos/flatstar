#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
This auxiliary module holds the definitions of a few useful objects used in the
code.
"""

import numpy as np


# The stellar grid object
class StarGrid(object):
    """
    Object that holds the stellar grid and its properties, as well as useful
    functions.

    Parameters
    ----------
    intensity_array (``numpy.ndarray``):
        Array of normalized intensities.

    radius_px (``int`` or ``float``, optional):
        Stellar radius in units of pixels.

    limb_darkening_law (``None`` or ``str``, optional):
        String with the name of the limb-darkening law.

    ld_coefficients (``float`` or ``array-like``):
        Limb-darkening law coefficients.

    supersampling (``int``, ``float``, or ``None``, optional):
        Supersampling factor.

    upscaling (``int``, ``float``, or ``None``, optional):
        Upscaling factor.

    resample_method (``str`` or ``None``, optional):
        Algorithm used to resize the grid.
    """
    def __init__(self, intensity_array, radius_px, limb_darkening_law,
                 ld_coefficients, supersampling, upscaling, resample_method):
        self.intensity = intensity_array
        self.radius_px = radius_px
        self.limb_darkening_law = limb_darkening_law
        self.ld_coefficients = ld_coefficients
        self.supersampling_factor = supersampling
        self.upscaling_factor = upscaling
        self.resample_method = resample_method

        # Parameters for a transit
        self.planet_px_coordinates = None  # Coords. of planet in pixel space
        self.planet_radius_px = None  # Planet radius in pixel space
        self.planet_impact_parameter = None
        self.planet_phase = None


# The following function is used to calculate distances from a reference point
# in the grid in unit of pixels (cylindrical radius)
def cylindrical_r(grid, reference=(0, 0)):
    """
    Calculate distances from a reference point of the grid in unit of pixels
    (cylindrical radius). The reference (0, 0) is the center of the grid.

    Parameters
    ----------
    grid (``numpy.ndarray``):
        Grid. It does not necessarily need to be square.

    reference (``array-like``, optional):
        Reference point. The coordinates of the center of the grid are (0, 0) in
        pixel space. The reference from which to calculate the distance must
        be in relation to the center. Default is (0, 0).

    Returns
    -------
    r_matrix (``numpy.ndarray``):
        Array containing the values of distances from the reference point in
        units of pixels.
    """
    ref_x, ref_y = reference
    grid_size_x, grid_size_y = np.shape(grid)
    grid_center_x, grid_center_y = np.array([grid_size_x // 2,
                                             grid_size_y // 2])
    coords_x = np.linspace(-grid_center_x, grid_center_x, grid_size_x)
    coords_y = np.linspace(-grid_center_y, grid_center_y, grid_size_y)
    map_x, map_y = np.meshgrid(coords_x - ref_x, coords_y - ref_y)
    r_matrix = (map_x ** 2 + map_y ** 2) ** 0.5
    return r_matrix
