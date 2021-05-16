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

    radius (``int`` or ``float``, optional):
        Stellar radius in units of the grid size. Default is 0.5.

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
    def __init__(self, intensity_array, radius, limb_darkening_law,
                 ld_coefficients, supersampling, upscaling, resample_method):
        self.intensity = intensity_array
        self.radius = radius
        self.limb_darkening_law = limb_darkening_law
        self.ld_coefficients = ld_coefficients
        self.supersampling_factor = supersampling
        self.upscaling_factor = upscaling
        self.resample_method = resample_method


# The following function is used to calculate distances from the center of
# the grid in unit of pixels (cylindrical radius)
def cylindrical_r(grid):
    """
    Calculate distances from the center of the grid in unit of pixels
    (cylindrical radius).

    Returns
    -------
    r_matrix (``numpy.ndarray``):
        Array containing the values of distances from the center of the grid
        in units of pixels.
    """
    grid_size = np.shape(grid)[0]
    center = grid_size // 2
    coords = np.linspace(-center, center, grid_size)
    coords_x, coords_y = np.meshgrid(coords, coords)
    r_matrix = (coords_x ** 2 + coords_y ** 2) ** 0.5
    return r_matrix
