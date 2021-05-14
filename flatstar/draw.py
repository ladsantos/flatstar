#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module is used to compute a grid containing drawing of disks of stars with
limb darkening.
"""

import numpy as np
import warnings
from flatstar import limb_darkening

from PIL import Image, ImageDraw

__all__ = ["star", "planet"]


IMPLEMENTED_LD_LAW = {"linear": limb_darkening.linear,
                      "quadratic": limb_darkening.quadratic,
                      "square-root": limb_darkening.square_root,
                      "log": limb_darkening.logarithmic,
                      "logarithmic": limb_darkening.logarithmic,
                      "exp": limb_darkening.exponential,
                      "exponential": limb_darkening.exponential,
                      "sing": limb_darkening.sing_three,
                      "sing-three": limb_darkening.sing_three,
                      "s3": limb_darkening.sing_three,
                      "claret": limb_darkening.claret_four,
                      "claret-four": limb_darkening.claret_four,
                      "c4": limb_darkening.claret_four}

RESAMPLING_ALIAS = {"nearest": Image.NEAREST, "box": Image.BOX,
                    "bilinear": Image.BILINEAR, "hamming": Image.HAMMING,
                    "bicubic": Image.BICUBIC, "lanczos": Image.LANCZOS}


# Draw a star
def star(grid_size, limb_darkening_law=None, ld_coefficient=None,
         custom_limb_darkening=None, supersampling=None, resample_method=None):
    """
    Make a normalized drawing of a star with a corresponding limb-darkening law
    in a square grid. The normalization is made in such a way that the flattened
    sum of the values inside the two-dimensional array is equal to 1.0. The
    normalization factor is calculated before the resampling, so more complex
    resampling algorithms may produce more inaccurate normalizations (by a
    factor of a few to hundreds of ppm) depending on the requested grid size and
    supersampling factor. If very precise normalized maps are required, then it
    is better to not use supersampling or use a ``"box"`` resampling algorithm.

    Parameters
    ----------
    grid_size (``int``):
        Size of the square grid in number pixels.

    limb_darkening_law (``None`` or ``str``, optional):
        String with the name of the limb-darkening law. The options currently
        implemented are: ``'linear'``, ``'quadratic'``, ``'square-root'``,
        ``'logarithmic'`` (or ``'log'``), ``'exponential'`` (or ``'exp'``),
        ``'sing-three'`` (or ``'sing'``, or ``'s3'``), ``'claret-four'``
        (or ``'claret'``, or ``'c4'``), ``None`` (no limb-darkening), or
        ``'custom'``. In case you choose the latter, you need to provide a
        callable function that defines your custom law using the parameter
        ``custom_limb_darkening``. Default is ``None``.

    ld_coefficient (``float`` or ``array-like``):
        In case of a linear limb-darkening law, the value of the coefficient
        should be a float. In all other options it should be array-like. Default
        is ``None``.

    custom_limb_darkening (``callable`` or ``None``, optional)
        In case you want to use a custom limb-darkening law, you need
        provide a function that defines it. The first parameter of this function
        must be ``mu`` (cosine of the angle between a line normal to the stellar
        surface and the line of sight), and the second must be the coefficient
        (in case it uses multiple coefficients, it must accept them as an
        array-like object). Default is ``None``.

    supersampling (``int``, ``float``, or ``None``, optional):
        In order to avoid "pixelized" maps, you can supersample the array by a
        certain factor defined by this parameter, and then the map is resampled
        to your requested grid size using the algorithm defined in
        ``resample_method``. Default is ``None`` (no supersampling).

    resample_method (``str`` or ``None``, optional):
        Resampling algorithm. The options currently available are:
        ``"nearest"``, ``"box"``, ``"bilinear"``, ``"hamming"``, ``"bicubic"``,
        and ``"lanczos"``. If ``None``, then fallback to ``"box"``. Default
        is ``None``.

    Returns
    -------

    """
    # Define the effective grid size on which to start
    if supersampling is None:
        effective_grid_size = grid_size
    else:
        effective_grid_size = int(round(supersampling * grid_size))
    shape = (effective_grid_size, effective_grid_size)

    # Draw the host star
    star_radius = effective_grid_size // 2
    star_array = _disk(center=(star_radius, star_radius), radius=star_radius,
                       shape=shape)

    # We need to know what is the distance of each pixel from the stellar center
    # There is a little bit of hack in here to avoid using for-loops
    coords = np.linspace(-star_radius, star_radius, effective_grid_size)
    coords_x, coords_y = np.meshgrid(coords, coords)
    r_array = (coords_x ** 2 + coords_y ** 2) ** 0.5

    # Now we calculate the mu for the limb-darkening law
    # We ignore a RuntimeWarning here because any NaN will be multiplied by zero
    # anyway.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mu = (1 - (r_array / star_radius) ** 2) ** 0.5
    mu[np.isnan(mu)] = 0.0

    # Apply the limb-darkening law
    # No limb-darkening
    if limb_darkening_law is None:
        pass
    # Custom limb-darkening law
    elif limb_darkening_law == 'custom':
        star_array *= custom_limb_darkening(mu, ld_coefficient)
    # Laws implemented in this code
    else:
        try:
            star_array *= IMPLEMENTED_LD_LAW[limb_darkening_law](mu,
                                                                 ld_coefficient)
        except KeyError:
            raise NotImplementedError("This limb-darkening law is not "
                                      "implemented.")

    # Calculate the normalization factor
    norm = np.sum(star_array)  # Normalization factor is the total intensity

    # Downsample the supersampled array to the desired grid size
    if supersampling is not None:
        # We use PIL.Image to perform the downsampling
        im = Image.fromarray(star_array)
        final_shape = (grid_size, grid_size)
        # If the resample_method is defined by the user with one of the
        # available options, then use it
        if resample_method is not None:
            try:
                final_star_array = im.resize(
                    final_shape, resample=RESAMPLING_ALIAS[resample_method])
            except KeyError:
                raise NotImplementedError("This resampling method is not "
                                          "implemented.")
        # If the resample_method is not defined, then simply use a nearest
        # interpolation
        else:
            final_star_array = im.resize(final_shape, resample=Image.BOX)
        # Finally make `star_array` as a copy of the downsampled array
        star_array = np.copy(final_star_array)
    # If no supersampling was used
    else:
        pass

    # Adding the star to the grid
    if supersampling is not None:
        grid = star_array / (norm / supersampling ** 2)
    else:
        grid = star_array / norm

    return grid


# Why not draw a planet?
def planet():
    raise NotImplementedError("Drawing a planet is not implemented yet.")


# Combine the star drawing with other drawings
def combine():
    raise NotImplementedError("This feature is not implemented yet.")


# General function to draw a disk
def _disk(center, radius, shape, value=1.0):
    """
    Hidden function used to draw disks with PIL.

    Parameters
    ----------
    center (``int``):
        Center of the disk in pixel space.

    radius (``int``):
        Radius of the disk in number of pixels.

    shape (``array-like``):
        Shape of the grid in number of pixels.

    value (``float``, optional):
        Value with which to fill the disk. Default is 1.0.

    Returns
    -------
    disk (``numpy.ndarray``):
        Grid containing a drawing of the disk.
    """
    top_left = (center[0] - radius, center[1] - radius)
    bottom_right = (center[0] + radius, center[1] + radius)
    image = Image.new('1', shape)
    draw = ImageDraw.Draw(image)
    draw.ellipse([top_left, bottom_right], outline=1, fill=1)
    disk = np.reshape(np.array(list(image.getdata())), shape) * value
    return disk