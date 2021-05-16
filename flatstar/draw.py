#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module is used to compute a grid containing drawing of disks of stars with
limb darkening.
"""

import numpy as np
import warnings
from flatstar import limb_darkening, utils

from PIL import Image, ImageDraw

__all__ = ["star", "planet_transit"]


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
def star(grid_size, radius=0.5, limb_darkening_law=None, ld_coefficient=None,
         custom_limb_darkening=None, supersampling=None, upscaling=None,
         resample_method=None):
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

    radius (``int`` or ``float``, optional):
        Stellar radius in units of ``grid_size``. Default is 0.5.

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
        For low-resolution grid sizes, in order to avoid intensity maps with
        hard edges, you can supersample  the array by a certain factor defined
        by this parameter, and then the map is downscaled to your requested grid
        size using the algorithm  defined in ``resample_method``. Default is
        ``None`` (no supersampling).

    upscaling (``int``, ``float``, or ``None``, optional):
        For fast output of high-resolution grids, you may want to upscale
        them from a low-resolution setup to save about one order of magnitude
        in computation time. This parameter is the factor by which to upscale
        the grids to match the requested grid size. The resizing algorithm is
        defined in ``resample_method``. Default is ``None`` (no upscaling).

    resample_method (``str`` or ``None``, optional):
        Resampling algorithm. The options currently available are:
        ``"nearest"``, ``"box"``, ``"bilinear"``, ``"hamming"``, ``"bicubic"``,
        and ``"lanczos"``. If ``None``, then fallback to ``"box"``. Default
        is ``None``.

    Returns
    -------
    grid (``flatstar.utils.StarGrid`` object):
        Intensity map of the star.
    """
    # Emit a warning if the radius is larger than 0.5
    if radius > 0.5:
        warnings.warn('Using a radius larger than 0.5 will yield inaccurate '
                      'intensities.', RuntimeWarning)

    # Define the effective grid size on which to start
    if supersampling is not None:
        effective_grid_size = int(round(supersampling * grid_size))
    elif upscaling is not None:
        effective_grid_size = int(grid_size // upscaling)
    else:
        effective_grid_size = grid_size
    shape = (effective_grid_size, effective_grid_size)

    # Draw the host star
    star_radius = radius * effective_grid_size
    center = effective_grid_size // 2
    star_array = _disk(center=(center, center), radius=star_radius,
                       shape=shape)

    # We need to know what is the distance of each pixel from the stellar center
    # There is a useful function in ``utils`` for that, and it does not use
    # for-loops
    r_array = utils.cylindrical_r(star_array)

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

    # We use PIL.Image to perform the resizing
    im = Image.fromarray(star_array)
    final_shape = (grid_size, grid_size)

    # Resize the array to the desired grid size if necessary
    if supersampling is not None or upscaling is not None:
        pass
    else:  # No resizing needed
        norm = np.sum(star_array)
        intensity_array = star_array / norm
        grid = utils.StarGrid(intensity_array, radius, limb_darkening_law,
                              ld_coefficient, supersampling, upscaling,
                              resample_method)
        return grid

    # If the resample_method is defined by the user with one of the
    # available options, then use it
    if resample_method is not None:
        try:
            final_star_array = im.resize(
                final_shape, resample=RESAMPLING_ALIAS[resample_method])
        except KeyError:
            raise NotImplementedError("This resampling method is not "
                                      "implemented.")
    # If the resample_method is not defined, then simply use a box interpolation
    else:
        resample_method = 'box'
        final_star_array = im.resize(final_shape, resample=Image.BOX)
    # Finally make `star_array` as a copy of the downsampled array
    star_array = np.copy(final_star_array)

    # Adding the star to the grid
    norm = np.sum(star_array)
    intensity_array = star_array / norm

    grid = utils.StarGrid(intensity_array, radius, limb_darkening_law,
                          ld_coefficient, supersampling, upscaling,
                          resample_method)
    return grid


# Draw a transit on a star
def planet_transit(star_grid, planet_to_star_ratio, impact_parameter=0.0,
                   phase=0.0):
    """
    Draw a transit in the ``StarGrid`` object.

    Parameters
    ----------
    star_grid (``flatstar.utils.StarGrid`` object):

    planet_to_star_ratio (``float``):
        Ratio between the radii of the planet and the star.

    impact_parameter (``float``, optional):
        Impact parameter of the transit in units of stellar radii. Default is 0.

    phase (``float``, optional):
        Phase of the transit. -0.5, 0.0, and +0.5 correspond respectively to the
        time of first contact, transit mid-center, and time of fourth contact.
        Default is 0.

    Returns
    -------
    star_grid (``flatstar.utils.StarGrid`` object):
        Updated ``StarGrid`` object containing the transit.
    """
    b = impact_parameter
    rp_rs = planet_to_star_ratio

    # Radii of the star and the planet in units of grid size
    grid_length_x, grid_length_y = np.shape(star_grid.intensity)
    star_radius = star_grid.radius * grid_length_x
    planet_radius = star_radius * rp_rs

    # Before drawing the planet, we need to figure out the exact coordinate
    # of the center of the planet
    y_p = (impact_parameter * star_radius) + grid_length_y // 2

    # The x coordinate of the planet is a bit trickier to figure out. Since we
    # want the -0.5 and 0.5 phases to always match the times of first and fourth
    # contact, respectively, x_p will depend on the impact parameter in a very
    # non-trivial manner. Sorry for the ugliness, but it is the price of
    # convenience!
    beta = (1 - (b * star_radius / (planet_radius + star_radius)) ** 2) ** 0.5
    alpha = grid_length_x // 2 - (planet_radius + star_radius) * beta
    x_p = alpha + (phase + 0.5) * 2 * (planet_radius + star_radius) * beta

    # And now we draw it
    planet = _disk(center=(x_p, y_p), radius=planet_radius,
                   shape=np.shape(star_grid.intensity),
                   value=1.0)
    updated_intensity = star_grid.intensity - planet
    # Remove infinities in the planet disk and set the intensity to zero
    updated_intensity[updated_intensity < 0] = 0.0

    # Update the ``StarGrid`` object
    star_grid.intensity = updated_intensity
    star_grid.planet_px_coordinates = (x_p, y_p)
    star_grid.planet_to_star_ratio = planet_to_star_ratio
    star_grid.planet_impact_parameter = b
    star_grid.phase = phase
    return star_grid


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
