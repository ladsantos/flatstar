#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from flatstar import draw


REQUIRED_INTENSITY_PRECISION = 1E-6

IMPLEMENTED_LD_LAWS = ["linear", "quadratic", "square-root", "log", "exp",
                       "sing", "claret"]
N_LAWS = len(IMPLEMENTED_LD_LAWS)
TEST_COEFFICIENTS = [np.random.random(),
                     np.random.random(size=2),
                     np.random.random(size=2),
                     np.random.random(size=2),
                     np.random.random(size=2),
                     np.random.random(size=3),
                     np.random.random(size=4)]

IMPLEMENTED_SAMPLERS = ["nearest", "box", "bilinear", "hamming", "bicubic",
                        "lanczos"]
N_SAMPLERS = len(IMPLEMENTED_SAMPLERS)


# Test each limb-darkening law
def test_ld_laws(grid_size=101):
    did_it_work = 1
    exception_in = []
    for i in range(N_LAWS):
        star = draw.star(grid_size,
                         limb_darkening_law=IMPLEMENTED_LD_LAWS[i],
                         ld_coefficient=TEST_COEFFICIENTS[i])
        total_intensity = np.sum(star.intensity)
        obtained_precision = abs(1.0 - total_intensity)
        assert (obtained_precision < REQUIRED_INTENSITY_PRECISION)


# Test a custom limb-darkening law
def test_custom_ld(grid_size=200):
    # Let's come up with a semi-arbitrary LD law here
    def custom_ld(mu, c, i0=1.0):
        c1, c2 = c
        attenuation = 1 - c1 * (1 - mu ** 0.9) - c2 * (1 - mu ** (3 / 2))
        i_mu = i0 * attenuation
        return i_mu

    star = draw.star(grid_size,
                     limb_darkening_law='custom',
                     ld_coefficient=TEST_COEFFICIENTS[1],
                     custom_limb_darkening=custom_ld)
    total_intensity = np.sum(star.intensity)
    obtained_precision = abs(1.0 - total_intensity)
    assert(obtained_precision < REQUIRED_INTENSITY_PRECISION)


# Test no limb-darkening law
def test_no_ld(grid_size=200):
    star = draw.star(grid_size,
                     limb_darkening_law=None)
    total_intensity = np.sum(star.intensity)
    obtained_precision = abs(1.0 - total_intensity)
    assert (obtained_precision < REQUIRED_INTENSITY_PRECISION)


# Test the supersampling and the resampling (aka downsampling)
def test_supersampling(grid_size=100, factor=np.random.randint(2, 10), use_ld=6):
    did_it_work = 1
    exception_in = []
    for i in range(N_SAMPLERS):
        star = draw.star(grid_size,
                         limb_darkening_law=IMPLEMENTED_LD_LAWS[use_ld],
                         ld_coefficient=TEST_COEFFICIENTS[use_ld],
                         supersampling=factor,
                         resample_method=IMPLEMENTED_SAMPLERS[i])
        total_intensity = np.sum(star.intensity)
        obtained_precision = abs(1.0 - total_intensity)
        assert (obtained_precision < REQUIRED_INTENSITY_PRECISION)


# Test the upscaling
def test_upscaling(grid_size=500, factor=np.random.random() * 10, use_ld=6):
    did_it_work = 1
    exception_in = []
    for i in range(N_SAMPLERS):
        star = draw.star(grid_size,
                         limb_darkening_law=IMPLEMENTED_LD_LAWS[use_ld],
                         ld_coefficient=TEST_COEFFICIENTS[use_ld],
                         upscaling=factor,
                         resample_method=IMPLEMENTED_SAMPLERS[i])
        total_intensity = np.sum(star.intensity)
        obtained_precision = abs(1.0 - total_intensity)
        assert (obtained_precision < REQUIRED_INTENSITY_PRECISION)


# Test drawing a transit
def test_transit(grid_size=2001, planet_to_star_ratio=0.15,
                 transit_required_precision=1E-3):
    star_grid = draw.star(grid_size)
    total_intensity_0 = np.sum(star_grid.intensity)
    transit_grid = draw.planet_transit(star_grid, planet_to_star_ratio)
    total_intensity_1 = np.sum(transit_grid.intensity)
    transit_depth = total_intensity_0 - total_intensity_1
    obtained_precision = abs(transit_depth - planet_to_star_ratio ** 2)
    assert (obtained_precision < transit_required_precision)
