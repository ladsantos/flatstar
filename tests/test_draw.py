#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from flatstar import draw


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
        try:
            star = draw.star(grid_size,
                             limb_darkening_law=IMPLEMENTED_LD_LAWS[i],
                             ld_coefficient=TEST_COEFFICIENTS[i])
            did_it_work *= 1
        except:
            did_it_work *= 0
            exception_in.append(IMPLEMENTED_LD_LAWS[i])

    assert(did_it_work == 1)


# Test the supersampling and the resampling (aka downsampling)
def test_supersampling(grid_size=100, factor=np.random.random() * 10, use_ld=6):
    did_it_work = 1
    exception_in = []
    for i in range(N_SAMPLERS):
        try:
            star = draw.star(grid_size,
                             limb_darkening_law=IMPLEMENTED_LD_LAWS[use_ld],
                             ld_coefficient=TEST_COEFFICIENTS[use_ld],
                             supersampling=factor,
                             resample_method=IMPLEMENTED_SAMPLERS[i])
            did_it_work *= 1
        except:
            did_it_work *= 0
            exception_in.append(IMPLEMENTED_SAMPLERS[i])

    assert(did_it_work == 1)


# Test the upscaling
def test_upscaling(grid_size=500, factor=np.random.random() * 10, use_ld=6):
    did_it_work = 1
    exception_in = []
    for i in range(N_SAMPLERS):
        try:
            star = draw.star(grid_size,
                             limb_darkening_law=IMPLEMENTED_LD_LAWS[use_ld],
                             ld_coefficient=TEST_COEFFICIENTS[use_ld],
                             upscaling=factor,
                             resample_method=IMPLEMENTED_SAMPLERS[i])
            did_it_work *= 1
        except:
            did_it_work *= 0
            exception_in.append(IMPLEMENTED_SAMPLERS[i])

    assert(did_it_work == 1)
