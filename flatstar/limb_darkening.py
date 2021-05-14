#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module is used to compute the limb-darkening functions of stars.
"""

import numpy as np


__all__ = ["linear", "quadratic", "square_root", "logarithmic", "exponential",
           "sing_three", "claret_four"]


# Schwarzschild (1906; Nachrichten von der Königlichen Gesellschaft der
# Wissenschaften zu Göttingen. Mathematisch-Physikalische Klasse, p. 43)
def linear(mu, c, i0=1.0):
    """
    Calculates the intensity of a given cell in the stellar surface using a
    linear limb-darkening law.

    Parameters
    ----------
    mu (``float`` or ``numpy.ndarray``):
        Cosine of the angle between a line normal to the stellar surface and the
        line of sight.

    c (``float``):
        Limb-darkening coefficient.

    i0 (``float``, optional):
        Intensity without limb-darkening. Default is 1.0.

    Returns
    -------
    i_mu (``float`` or ``numpy.ndarray``):
        Intensity with limb-darkening. The format is the same as the input
        ``mu``.
    """
    attenuation = 1 - c * (1 - mu)
    i_mu = i0 * attenuation
    return i_mu


# Source: https://ui.adsabs.harvard.edu/abs/1950HarCi.454....1K/abstract
def quadratic(mu, c, i0=1.0):
    """
    Calculates the intensity of a given cell in the stellar surface using a
    quadratic limb-darkening law.

    Parameters
    ----------
    mu (``float`` or ``numpy.ndarray``):
        Cosine of the angle between a line normal to the stellar surface and the
        line of sight.

    c (``array-like``):
        Limb-darkening coefficients in the order c1, c2.

    i0 (``float``, optional):
        Intensity without limb-darkening. Default is 1.0.

    Returns
    -------
    i_mu (``float`` or ``numpy.ndarray``):
        Intensity with limb-darkening. The format is the same as the input
        ``mu``.
    """
    c1, c2 = c
    attenuation = 1 - c1 * (1 - mu) - c2 * (1 - mu) ** 2
    i_mu = i0 * attenuation
    return i_mu


# Source: https://ui.adsabs.harvard.edu/abs/1992A%26A...259..227D/abstract
def square_root(mu, c, i0=1.0):
    """
    Calculates the intensity of a given cell in the stellar surface using a
    square-root limb-darkening law.

    Parameters
    ----------
    mu (``float`` or ``numpy.ndarray``):
        Cosine of the angle between a line normal to the stellar surface and the
        line of sight.

    c (``array-like``):
        Limb-darkening coefficients in the order c1, c2.

    i0 (``float``, optional):
        Intensity without limb-darkening. Default is 1.0.

    Returns
    -------
    i_mu (``float`` or ``numpy.ndarray``):
        Intensity with limb-darkening. The format is the same as the input
        ``mu``.
    """
    c1, c2 = c
    attenuation = 1 - c1 * (1 - mu) - c2 * (1 - mu ** 0.5)
    i_mu = i0 * attenuation
    return i_mu


# Source: https://ui.adsabs.harvard.edu/abs/1970AJ.....75..175K/abstract
def logarithmic(mu, c, i0=1.0):
    """
    Calculates the intensity of a given cell in the stellar surface using a
    logarithmic limb-darkening law.

    Parameters
    ----------
    mu (``float`` or ``numpy.ndarray``):
        Cosine of the angle between a line normal to the stellar surface and the
        line of sight.

    c (``array-like``):
        Limb-darkening coefficients in the order c1, c2.

    i0 (``float``, optional):
        Intensity without limb-darkening. Default is 1.0.

    Returns
    -------
    i_mu (``float`` or ``numpy.ndarray``):
        Intensity with limb-darkening. The format is the same as the input
        ``mu``.
    """
    c1, c2 = c
    attenuation = 1 - c1 * (1 - mu) - c2 * mu * np.log(mu) ** 2
    i_mu = i0 * attenuation
    return i_mu


# Source: https://ui.adsabs.harvard.edu/abs/2003A%26A...412..241C/abstract
def exponential(mu, c, i0=1.0):
    """
    Calculates the intensity of a given cell in the stellar surface using an
    exponential limb-darkening law.

    Parameters
    ----------
    mu (``float`` or ``numpy.ndarray``):
        Cosine of the angle between a line normal to the stellar surface and the
        line of sight.

    c (``array-like``):
        Limb-darkening coefficients in the order c1, c2.

    i0 (``float``, optional):
        Intensity without limb-darkening. Default is 1.0.

    Returns
    -------
    i_mu (``float`` or ``numpy.ndarray``):
        Intensity with limb-darkening. The format is the same as the input
        ``mu``.
    """
    c1, c2 = c
    attenuation = 1 - c1 * (1 - mu) - c2 / (1 - np.exp(mu))
    i_mu = i0 * attenuation
    return i_mu


# Source: https://ui.adsabs.harvard.edu/abs/2009A%26A...505..891S/abstract
def sing_three(mu, c, i0=1.0):
    """
    Calculates the intensity of a given cell in the stellar surface using the
    Sing et al (2009) limb-darkening law.

    Parameters
    ----------
    mu (``float`` or ``numpy.ndarray``):
        Cosine of the angle between a line normal to the stellar surface and the
        line of sight.

    c (``array-like``):
        Limb-darkening coefficients in the order c1, c2, c3.

    i0 (``float``, optional):
        Intensity without limb-darkening. Default is 1.0.

    Returns
    -------
    i_mu (``float`` or ``numpy.ndarray``):
        Intensity with limb-darkening. The format is the same as the input
        ``mu``.
    """
    c1, c2, c3 = c
    attenuation = 1 - c1 * (1 - mu) - c2 * (1 - mu ** (3 / 2)) - c3 * \
        (1 - mu ** 2)
    i_mu = i0 * attenuation
    return i_mu


# Source: https://ui.adsabs.harvard.edu/abs/2000A%26A...363.1081C/abstract
def claret_four(mu, c, i0=1.0):
    """
    Calculates the intensity of a given cell in the stellar surface using the
    Claret et al. (2000) limb-darkening law.

    Parameters
    ----------
    mu (``float`` or ``numpy.ndarray``):
        Cosine of the angle between a line normal to the stellar surface and the
        line of sight.

    c (``array-like``):
        Limb-darkening coefficients in the order c1, c2, c3, c4.

    i0 (``float``, optional):
        Intensity without limb-darkening. Default is 1.0.

    Returns
    -------
    i_mu (``float`` or ``numpy.ndarray``):
        Intensity with limb-darkening. The format is the same as the input
        ``mu``.
    """
    c1, c2, c3, c4 = c
    attenuation = 1 - c1 * (1 - mu ** 0.5) - c2 * (1 - mu) - c3 * \
        (1 - mu ** (3 / 2)) - c4 * (1 - mu ** 2)
    i_mu = i0 * attenuation
    return i_mu
