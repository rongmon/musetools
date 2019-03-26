from __future__ import print_function, absolute_import, division, unicode_literals
import numpy as np


def veldiff(wave,wave_center):
    z = (wave / wave_center) - 1.
    c = 299792.458  # Speed of light in km/s
    beta = ((z + 1.)**2. - 1.)/((z+1.)**2. + 1.)
    del_v = beta * c
    return del_v
