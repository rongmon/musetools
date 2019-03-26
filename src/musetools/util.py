from __future__ import print_function, absolute_import, division, unicode_literals
import numpy as np


def veldiff(wave,wave_center):
    z = (wave / wave_center) - 1.
    c = 299792.458  # Speed of light in km/s
    beta = ((z + 1.)**2. - 1.)/((z+1.)**2. + 1.)
    del_v = beta * c
    return del_v

def open_muse_cube(fitsfile):
    #fitsfile = input("Enter the path to your file: ")
    a = fits.open(fitsfile)
    data = a[1].data  # the spectrum data are included in the data extension of the fits file
    # data.shape   # you can use this command to know the dimensions of your data cube
    hdu_hdr = a[1].header  # reading the header of the data into a variable called hdu_hdr
    # We will transform the pixels' indexes into wavelength array
    crval3 = hdu_hdr['CRVAL3']  # This is the starting value of the wavelength
    crpix3 = hdu_hdr['CRPIX3']  # The pixel that has the value of CRVAL3
    cd3_3 = hdu_hdr['CD3_3']    # The range of wavelength difference delta_lambda
    wavedim = hdu_hdr['NAXIS3'] # The dimension of the data axis 3 (Wavelength)
    # Do it
    wave = crval3 + (crpix3 + np.arange(0, wavedim, 1.0)) * cd3_3 # This array contains the wavelength
    return data, wave
