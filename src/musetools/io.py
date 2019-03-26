from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt

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
'''
username=getpass.getuser()

if username == 'bordoloi':
	fitsfile = '/Users/bordoloi/Dropbox/MUSE/LensedArc/RCS0327_16mc_zap.fits'
else:
	fitsfile = '/home/ahmed/astro/data/RCS0327_16mc_zap.fits'
'''
