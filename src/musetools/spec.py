# importing the required libraries
from __future__ import print_function, absolute_import, division, unicode_literals

from astropy.io import fits
import numpy as np
#import matplotlib.pyplot as plt


'''
fitsfile = input("Enter the path to your file: ")
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
wave = crval3 + (crpix3 + np.arange(0, wavedim, 1.0)) * cd3_3  # This array contains the wavelength 

####
'''
# This is the function which will give us the spectrum of each square
def extract_square(xcen, ycen, wave, flux_data, squaresize, outfile=None):
    halfbox = (squaresize - 1)//2
    flux = flux_data[:,ycen-halfbox:ycen+halfbox+1, xcen-halfbox:xcen+halfbox+1]
    spec = np.sum(flux, axis=(1,2))
    return spec 









'''
    fig = plt.figure()
    plt.title('The spectrum of the square centered at ('+str(xcen)+','+str(ycen)+')')
    plt.xlabel('Wavelength')
    plt.ylabel('Normalized Flux')
    plt.plot(wave[2300:2500],spec[2300:2500])
    # use the indexes interval [2300:2500] to get Mg II lines
    plt.show()
####

squaresize = int(input("Enter the value of the square side length in pixels: "))
xcen = int(input("Enter the x-coordinate of your central pixel of the square: "))
ycen = int(input("Enter the y-coordinate of your central pixel of the square: "))

sq_spectrum(xcen, ycen, wave, data, squaresize)
'''



