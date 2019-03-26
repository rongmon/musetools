# importing the required libraries
from __future__ import print_function, absolute_import, division, unicode_literals

from astropy.io import fits
import numpy as np

# This is the function which will give us the spectrum of each square
def extract_square(xcen, ycen, wave, flux_data, var, squaresize, outfile=None):
    halfbox = (squaresize - 1)//2
    flux = flux_data[:,ycen-halfbox:ycen+halfbox+1, xcen-halfbox:xcen+halfbox+1]
    sub_var = var[:, ycen-halfbox:ycen+halfbox+1, xcen-halfbox:xcen+halfbox+1]

    spec = np.sum(flux, axis=(1,2))
    err_spec = np.sqrt(np.sum(sub_var, axis=(1,2)))

    return spec, err_spec 









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
