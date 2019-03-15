# importing the required libraries
from __future__ import division
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt


fitsfile = '/home/ahmed/Research/data/RCS0327_16mc_zap.fits'
# input("Enter the path to your file: ")
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


    
def vel(xcen, ycen, squaresize, lam_galaxy, wave, flux_data):
    # lam_galaxy : represents the observed wavelength of the gas from the galaxy itself
    halfbox = (squaresize - 1)//2
    flux = flux_data[:,ycen-halfbox:ycen+halfbox+1, xcen-halfbox:xcen+halfbox+1]
    spec = np.sum(flux, axis=(1,2))
    del_v = np.zeros(len(wave))
    for i in range(len(wave)-1):
        z = (wave[i] / lam_galaxy) - 1
        c = 299792.458  # Speed of light in km/s
        beta = ((z + 1)**2 - 1)/((z+1)**2 + 1)
        del_v[i] = beta * c
    
    plt.figure()
    plt.title('The spectrum centered at ('+str(xcen)+','+str(ycen)+') using the velocity with wavelength:'+str(lam_galaxy)+'')
    plt.xlabel('Velocity')
    plt.ylabel('Normalized Flux')
    plt.plot(del_v[1800:2650],spec[1800:2650]) 
    plt.show()   

    

#lam_galaxy = float(input('Enter the wavelength value of the emission line from the galaxy: ')) # This is calculated using the redshift z
# which you got using z of the galaxy: '))
#squaresize = int(input("Enter the value of the square side length in pixels: "))
xcen = int(input("Enter the x-coordinate of your central pixel of the square: "))
ycen = int(input("Enter the y-coordinate of your central pixel of the square: "))
lam_galaxy = np.array([7550.1477, 7569.5256, 7020.4671, 6983.955, 7091.4177, 7054.1658])

for j in np.nditer(lam_galaxy):
    vel(xcen, ycen, 5, j, wave, data)



