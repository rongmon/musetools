from __future__ import print_function, absolute_import, division, unicode_literals
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt

'''
This input and output file contains all the required function to open the MUSE data
cube and the output images from it.
'''

def open_muse_cube(fitsfile):
    '''
    This function takes the path to the data cube and gives as an output two arrays
    the 2-D flux data and the wavelength interval.
    '''
    #fitsfile = input("Enter the path to your file: ")
    a = fits.open(fitsfile)
    data = a[1].data  # the spectrum data are included in the data extension of the fits file
    # data.shape   # you can use this command to know the dimensions of your data cube
    var = a[2].data
    hdu_hdr = a[1].header  # reading the header of the data into a variable called hdu_hdr
    # We will transform the pixels' indexes into wavelength array

    crval3 = hdu_hdr['CRVAL3']  # This is the starting value of the wavelength
    crpix3 = hdu_hdr['CRPIX3']  # The pixel that has the value of CRVAL3
    cd3_3 = hdu_hdr['CD3_3']    # The range of wavelength difference delta_lambda
    wavedim = hdu_hdr['NAXIS3'] # The dimension of the data axis 3 (Wavelength)
    # Do it
    wave = crval3 + (crpix3 + np.arange(0, wavedim, 1.0)) * cd3_3 # This array contains the wavelength
    return wave, data, var

#def open_muse_var(fitsfile):
#    a = fits.open(fitsfile)
#    var = a[2].data
#    return var 

def narrow_band(minwave, maxwave, wave, flux_data):
    '''
    minwave:   minimum wavelength of the narrow band
    maxwave:   maximum wavelength of the narrow band
    wave:      The wavelength interval from the data cube
    flux_data: The 2-D flux data from the data cube

    The output of this function is a wavelength narrow band image of the LensedArc
    '''
    q = np.where(( wave > minwave) & (wave < maxwave)) # Defining the chosen wavelength interval
    image = np.sum(flux_data[q,:,:], axis = 1)              # We now sum the wavelength within the given interval
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(np.log10(np.abs(image[0,:,:])), cmap = plt.get_cmap('viridis'), origin='lower')
    plt.show()


def w_image(wave, flux_data):
    '''
    wave:      is the wavelength array derived from the data cube
    flux_data: the 2-D flux data from the data cube.

    The output of this function is a 2-D image of the LensedArc with summing the total wavelength array.
    '''
    q=np.where(( wave > float(wave[0])) & (wave < float(wave[len(wave)-1])))  # Choosing specific wavelength interval to view (In my case, I used all the wavelength interval)
    image = np.sum(flux_data[q,:,:],axis=1)  # We now sum all the wavelength
    # We will plot the image now using log10 scaling
    fig = plt.figure()
    ax = fig.add_subplot(111)
    d = ax.imshow(np.log10(np.abs(image[0,:,:])), cmap = plt.get_cmap('viridis'), origin='lower')
    plt.show()
