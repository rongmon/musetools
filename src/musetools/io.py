from __future__ import print_function, absolute_import, division, unicode_literals
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from astropy.wcs import WCS

def tweak_header(header):
    '''
    Header tweaks to make the 3D kcwi header compatible with astropy image header object
    '''
    header['NAXIS']=2
    header['WCSDIM']=2
    if 'CRVAL3' in header.keys():
        header.remove('CRVAL3')
    if 'CRPIX3' in header.keys():
        header.remove('CRPIX3')
    if 'CD3_3' in header.keys():
        header.remove('CD3_3')
    if 'NAXIS3' in header.keys():
        header.remove('NAXIS3')
    if 'CUNIT3' in header.keys():
        header.remove('CUNIT3')
    if 'CTYPE3' in header.keys():
        header.remove('CTYPE3')
    if 'CNAME3' in header.keys():
        header.remove('CNAME3')

    return header


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
    print(crval3, cd3_3, np.arange(wavedim))
    # Do it
    wave = crval3 + cd3_3 * (np.arange(wavedim) + 1. - crpix3)# This array contains the wavelength
    header = tweak_header(hdu_hdr)
    return wave, data, var, header


def narrow_band(minwave, maxwave, wave, flux_data,plot=False):
    '''
    minwave:   minimum wavelength of the narrow band
    maxwave:   maximum wavelength of the narrow band
    wave:      The wavelength interval from the data cube
    flux_data: The 2-D flux data from the data cube

    The output of this function is a wavelength narrow band image of the LensedArc
    '''
    q = np.where(( wave > minwave) & (wave < maxwave)) # Defining the chosen wavelength interval
    image = np.sum(flux_data[q,:,:], axis = 1)              # We now sum the wavelength within the given interval
    factor = 10.**(-20.)*(maxwave-minwave) / (0.2)**2.
    image_SB = image[0,:,:]*factor
    if plot== True:
        #title = input('Enter the title of the plot: ')
        width_in = 10
        fig=plt.figure(1, figsize=(width_in, 15))
        #fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(np.log10(np.abs(image[0,:,:])), cmap = plt.get_cmap('viridis'), origin='lower')
        #ax.set_ylim([205,300])
        #ax.set_title(title)
        plt.show()

    return image_SB

def wl_image(wave, flux_data):
    '''
    wave:      is the wavelength array derived from the data cube
    flux_data: the 2-D flux data from the data cube.

    The output of this function is a 2-D image of the LensedArc with summing the total wavelength array.
    '''
    q=np.where(( wave > float(wave[0])) & (wave < float(wave[-1])))  # Choosing specific wavelength interval to view (In my case, I used all the wavelength interval)
    image = np.sum(flux_data[q,:,:],axis=1)  # We now sum all the wavelength
    factor = 10.**(-20.)*(wave[-1]-wave[0]) / (0.2)**2.
    image_SB = image[0,:,:] * factor
    # We will plot the image now using log10 scaling
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    d = ax.imshow(np.log10(np.abs(image[0,:,:])), cmap = plt.get_cmap('viridis'), origin='lower')
    plt.show()
    '''
    return image_SB

def write_cube(flx,var,header,outfile):
    #create a new datacube
    hdu1 = fits.PrimaryHDU(s,header=h)
    hdu2 = fits.ImageHDU(sv,header=h)
    new_hdul = fits.HDUList([hdu1, hdu2])
    new_hdul.writeto(outfile, overwrite=True)
    print("Wrote cube to {}".format(outfile))
