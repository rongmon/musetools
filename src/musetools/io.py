"""A module for the input and output manipulation of MUSE data cubes and producing 2D narrowband and wideband surface brightness images."""
from __future__ import print_function, absolute_import, division, unicode_literals
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from astropy.wcs import WCS

def tweak_header(header):
    """
    tweakes the header to make the 3D muse datacube header compatible with astropy image header object

    Parameters
    ----------
    header: 3D datacube header 
        the MUSE datacube header

    returns
    -------
    header: 2D image header
        the modified header compatiable with astropy image
    """
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




def open_muse_cube(fitsfile):
    """
    Input/output function to read the MUSE datacube.
    
    Parameters
    ----------
    fitsfile: string
        string that contains the directory and the data cube fits file name

    Returns
    -------
    wave: numpy.ndarray
        1D array that contains the observed wavelength of the datacube. (In AIR)
    data: numpy.ndarray
        3D array that contains the flux data
    var: numpy.ndarray
        3D array that contains the variance data
    header: fits header
        The MUSE cube's header
    """
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
    """
    A function to produce narrowband surface brightness image at between two wavelengths in the datacube.

    Parameters
    ----------
    minwave: float   
        minimum wavelength of the narrow band
    maxwave: float  
        maximum wavelength of the narrow band
    wave: numpy.ndarray     
        1D array that contains the observed wavelength interval from the data cube
    flux_data: numpy.ndarray 
        3D array that contains the flux data from the MUSE cube.
    plot: bool, optional
        A keyword to show or not show the figure for the narrowband image

    Returns
    -------
    image_SB: numpy.ndarray
        2D numpy array that contains the surface brightness narrowband image

    """
    dlambda = 1.25 # Angstrom for the pixel size along the wavelength axis for the MUSE wide field mode
    q = np.where(( wave > minwave) & (wave < maxwave)) # Defining the chosen wavelength interval
    image = np.sum(flux_data[q,:,:], axis = 1)              # We now sum the wavelength within the given interval
    factor = 10.**(-20.)*dlambda / (0.2)**2. # (maxwave-minwave)
    image_SB = image[0,:,:]*factor
    q_nan = np.isnan(image_SB)
    q_neg = np.where(image_SB < 0.0)
    image_SB[q_nan] = 0.0
    image_SB[q_neg] = 0.0
    if plot== True:
        width_in = 10
        fig=plt.figure(1, figsize=(width_in, 15))
        ax = fig.add_subplot(111)
        ax.imshow(np.log10(np.abs(image_SB[0,:,:])), cmap = plt.get_cmap('viridis'), origin='lower')
        plt.show()

    return image_SB

def wl_image(wave, flux_data, plot=False):
    """
    A function to produce a white light surface brightness image by summing along the wavelength axis of the MUSE datacube.

    Parameters
    ----------
    wave: numpy.ndarray
        1D array that contains the observed wavelength array from the MUSE datacube
    flux_data: numpy.ndarray
        3D array that contains the flux data from the MUSE datacube.
   plot: bool, optional
        A keyword to show or not show the figure for the narrowband image


    Returns
    -------
    image_SB: numpy.ndarray
        2D numpy array that contains the surface brightness whitelight image for the MUSE datacube

    """
    dlambda = 1.25 # Angstrom for the pixel size along the wavelength axis for the MUSE wide field mode
    q=np.where(( wave > float(wave[0])) & (wave < float(wave[-1])))  # Choosing specific wavelength interval to view (In my case, I used all the wavelength interval)
    image = np.sum(flux_data[q,:,:],axis=1)  # We now sum all the wavelength
    factor = 10.**(-20.)*dlambda / (0.2)**2.  # (wave[-1]-wave[0])
    image_SB = image[0,:,:] * factor
    q_nan = np.isnan(image_SB)
    q_neg = np.where(image_SB < 0.0)
    image_SB[q_nan] = 0.0
    image_SB[q_neg] = 0.0
    if plot== True:
        width_in = 10
        fig=plt.figure(1, figsize=(width_in, 15))
        ax = fig.add_subplot(111)
        ax.imshow(np.log10(np.abs(image_SB[0,:,:])), cmap = plt.get_cmap('viridis'), origin='lower')
        plt.show()
    return image_SB

def write_cube(flux, var, header, outfile):
    """
    A function to write a cube (or sub-cube) to a fits file

    Parameters:
    flux: numpy.ndarray
        3D array that contains the flux data 
    var: numpy.ndarray
        3D array that contains the variance data (same size as the flux array)
    header: fitsfile header
        the header for the cube (updated header with any new information for the datacube)
    outfile: str
        a string that contains the directory, to save the file, plus the name of the new fits file name
    """
    #create a new datacube
    hdu1 = fits.PrimaryHDU(s,header=h)
    hdu2 = fits.ImageHDU(sv,header=h)
    new_hdul = fits.HDUList([hdu1, hdu2])
    new_hdul.writeto(outfile, overwrite=True)
    print("Wrote cube to {}".format(outfile))
