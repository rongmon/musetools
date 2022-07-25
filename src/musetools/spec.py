"""A module for producing spectra from the MUSE datacube"""
# importing the required libraries
from __future__ import print_function, absolute_import, division, unicode_literals

from astropy.io import fits
import numpy as np
from astropy.modeling import models, fitting
from linetools.spectra.xspectrum1d import XSpectrum1D
import matplotlib.pyplot as plt
import warnings


def extract_square(xcen, ycen, wave, flux_data, var, squaresize=5, outfile=None):
    """
    A function to produce a spectrum from a square region using summation along the spatial axes of the region
    in the MUSE datacube.

    Parameters
    ----------
    xcen: int
        the x-pixel coordinate of the center of the square region.
    ycen: int
        the y-pixel coordinate of the center of the square region
    wave: numpy.ndarray
        1D array that contains the observed wavelength array
    flux_data: numpy.ndarray
        3D array that contains the flux from the MUSE datacube
    var: numpy.ndarray
        3D array that contains the variance from the MUSE datacube
    squqaresize: int, optional
        the size of the side of the square regions in MUSE spatial pixels (1 spatial pixel = 0.2 arcsecond)
    outfile: str, optional
        saving directory name plus the name of the output file

    Returns
    -------
    xspec: XSpectrum1D object
        Object that contains the wavelength, flux, and the flux error.
    """
    halfbox = (squaresize - 1)//2
    flux = flux_data[:,ycen-halfbox:ycen+halfbox+1, xcen-halfbox:xcen+halfbox+1]
    sub_var = var[:, ycen-halfbox:ycen+halfbox+1, xcen-halfbox:xcen+halfbox+1]

    spec = np.sum(flux, axis=(1,2))
    err_spec = np.sqrt(np.sum(sub_var, axis=(1,2)))


    # Object
    xspec = XSpectrum1D.from_tuple((wave, spec, err_spec))

    if outfile is not None:
        xspec.write_to_fits(outfile)

    return xspec






def extract_weighted_spectrum(flux,variance,wave,verbose=False,weights='Gaussian',porder=9):
    """
    A function to extract a weighted spectrum using a white light image of the region in the MUSE cube.

    Parameters
    ----------
    flux: numpy.ndarray
        3D array that contains the MUSE flux cube in the aperture of extraction
    variance: numpy.ndarray
        3D array the contains the variance cube in the aperture of extraction
    wave: numpy.ndarray
        1D array that contains the wavelength of the datacube that is fed in
    verbose: bool, optional 
        [default False]: if set True shows the fitted light profile
    weights: str 
        if = 'Gaussian',fits a Gaussian profile
           = 'poly', fits an n order polynomial where n is set by porder
           = 'Data', Weights by the raw white light image
    porder: int, optional 
        polynomial order [default 9]


    Returns
    -------
    xspec: XSpectrum1D object [optimal extraction]
        This object contains the wavelength array, the flux array, and the flux error.
    """
    # Sanity Checks
    q=np.isnan(variance)
    variance[q]=0.
    q=variance<0
    variance[q]=0.

    q=np.isnan(flux)
    flux[q]=0.

    img=flux.sum(axis=0)
    img[img<0] =0
    amp_init = flux.max()


    ydim,xdim=img.shape
    stdev_init_x = 0.33 * ydim
    stdev_init_y = 0.33 * xdim
    theta=0

    if weights=='gaussian':
        g_init = models.Gaussian2D(amp_init, 5, 5, stdev_init_x, stdev_init_y,theta=10)
    elif weights =='poly':
        g_init =models.Polynomial2D(degree=porder)#
    else:
        g_init = models.Gaussian2D(amp_init, 5, 5, stdev_init_x, stdev_init_y,theta=10)


    yi, xi = np.indices(img.shape)

    fit_g = fitting.LevMarLSQFitter()
    with warnings.catch_warnings():
        # Ignore model linearity warning from the fitter
        warnings.simplefilter('ignore')
        p = fit_g(g_init, xi, yi, img)

    if verbose == True:
        # Plot the data with the best-fit model
        print('Plotting fitted model profile')
        plt.figure(figsize=(8, 2.5))
        plt.subplot(1, 3, 1)
        plt.imshow(img, origin='lower', interpolation='nearest')
        plt.title("Data")
        plt.subplot(1, 3, 2)
        plt.imshow(p(xi, yi), origin='lower', interpolation='nearest')
        plt.title("Model")
        plt.subplot(1, 3, 3)
        plt.imshow(img - p(xi, yi), origin='lower', interpolation='nearest')
        plt.title("Residual")

    if weights =='Data':
        weights=img
    else:
        weights = p(xi, yi)



    w = wave
    n = len(w)
    fl = np.zeros(n)
    sig = np.zeros(n)

    for wv_ii in range(n):
        # n_spaxels = np.sum(mask)
        weights = weights / np.sum(weights)
        fl[wv_ii] = np.nansum(flux[wv_ii] * weights)  # * n_spaxels
        sig[wv_ii] = np.sqrt(np.nansum(variance[wv_ii] * (weights ** 2)))  # * n_spaxels

    # renormalize
    fl_sum = np.nansum(flux,axis=(1,2))
    norm = np.sum(fl_sum) / np.sum(fl)
    fl = fl * norm
    sig = sig * norm

     # Object
    xspec = XSpectrum1D.from_tuple((wave, fl, sig))

    return xspec
