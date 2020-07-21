from __future__ import print_function, absolute_import, division, unicode_literals
import numpy as np
from astropy.convolution import Gaussian2DKernel, convolve
from copy import deepcopy
import math as mt
from scipy.optimize import curve_fit
import corner
from multiprocessing import Pool
import emcee
import matplotlib.pyplot as plt

from scipy import interpolate


def veldiff(wave,wave_center):
    z = (wave / wave_center) - 1.
    c = 299792.458  # Speed of light in km/s
    beta = ((z + 1.)**2. - 1.)/((z+1.)**2. + 1.)
    del_v = beta * c
    return del_v


import numpy as np



def spectral_res(obs_lambda):
    l_obs = np.asarray([ 4650., 5000., 5500., 6000., 6500., 7000., 7500., 8000., 8500., 9000., 9350.])
    R     = np.asarray([ 1609., 1750., 1978., 2227., 2484., 2737., 2975., 3183., 3350., 3465., 3506.])
    R_err = np.asarray([    6.,    4.,    6.,    6.,    5.,    4.,    4.,    4.,    4.,    5.,   10.])
    l_ex  = np.linspace(4650., 9350., 5000)
    spl = 299792.458 # speed of light km/s
    V = (1/R)*spl

    f = interpolate.interp1d(l_obs, R, fill_value="extrapolate")

    mark = 5

    R_res = f(l_ex)

    V_res = (1 / R_res) * spl

    specR_res = f(obs_lambda)
    vel_res = (1 / specR_res) * spl
    lam_res = (obs_lambda / specR_res)
    return lam_res, vel_res



def compute_EW(lam,flx,wrest,lmts,flx_err,plot=False,**kwargs):
    #------------------------------------------------------------------------------------------
    #   Function to compute the equivalent width within a given velocity limits lmts=[vmin,vmax]
    #           [Only good for high resolution spectra]
    #  Caveats:- Not automated, must not include other absorption troughs within the velocity range.
    #
    #   Input:-
    #           lam         :- Observed Wavelength vector (units of Angstrom)
    #           flx         :- flux vector ( same length as wavelgnth vector, preferably continuum normalized)
    #           wrest       :- rest frame wavelength of the line [used to make velcity cuts]
    #           lmts        :- [vmin,vmax], the velocity window within which equivalent width is computed.
    #           flx_err     :- error spectrum [same length as the flux vector]
    #
    #   OPTIONAL :-
    #           f0=f0       :- fvalue of the transition
    #           zabs=zabs   :- absorber redshift
    #           plot        :- plot keyword, default = no plots plot=0
    #                           plot=1 or anything else will plot the corresponding spectrum
    #                            and the apparent optical depth of absorption.
    #
    #
    #
    # Output:-  In a Python dictionary format
    #           output['ew_tot']      :- rest frame equivalent width of the absorpiton system [Angstrom]
    #           output['err_ew_tot']  :- error on rest fram equivalent width
    #           output['col']         :- AOD column denisty
    #           output['colerr']      :- 1 sigma error on AOD column density
    #           output['n']           :- AOD column density as a function of velocity
    #           output['Tau_a']       :- AOD as a function of velocity
    #
    #
    #   Written :- Rongmon Bordoloi                             2nd November 2016
    #-  I translated this from my matlab code compute_EW.m, which in turn is from Chris Thom's eqwrange.pro.
    #   This was tested with COS-Halos/Dwarfs data.
    #   Edit:  RB July 5 2017. Output is a dictionary. Edited minor dictionary arrangement
    #------------------------------------------------------------------------------------------
    defnorm=1.0;
    spl=2.9979e5;  #speed of light
    if 'zabs' in kwargs:
        zabs=kwargs['zabs']
    else:
        zabs=0.

    if 'sat_limit' in kwargs:
        sat_limit=kwargs['sat_limit']
    else:
        sat_limit=0.10 #  Limit for saturation (COS specific). Set to same as fluxcut for now. WHAT SHOULD THIS BE???
    vel = (lam-wrest*(1.0 + zabs))*spl/(wrest*(1.0 + zabs));
    lambda_r=lam/(1.+zabs);



    norm=defnorm

    norm_flx=flx/norm;
    flx_err=flx_err/norm;
    sq=np.isnan(norm_flx);
    tmp_flx=flx_err[sq]
    norm_flx[sq]=tmp_flx
    #clip the spectrum. If the flux is less than 0+N*sigma, then we're saturated. Clip the flux array(to avoid inifinite optical depth) and set the saturated flag
    q=np.where(norm_flx<=sat_limit);
    tmp_flx=flx_err[q]
    norm_flx[q]=tmp_flx
    q=np.where(norm_flx<=0.);
    tmp_flx=flx_err[q]+0.01
    norm_flx[q]=tmp_flx;


    del_lam_j=np.diff(lambda_r);
    del_lam_j=np.append([del_lam_j[0]],del_lam_j);


    pix = np.where( (vel >= lmts[0]) & (vel <= lmts[1]));
    Dj=1.-norm_flx

    # Equivalent Width Per Pixel
    ew=del_lam_j[pix]*Dj[pix];


    sig_dj_sq=(flx_err)**2.;
    err_ew=del_lam_j[pix]*np.sqrt(sig_dj_sq[pix]);
    err_ew_tot=np.sqrt(np.sum(err_ew**2.));
    ew_tot=np.sum(ew);
    print('W_lambda = ' + np.str('%.3f' % ew_tot) + ' +/- ' + np.str('%.3f' % err_ew_tot)  +'  \AA   over [' + np.str('%.1f' % np.round(lmts[0]))+' to ' +np.str('%.1f' % np.round(lmts[1])) + ']  km/s')
    output={}
    output["ew_tot"]=ew_tot
    output["err_ew_tot"]=err_ew_tot


    if 'f0' in kwargs:
        f0=kwargs['f0']
        #compute apparent optical depth
        Tau_a =np.log(1./norm_flx);

        # REMEMBER WE ARE SWITCHING TO VELOCITY HERE
        del_vel_j=np.diff(vel);
        del_vel_j=np.append([del_vel_j[0]],del_vel_j)

        # Column density per pixel as a function of velocity
        nv = Tau_a/((2.654e-15)*f0*lambda_r);# in units cm^-2 / (km s^-1), SS91
        n = nv* del_vel_j# column density per bin obtained by multiplying differential Nv by bin width
        tauerr = flx_err/norm_flx;
        nerr = (tauerr/((2.654e-15)*f0*lambda_r))*del_vel_j;
        col = np.sum(n[pix]);
        colerr = np.sum((nerr[pix])**2.)**0.5;
        print('Direct N = ' + np.str('%.3f' % np.log10(col))  +' +/- ' + np.str('%.3f' % (np.log10(col+colerr) - np.log10(col))) + ' cm^-2')
        output["col"]=col
        output["colerr"]=colerr
        output["Tau_a"]=Tau_a





    # If plot keyword is  set start plotting
    if plot is not False:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax1=fig.add_subplot(211)
        ax1.step(vel,norm_flx)
        ax1.step(vel,flx_err,color='r')
        #plt.xlim([lmts[0]-2500,lmts[1]+2500])
        plt.xlim([-600,600])
        plt.ylim([-0.02,1.8])
        ax1.plot([-2500,2500],[0,0],'k:')
        ax1.plot([-2500,2500],[1,1],'k:')
        plt.plot([lmts[0],lmts[0]],[1.5,1.5],'r+',markersize=15)
        plt.plot([lmts[1],lmts[1]],[1.5,1.5],'r+',markersize=15)
        plt.title(r' $W_{rest}$= ' + np.str('%.3f' % ew_tot) + ' $\pm$ ' + np.str('%.3f' % err_ew_tot) + ' $\AA$')
        ax1.set_xlabel('vel [km/s]')

        ax2=fig.add_subplot(212)
        ax2.step(vel,n)
        ax2.set_xlabel('vel [km/s]')
        ax2.plot([-2500,2500],[0,0],'k:')
        #plt.xlim([lmts[0]-2500,lmts[1]+2500])
        plt.xlim([-600,600])
        plt.show()


    return output

def airtovac(wave):
    """ Convert air-based wavelengths to vacuum

    Parameters:
    ----------
    wave: ndarray
      Wavelengths

    Returns:
    ----------
    wavelenght: ndarray
      Wavelength array corrected to vacuum wavelengths
    """
    # Assume AA
    wavelength = wave

    # Standard conversion format
    sigma_sq = (1.e4/wavelength)**2. #wavenumber squared
    factor = 1 + (5.792105e-2/(238.0185-sigma_sq)) + (1.67918e-3/(57.362-sigma_sq))
    factor = factor*(wavelength>=2000.) + 1.*(wavelength<2000.) #only modify above 2000A

    # Convert
    wavelength = wavelength*factor

    return wavelength

def compute_abs(wrest,flx_norm, lam_center, tau, f0, sig, vmin,vmax):
    #F = 1. - flx_norm
    vel = veldiff(wrest,lam_center)
    l = np.where((vel < vmax) & (vel > vmin))
    # equivalent widthfrom scipy.optimize import curve_fit

    ew = np.trapz(1.-flx_norm[l],x=wrest[l])
    # Calculating the average velocity
    norm = np.trapz(1. - flx_norm[l],x=vel[l])
    v_avg = (1/norm)*np.trapz(vel[l]*(1. - flx_norm[l]),x=vel[l])
    # Calculating the column density
    b_D = (np.sqrt(2)* 2.35482 * sig)
    N = (tau * b_D) / ((1.497* 10**(-15)) * lam_center * f0)
    return ew, v_avg, np.log10(N)

def compute_ems(wrest, flx_norm, lam_center, vmin, vmax):
    vel =  veldiff(wrest, lam_center)
    l = np.where((vel < vmax) & (vel > vmin))
    # Equivalent Width
    ew = np.trapz(1. - flx_norm[l],x=wrest[l])
    # Average Velocity
    #norm = np.trapz(flx_norm[l] -1.,x=vel[l])
    #v_avg = (1/norm)*np.trapz(vel[l]*(flx_norm[l] - 1.),x=vel[l])
    return ew#, v_avg

def cont_func(wave, flx, flx_er, winmin, winmax, minw, maxw):
    # wave: wavelength
    # flx: flux
    # flx_er: error in flux
    # winmin: the start wavelength for the total shown window around the line
    # winmax: the end wavelength for the total shown window around the line
    # minw: the start wavelength of the narrow window around the line
    # maxw: the end wavelegnth of the narrow window around the line
    q1 = np.where((wave >= winmin) & (wave <= winmax))
    wave = wave[q1]
    flx  = flx[q1]
    flx_er = flx_er[q1]
    ### Doing the continuum fitting
    q2 = np.where((wave >= minw) & (wave <= maxw))
    wave_fit = np.delete(wave, q2)
    flx_fit = np.delete(flx, q2)
    cont = np.poly1d(np.polyfit(wave_fit, flx_fit, 3))
    continuum = cont(wave)
    flx_norm = flx/continuum
    flx_er_norm = flx_er/continuum
    return wave, flx, flx_er, continuum, flx_norm, flx_er_norm


def multi_band(minwave, maxwave, wave, flux_data):
    '''
    minwave:   array of minimum wavelength of the narrow bands
    maxwave:   array of maximum wavelength of the narrow bands
    wave:      The wavelength interval from the data cube
    flux_data: The 2-D flux data from the data cube

    The output of this function is a wavelength narrow band image of the LensedArc
    '''
    q = []
    sh = flux_data.shape
    image = np.zeros((sh[1],sh[2], sh[0]))#np.zeros((349,352,len(minwave)))

    for i in range(len(minwave)):
        factor = 10.**(-20.)*(maxwave[i]-minwave[i]) / (0.2)**2.
        q_i = np.where(( wave > minwave[i]) & (wave < maxwave[i]))# Defining the chosen wavelength interval
        image[:,:,i] = (np.sum(flux_data[q_i,:,:], axis = 1))*factor              # We now sum the wavelength within the given interval

    image_SB=np.sum(image,axis=2)

    return image_SB



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
    q_nan = np.isnan(image_SB)
    q_neg = np.where(image_SB < 0.0)
    image_SB[q_nan] = 0.0
    image_SB[q_neg] = 0.0
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
    q_nan = np.isnan(image_SB)
    q_neg = np.where(image_SB < 0.0)
    image_SB[q_nan] = 0.0
    image_SB[q_neg] = 0.0
    # We will plot the image now using log10 scaling
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    d = ax.imshow(np.log10(np.abs(image[0,:,:])), cmap = plt.get_cmap('viridis'), origin='lower')
    plt.show()
    '''
    return image_SB


def convolve_image(image,stdev=1.):
   # We smooth with a Gaussian kernel with stddev=2
   # It is a 9x9 array
   kernel = Gaussian2DKernel(x_stddev=stdev)
   # create a "fixed" image with NaNs replaced by interpolated values
   # astropy's convolution replaces the NaN pixels with a kernel-weighted
   # interpolation from their neighbors
   astropy_conv = convolve(image, kernel)
   mean_val=(np.mean(astropy_conv))
   std_val=(np.std(astropy_conv))
   sig_threshold=3.

   qq=np.where((astropy_conv < sig_threshold*std_val))
   qq_complement=np.where((astropy_conv >= sig_threshold*std_val))

   ROI=deepcopy(astropy_conv)
   ROI[qq]= 0.

   #Now create a mask

   Mask=deepcopy(astropy_conv)
   Mask[qq_complement]=1.
   Mask[qq]=0.

   final_image= astropy_conv*Mask

   return final_image,astropy_conv


def lnlike(theta, model, x, y, y_err):
    l = -0.5 * (np.sum( ((y - model(x,*theta))/y_err) **2. ))
    return l


def lnprior(theta,lower,upper):
    # theta: is the array that contain my parameters
    # upper: upper bounds on my parameters
    # lower: lower bounds on my parameters
    for i in range(len(theta)):
        if ((theta[i] < lower[i]) or (theta[i] > upper[i]) ):
            return -np.inf
            break

    return 0.0

def lnprob(theta, model, x, y, y_err, lower, upper):
    lp = lnprior(theta, lower, upper)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, model, x, y, y_err)





def emcee_fit(model, x, y, y_err, p0, p_low, p_up, p_names, names, samples_dir, walkers_dir, corner_dir, plot=False, write=False):
    """
    model: given model to fit the data
    x:     the x data points (Independant variable)
    y:     the y data points (Dependant varilable)
    y_err: the error on the y data points
    p0:    an array that have initial vlues for the model parameters
    p_low: lower limits on the model parameters (lower bounds of the parameter space)
    p_up:  upper limits on the model Parameters (uppper bounds of the parameter space)
    p_names: an arrays that contains the names of each parameter to be used in the plots
    names:   an array that will have strings to mark the names of the output files
    samples_dir: the directory where you want to save your output parameters values
    walkers_dir: the directory where you want to save the walkers figures
    corner_dir : the directory where you want to save the corner figures
    plot: it can have two logical values either True or False to make or not make plots  (default: False)
    write: it can have two logical values either True or False to write or not write your arrays and plots (default: False)
    """
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    mpl.rcParams.update(mpl.rcParamsDefault)
    mpl.rc('text',usetex=True)
    XSMALL_SIZE = 10
    SMALL_SIZE = 12
    MEDIUM_SIZE = 14
    LARGE_SIZE = 16
    XLARGE_SIZE = 18
    XXLARGE_SIZE = 24

    plt.rc('font',size=SMALL_SIZE)
    plt.rc('axes',titlesize=XLARGE_SIZE)
    plt.rc('axes',labelsize=LARGE_SIZE)
    plt.rc('axes',labelweight=700)
    plt.rc('axes',titleweight=700)
    plt.rc('xtick',labelsize=MEDIUM_SIZE)
    plt.rc('ytick',labelsize=MEDIUM_SIZE)
    plt.rc('legend',fontsize=XSMALL_SIZE)
    plt.rc('figure',titlesize=XXLARGE_SIZE)

    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = 'Times'
    mpl.rcParams['font.monospace'] = 'Ubuntu mono'

    x = np.asarray(x);    y = np.asarray(y);          y_err = np.asarray(y_err)
    p0 = np.asarray(p0);  p_low = np.asarray(p_low);  p_up = np.asarray(p_up)

    popt, pcov = curve_fit(model, x, y, p0, sigma= y_err, bounds=(p_low, p_up))
    perr = np.sqrt(np.diag(pcov))

    popt = np.around(popt, decimals=3)
    ndim, nwalkers = int(len(p0)), 50

    pos = [popt + 1e-5* np.random.randn(ndim) for i in range(nwalkers)]

    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool, args=(model, x, y, y_err[:], p_low, p_up))
        n = 1500
        sampler.run_mcmc(pos, n, progress=True)
    samples = sampler.chain[:,int(0.2*n):,:].reshape((-1,ndim))
    if write == True:
        np.save(samples_dir+'samples'+names[0]+'_'+names[1]+'.npy',samples)
    if plot == True:
        fig_height = int(len(p0) * 4)
        fig_width  = 12
        plt.close()
        fig, ax = plt.subplots(ndim)
        fig.set_figheight(fig_height)
        fig.set_figwidth(fig_width)
        fig.suptitle('Walkers')
        for i in range(ndim):
            ax[i].plot(sampler.chain[:,:,i].T,'k', alpha=0.2)
            ax[i].set_ylabel(p_names[i])
            ax[i].set_xlabel('Steps Number')
        fig.tight_layout()
        fig.subplots_adjust(top=0.94)
        #plt.show()
        fig.savefig(walkers_dir+'walkers'+names[0]+'_'+names[1]+'.pdf', overwrite=True, bbox_inches='tight', dpi=300)
        fig.savefig(walkers_dir+'walkers'+names[0]+'_'+names[1]+'.png', overwrite=True, bbox_inches='tight', dpi=100)
        #fig.savefig(walkers_dir+'walkers'+names[0]+'_'+names[1]+'_600.png', overwrite=True, bbox_inches='tight', dpi=600)
        plt.close()

    flat_samples = sampler.get_chain(discard=int(0.2*n), flat=True)
    p_opt = []; up_sig = []; lw_sig = [];
    for i in range(ndim):
        mcmc = np.percentile(flat_samples[:,i],[16,50,84])
        q = np.diff(mcmc)
        p_opt.append(mcmc[1])
        up_sig.append(q[1])
        lw_sig.append(q[0])
    if plot == True:
        plt.close()
        figure = corner.corner(samples, labels = p_names, quantiles=[0.16, 0.5, 0.84],show_titles=True, title_kwargs={"fontsize": 12})
        # Loop over the diagonal
        axes = np.array(figure.axes).reshape((ndim, ndim))
        for i in range(ndim):
            ax = axes[i, i]
            ax.axvline(p_opt[i], color="b",alpha=0.5)
        for yi in range(ndim):
            for xi in range(yi):
                ax = axes[yi, xi]
                ax.axvline(p_opt[xi], color="b",alpha=0.5)
                ax.axhline(p_opt[yi], color="b",alpha=0.5)
                ax.plot(p_opt[xi], p_opt[yi], "sb")
        #plt.show()
        figure.savefig(corner_dir+'corner'+names[0]+'_'+names[1]+'.pdf',overwrite=True,bbox_inches='tight',dpi=300)
        figure.savefig(corner_dir+'corner'+names[0]+'_'+names[1]+'.png',overwrite=True,bbox_inches='tight',dpi=100)
        #figure.savefig(corner_dir+'corner'+names[0]+'_'+names[1]+'_600.png',overwrite=True,bbox_inches='tight',dpi=600)
        plt.close()


    return p_opt, up_sig, lw_sig






###    Compute the angular diameter distance
def ang_DA(w_m, w_l, z):
    """
    Constants
    h  : Dimensionless Parameter
    H0 : Current Value of Hubble Parameter measured in km s^{−1} Mpc^{−1}
    DH : Hubble Distance in meters    # DH = c/H0 = 3000 * (1/h) # in Mpc
    Inputs and variables:
    w_m : Matter Density Parameter       (dimensionless)
    w_l : Dark Energy Density Parameter  (dimensionless)
    w_k : Curvature Parameter            (dimensionless)
    E_z : Defined function of the redshift
    z   : is the redshift of the object
    Dc  : line of sight comoving distance
    Dm  : is the transverse comoving distance
    D_A : is the angular diameter distance of the object (will be given in meters)
    """

    # Defining our constants
    h = 0.7                        # Dimensionless Parameter
    H0 = 0.7 * 100.                # Current Value of Hubble Parameter measured in km s^{−1} Mpc^{−1}
    DH = 9.26 * 10**(25) * (1/h)   # Hubble Distance in meters    # DH = c/H0 = 3000 * (1/h) # in Mpc


    w_k = 1. - w_m - w_l


    zd = np.linspace(0., z, 1000)
    E_z = np.sqrt( w_m * ((1. + zd)**3.) + w_k * ((1. + zd)**2)  + w_l)

    Dc = DH * np.trapz( (1. / E_z) , x=zd )     # sight of line comoving distance

    # Defining the trnasverse comoving distance
    #print(w_k)
    if w_k > 0.0:
        Dm = (DH / np.sqrt(w_k) * np.sinh( (np.sqrt(w_k) * Dc )/ DH ) )   # for w_k > 0.0
    elif w_k == 0.0:
        Dm = Dc     # for w_k = 0.0
    else:
        Dm = (DH / np.sqrt( np.abs(w_k)) * np.sinh( (np.sqrt(np.abs(w_k)) * Dc )/ DH ) )  # for w_k < 0.0


    D_A = Dm / (1. + z)

    return D_A#/DH#, Dm, Dc



def ang_sep_D(cen, xc, yc):
    d  = np.arccos(np.sin(np.radians(cen[1])) * np.sin(np.radians(yc)) + np.cos(np.radians(cen[1])) * np.cos(np.radians(yc)) * np.cos(np.radians( cen[0] - xc )))
    # d is given in radians
    return d

