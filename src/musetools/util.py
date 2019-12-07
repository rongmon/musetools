from __future__ import print_function, absolute_import, division, unicode_literals
import numpy as np


def veldiff(wave,wave_center):
    z = (wave / wave_center) - 1.
    c = 299792.458  # Speed of light in km/s
    beta = ((z + 1.)**2. - 1.)/((z+1.)**2. + 1.)
    del_v = beta * c
    return del_v


import numpy as np
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
    # equivalent width
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
    q1 = np.where((wave > winmin) & (wave < winmax))
    wave = wave[q1]
    flx  = flx[q1]
    flx_er = flx_er[q1]
    ### Doing the continuum fitting
    q2 = np.where((wave > minw) & (wave < maxw))
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
    image=np.zeros((349,352,len(minwave)))

    for i in range(len(minwave)):
        factor = 10.**(-20.)*(maxwave[i]-minwave[i]) / (0.2)**2.
        q_i = np.where(( wave > minwave[i]) & (wave < maxwave[i]))# Defining the chosen wavelength interval
        image[:,:,i] = (np.sum(flux_data[q_i,:,:], axis = 1))*factor              # We now sum the wavelength within the given interval

    image_SB=np.sum(image,axis=2)

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
