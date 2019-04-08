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
