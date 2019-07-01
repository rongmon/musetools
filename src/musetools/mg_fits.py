### This file is used to extract the individual sepctra as fits files
from astropy.io import fits
from astropy.io import ascii
import numpy as np
import matplotlib.pyplot as plt
from musetools import io as io
from musetools import spec as s
import musetools.util as u
import musetools.modeling as m
from astropy.wcs import WCS
from lmfit import Model
import pdb

import getpass

username=getpass.getuser()

if username == 'bordoloi':
	fitsfile = '/Users/bordoloi/Dropbox/MUSE/LensedArc/RCS0327_16mc_zap.fits'
else:
	fitsfile = '/home/ahmed/astro/data/RCS0327_16mc_zap.fits'


'''
wave, data, var, header = io.open_muse_cube(fitsfile)
w = WCS(header)
zgal= 1.7037455
wrest = wave/(1.+zgal)
'''
#xcen = 121
#ycen = 245

xcen = [114,114,115,118,121,124,127,133,137,141,148,153,160,166,170,177,185,191,198,203,208,213,220,225,231,238,244,246,244,240,238,242]
ycen = [226,233,237,241,244,248,252,257,260,264,269,271,274,274,274,274,274,272,271,270,268,266,263,259,255,249,244,240,237,234,228,224]

for cx, cy in zip(xcen, ycen):
    data = ascii.read('/home/ahmed/astro/spectra/spectrum_mg_'+str(cx)+'_'+str(cy)+'.dat')
    # wave wrest flx flx_er continuum flx_norm flx_er_norm
    wave        = data[0][:]
    wrest       = data[1][:]
    flx         = data[2][:]
    flx_er      = data[3][:]
    continuum   = data[4][:]
    flx_norm    = data[5][:]
    flx_er_norm = data[6][:]
    lam_center = [2796.351,2803.528]
    vel1 = u.veldiff(wrest,lam_center[0])
    from scipy.optimize import curve_fit
    popt, pcov = curve_fit(m.modelMg, vel1, flx_norm,[0.,0.7,0.4,1.1,1.7,1.,150.,100.])
    perr = np.sqrt(np.diag(pcov))
    v1 = 	 popt[0]; 	v1_er =   	perr[0]
    tau1 = 	 popt[1]; 	tau1_er = 	perr[1]
    tau3 = 	 popt[2]; 	tau3_er = 	perr[2]
    c1 =     popt[3]; 	c1_er =   	perr[3]
    c2 = 	 popt[4]; 	c2_er = 	perr[4]
    c3 = 	 popt[5]; 	c3_er = 	perr[5]
    sigma1 = popt[6]; 	sigma1 = 	perr[6]
    sigma2 = popt[7]; 	sigma2 = 	perr[7]
    '''
    gmodel = Model(m.modelFe)
    result = gmodel.fit(norm_flx,v=vel1, v1=0, tau1=0.7, tau3= 0.4, c1=1.1, c2=1.7,c3=1., sigma1=150, sigma2=100)#,sigma3=100,sigma4=95)
    print(result.fit_report())
    '''
    fig, ax = plt.subplots()
    ax.step(vel1, flx_norm, label='Normalized Flux')
    ax.plot(vel1, m.modelMg(vel1,*popt), 'y-',label='Model')
    ax.step(vel1, flx_er_norm,'r',label='Error')
    ax.legend(loc=0)
    ax.set_title('Normalized Flux Vs Velocity for '+str(cx)+' & '+str(cy)+'')
    ax.set_xlabel('Velocity')
    ax.set_ylabel('Normalized Flux')
    ax.set_xlim([-2100,6200])
    plt.show()
    fig.savefig('/home/ahmed/astro/figures/fitting/model_fit_Mg_abs'+str(cx)+'_'+str(cy)+'.pdf')
    plt.close(fig)
