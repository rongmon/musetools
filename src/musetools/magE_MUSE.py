from astropy.io import ascii
import numpy as np
import matplotlib.pyplot as plt

def ascii_data(data,zgal=1.7037455,p=False):
    wave = data[0][:]
    #zgal = 1.7037455
    wrest = wave/(1. + zgal)

    flx  = data[1][:]
    flx_er = data[2][:]
    # We now ignore the negative flux values in the spectrum
    q = np.where((flx > 0.)&(flx < 1e-27))
    wrest = wrest[q]
    flx  = flx[q]
    flx_er = flx_er[q]
    #### Getting the good data points only defined by the next two indices
    min_index = 8000
    max_index = 11950
    wrest = wrest[min_index:max_index]
    flx  = flx[min_index:max_index]
    flx_er = flx_er[min_index:max_index]
    '''
    if p==True:
        fig, ax = plt.subplots()
        ax.step(wrest,flx,label='Flux')
        ax.step(wrest,flx_er,label='Error')
        ax.legend(loc=0)
        ax.set_xlabel('Wavelength')
        ax.set_ylabel('Flux')
        ax.set_title('Flux Vs. Wavelength')
        plt.show()
    '''
    return wrest, flx, flx_er

### For the iron lines, I try to find the best window to work with it
def ascii_spec(wrest, flx, flx_er, minindex, maxindex): #,p=False):
    #minFeindex = 2050
    #maxFeindex = 3130
    #minMgindex = 3400
    #maxMgindex = 3950
    wrest = wrest[minindex:maxindex]
    flx  = flx[minindex:maxindex]
    flx_er = flx_er[minindex:maxindex]
    return wrest, flx, flx_er

### Doing the continuum fitting for Fe
def ascii_contn_vel(wrest,flx, flx_er,lam_center,wmin,wmax):#p=False):
    #wmin_Fe = 2580.
    #wmax_Fe = 2640.
    #wmin_Mg = 2580.
    #wmax_Mg = 2640.
    #lam_center_Fe = [2586.650,2600.173,2612.654,2626.451]
    #lam_center_Mg = [2796.351,2803.528]
    f = np.where((wrest > wmin) & (wrest < wmax))
    wrest_fit = np.delete(wrest, f)
    flx_fit = np.delete(flx,f)
    cont = np.poly1d(np.polyfit(wrest_fit, flx_fit, 3))
    continuum = cont(wrest)
    flx_norm = flx/continuum
    flx_er_norm = flx_er/continuum

    import musetools.util as u
    vel = u.veldiff(wrest,lam_center)
    return flx_norm, flx_er_norm, vel

    def cont_flx_plot(wrest,flx,flx_er,continuum,vel,flx_norm,flx_er_norm):
        fig1, ax1 = plt.subplots(2)
        ax1[0].step(wrest,flx,label='Flux')
        ax1[0].step(wrest,continuum,label='Continuum')
        ax1[0].step(wrest,flx_er,label='Error')
        ax1[0].legend(loc=0,fontsize='x-small')
        ax1[0].set_xlabel('Wavelength')
        ax1[0].set_ylabel('Flux')
        ax1[0].set_title('Flux Vs Wavelength')
        ax1[1].step(wrest,flx_norm,label='Relative Flux')
        ax1[1].step(wrest,flx_er_norm,label='Relative Error')
        ax1[1].legend(loc=0,fontsize='x-small')
        ax1[1].set_xlabel('Wavelength')
        ax1[1].set_ylabel('Relative Flux')
        plt.show()

        fig2, ax2 = plt.subplots()
        ax2.step(vel,flx_norm,label='Relative Flux')
        ax2.step(vel,flx_er_norm,label='Relative Error')
        ax2.legend(loc=0)
        ax2.set_xlabel('Velocity')
        ax2.set_ylabel('Relative Flux')
        ax2.set_title('Relative Flux Vs. Velocity')
        plt.show()
        return


import musetools.modeling as m
from lmfit import Model
def model_lmfit(func,flx_norm,flx_er_norm,vel,p0,p=False):
    # for FeII func=m.modelFe
    # for MgII func=m.modelMg
    # p0 is the initial values of the parameters for the model
    # for FeII: p0 = [ v1=0, tau1=0.7, tau3=0.4, c1=1.1, c2=1.7,c3=1.,sigma1=150., sigma2=100.]
    # for MgII: p0 = [v1=0,tau1=0.9,c1 =1.,sigma1=100.]
    if func == m.modelFe:
        gmodel = Model(func)
        #result = gmodel.fit(flx_norm,v=vel, v1=0, tau1=0.7, tau3= 0.4, c1=1.1, c2=1.7,c3=1., sigma1=150, sigma2=100)#,sigma3=100,sigma4=95)
        result = gmodel.fit(flx_norm,v=vel,v1=p0[0],tau1=p0[1],tau3=p0[2],c1=p0[3],c2=p0[4],c3=p0[5],sigma1=p0[6],sigma2=p0[7])
        print(result.fit_report())
    elif func == m.modelMg:
        gmodel = Model(func)
        result = gmodel.fit(flx_norm, v=vel,v1=p0[0],v3=p0[1],tau1=p0[2],tau2=p0[3],c1=p0[4],c2=p0[5],sigma1=p0[6],sigma2=p0[7])
        print(result.fit_report())
    '''
    if p==True:
        fig3, ax3 = plt.subplots()
        ax3.step(vel, flx_norm, label='Relative Flux')
        ax3.plot(vel, result.best_fit, 'y-',label='Model')
        ax3.step(vel, flx_er_norm,'r',label='Relative Error')
        ax3.legend(loc=0,fontsize ='small')
        ax3.set_xlabel('Velocity')
        ax3.set_ylabel('Relative Flux')
        ax3.set_title('Relative Flux Vs Velocity using lmfit modeling')
        plt.show()
    '''
    return result

fitsfile = '/home/ahmed/astro/data/RCS0327_16mc_zap.fits'
cx = 114
cy = 230
from musetools import io as io
from musetools import spec as s
import musetools.util as u
import musetools.modeling as m
from astropy.wcs import WCS
from astropy.io import fits
wave, data, var, header = io.open_muse_cube(fitsfile)
w = WCS(header)
zgal= 1.7037455
wrest = wave/(1.+zgal)
spec, spec_err = s.extract_square(cx, cy, wave, data, var, 5)
minindex = 1750
maxindex = 2100
wave = wave[minindex:maxindex]
spec = spec[minindex:maxindex]
spec_err = spec_err[minindex:maxindex]
wrest= wrest[minindex:maxindex]
minw = 6967.
maxw = 7111.   # These are the wavelength limits for Fe lines
q = np.where(( wave > minw) & (wave < maxw))
wrest_fit = np.delete(wrest, q)
spec_fit = np.delete(spec, q)
cont = np.poly1d(np.polyfit(wrest_fit, spec_fit, 3))  # Defining my polynomial
continuum = cont(wrest)
lam_center = [2586.650,2600.173,2612.654,2626.451]
norm_flx = spec/continuum
norm_flx_err = spec_err/continuum
vel1 = u.veldiff(wrest,lam_center[0])

import musetools.modeling as m
import numpy as np
data1 = ascii.read('/home/ahmed/astro/data/rcs0327-knotE-allres-combwC1.txt')
wrest, flx, flx_er = ascii_data(data1,zgal=1.7037455)
print('For the FeII lines')
wrest_Fe, flx_Fe, flx_er_Fe = ascii_spec(wrest, flx, flx_er, 2050, 3130)
#v1=0, tau1=0.7, tau3= 0.4, c1=1.1, c2=1.7,c3=1., sigma1=150, sigma2=100
p0_Fe = [ 0., 0.7, 0.4, 1.1, 1.7, 1., 150., 100.]
flx_norm_Fe, flx_er_norm_Fe, vel_Fe = ascii_contn_vel(wrest_Fe,flx_Fe, flx_er_Fe,2586.650,2580.,2640.)


def model_curve_fit(func, vel, flx_norm, p0,flx_er_norm):
    # p0 is the initial parameters
    from scipy.optimize import curve_fit
    popt, pcov = curve_fit(func, vel, flx_norm,p0,sigma=flx_er_norm)
    print(popt)
    print(pcov)
    return popt, pcov
p0 = [0.,0.7,0.4,1.1,1.7,1.,150.,100.]
popt_muse, pcov_muse = model_curve_fit(m.modelFe, vel1, norm_flx,p0,norm_flx_err )
#m.modelFe,flx_norm_Fe,flx_er_norm_Fe,vel_Fe,p0_Fe,p=True
popt_magE, pcov_magE = model_curve_fit(m.modelFe, vel_Fe, flx_norm_Fe, p0, flx_er_norm_Fe)
fig, ax = plt.subplots(2)
ax[0].step(vel1, norm_flx, label='Normalized Flux Muse')
ax[0].plot(vel1, m.modelFe(vel1,*popt_muse), 'y-',label='Model MUSE')
#ax[0].step(vel1, norm_flx_err,'r',label='Error')
#ax[0].legend(loc=0)
ax[0].set_title('Normalized Flux Vs Velocity for '+str(cx)+' & '+str(cy)+'')
ax[0].set_xlabel('Velocity')
ax[0].set_ylabel('Normalized Flux')
ax[0].set_xlim([-2100,6200])
#ax[0].legend(loc=0,font=5)
ax[0].step(vel_Fe, flx_norm_Fe, label='Normalized Flux MagE')
ax[1].plot(vel_Fe, m.modelFe(vel_Fe,*popt_magE),'y-',label='Model MagE')
ax[1].set_xlabel('Velocity')
ax[1].set_ylabel('Normalized Flux')
ax[1].set_xlim([-2100,6200])
plt.show()
