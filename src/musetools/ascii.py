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
    if p==True:
        fig, ax = plt.subplots()
        ax.step(wrest,flx,label='Flux')
        ax.step(wrest,flx_er,label='Error')
        ax.legend(loc=0)
        ax.set_xlabel('Wavelength')
        ax.set_ylabel('Flux')
        ax.set_title('Flux Vs. Wavelength')
        plt.show()
    return wrest, flx, flx_er

### For the iron lines, I try to find the best window to work with it
def ascii_spec(wrest, flx, flx_er, minindex, maxindex,p=False):
    #minFeindex = 2050
    #maxFeindex = 3130
    #minMgindex = 3400
    #maxMgindex = 3950
    wrest = wrest[minindex:maxindex]
    flx  = flx[minindex:maxindex]
    flx_er = flx_er[minindex:maxindex]
    if p==True:
        fig1, ax1 = plt.subplots()
        ax1.step(wrest,flx,label='Flux')
        ax1.step(wrest,flx_er,label='Error')
        ax1.legend(loc=0)
        ax1.set_xlabel('Wavelength')
        ax1.set_ylabel('Flux')
        ax1.set_title('Flux Vs. Wavelength')
        plt.show()
    return wrest, flx, flx_er

### Doing the continuum fitting for Fe
def ascii_contn_vel(wrest,flx, flx_er,lam_center,wmin,wmax,p=False):
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
    if p==True:
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
    return flx_norm, flx_er_norm, vel

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
        report = result.fit_report()
    elif func == m.modelMg:
        gmodel = Model(func)
        result = gmodel.fit(flx_norm, v=vel,v1=p0[0],v3=p0[1],tau1=p0[2],tau2=p0[3],c1=p0[4],c2=p0[5],sigma1=p0[6],sigma2=p0[7])
        report = result.fit_report()

    if p==True:
        fig3, ax3 = plt.subplots()
        ax3.step(vel, flx_norm, label='Relative Flux')
        ax3.step(vel, result.best_fit, 'y-',label='Model')
        ax3.step(vel, flx_er_norm,'r',label='Relative Error')
        ax3.legend(loc=0,fontsize ='small')
        ax3.set_xlabel('Velocity')
        ax3.set_ylabel('Relative Flux')
        ax3.set_title('Relative Flux Vs Velocity using lmfit modeling')
        plt.show()
    return report

import musetools.modeling as m
import numpy as np
data1 = ascii.read('/home/ahmed/astro/data/rcs0327-knotE-allres-combwC1.txt')
wrest, flx, flx_er = ascii_data(data1,zgal=1.7037455)
print('For the FeII lines')
wrest_Fe, flx_Fe, flx_er_Fe = ascii_spec(wrest, flx, flx_er, 2050, 3130)
#v1=0, tau1=0.7, tau3= 0.4, c1=1.1, c2=1.7,c3=1., sigma1=150, sigma2=100
p0_Fe = [ 0., 0.7, 0.4, 1.1, 1.7, 1., 150., 100.]
flx_norm_Fe, flx_er_norm_Fe, vel_Fe = ascii_contn_vel(wrest_Fe,flx_Fe, flx_er_Fe,2586.650,2580.,2640.)
report_Fe = model_lmfit(m.modelFe,flx_norm_Fe,flx_er_norm_Fe,vel_Fe,p0_Fe,p=True)
print(report_Fe)

print('For the MgII lines')
p0_Mg = [ 0., 130., 0.9, 1., 0.8, 0.7, 150., 80.]
wrestMg, flx_Mg, flx_er_Mg = ascii_spec(wrest, flx, flx_er, 3400, 3950,p=True)
flx_norm_Mg, flx_er_norm_Mg, vel_Mg= ascii_contn_vel(wrestMg,flx_Mg,flx_er_Mg,2796.351,2780.,2810.,p=True)
report_Mg = model_lmfit(m.modelMg,flx_norm_Mg, flx_er_norm_Mg, vel_Mg,p0_Mg,p=True)
print(report_Mg)

















'''
from astropy.io import ascii
import numpy as np
import matplotlib.pyplot as plt

data = ascii.read('/home/ahmed/astro/data/rcs0327-knotE-allres-combwC1.txt')

wave = data[0][:]
zgal = 1.7037455
wrest = wave/(1. + zgal)

flux  = data[1][:]
noise = data[2][:]
# We now ignore the negative flux values in the spectrum
q = np.where(flux > 0)
wrest = wrest[q]
flux  = flux[q]
noise = noise[q]
#### Getting the good data points only defined by the next two indices
min_index = 8000
max_index = 11950

wrest = wrest[min_index:max_index]
flux  = flux[min_index:max_index]
noise = noise[min_index:max_index]
### For the iron lines, I try to find the best window to work with it
minFeindex = 2050
maxFeindex = 3130
wrestFe = wrest[minFeindex:maxFeindex]
fluxFe  = flux[minFeindex:maxFeindex]
noiseFe = noise[minFeindex:maxFeindex]

### Doing the continuum fitting for Fe
wmin = 2580.
wmax = 2640.
f = np.where((wrestFe > wmin) & (wrestFe < wmax))
wrestFe_fit = np.delete(wrestFe, f)
fluxFe_fit = np.delete(fluxFe,f)
cont = np.poly1d(np.polyfit(wrestFe_fit, fluxFe_fit, 3))
continuum = cont(wrestFe)
fluxFe_norm = fluxFe/continuum
noiseFe_norm = noiseFe/continuum

fig1, ax1 = plt.subplots(2)
ax1[0].step(wrestFe,fluxFe)
ax1[0].step(wrestFe,continuum)
ax1[0].step(wrestFe,noiseFe)
ax1[0].set_xlabel('Wavelength')
ax1[0].set_ylabel('Flux')

ax1[1].step(wrestFe,fluxFe_norm)
ax1[1].step(wrestFe,noiseFe_norm)
ax1[1].set_xlabel('Wavelength')
ax1[1].set_ylabel('Relative Flux')
plt.show()

import musetools.util as u
lam_center = [2586.650,2600.173,2612.654,2626.451]
vel = u.veldiff(wrestFe,lam_center[0])


import musetools.modeling as m
from lmfit import Model
gmodel = Model(m.modelFe)
result = gmodel.fit(fluxFe_norm, v = vel, v1=0, tau1=0.7, tau3=0.4, c1=1.1, c2=1.7,c3=1.,sigma1=150., sigma2=100.)
print(result.fit_report())
fig3, ax3 = plt.subplots()
ax3.step(vel, fluxFe_norm, label='Relative Flux')
ax3.plot(vel, result.best_fit, 'y-',label='Model')
ax3.step(vel, noiseFe_norm,'r',label='Relative Error')
ax3.set_title('Modeling FeII lines')
ax3.set_xlabel('Velocity')
ax3.set_ylabel('Relative Flux')
plt.show()
'''
