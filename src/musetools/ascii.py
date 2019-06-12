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
