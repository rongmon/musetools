# importing the required libraries
from __future__ import division
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from musetools import spec as s
from musetools import util as u
from musetools import io as io
import getpass



username=getpass.getuser()

if username == 'bordoloi':
	fitsfile = '/Users/bordoloi/Dropbox/MUSE/LensedArc/RCS0327_16mc_zap.fits'
else:
	fitsfile = '/home/ahmed/astro/data/RCS0327_16mc_zap.fits'



wave, data, var = io.open_muse_cube(fitsfile)

xcen = 121
ycen = 245
zgal= 1.7037455
lam_center=2600.173  # Rest wavelength for one of the iron lines
#lam_center = 2796.351 # Rest Wavelgnth for the first line of Mg II

wave_rest= wave/(1.+zgal)

spec, err_spec = s.extract_square(xcen, ycen, wave_rest, data, var, 5)

vel= u.veldiff(wave_rest,lam_center)
####
'''
minindex = 2100
maxindex = 2800 #for Mg II lines
'''
minindex = 1750
maxindex = 2100
vel = vel[minindex:maxindex]
spec = spec[minindex:maxindex]
err_spec = err_spec[minindex:maxindex]

'''
minw = 7530.
maxw = 7600.  These the wavelength limits for Mg II
'''
minw = 6967.
maxw = 7111.   # These are the wavelength limits for Fe lines
q = np.where(( wave > minw) & (wave < maxw))
vel_fit = np.delete(vel, q)
spec_fit = np.delete(spec, q)

cont = np.poly1d(np.polyfit(vel_fit, spec_fit, 3))
####
fig = plt.figure()
plt.subplot(2,1,1)
plt.step(vel,spec,'-',vel,cont(vel),'--',vel,err_spec,'r-')
plt.xlabel('Velocity')
plt.ylabel('Flux')
plt.xlim([-10500.,8000.])
#plt.xlim([-15000.,15000.])  # for Mg II

plt.subplot(2,1,2)
plt.step(vel,spec/cont(vel))
#plt.ylim([-1,2])
plt.xlabel('Velocity')
plt.ylabel(' Normalized Flux')
plt.xlim([-10500.,8000.])
#plt.xlim([-15000.,15000.])   # for Mg II

#plt.subplot(3,1,3)
#plt.step(vel,1 - (spec/cont(vel)))
#plt.xlabel('Velocity')
#plt.ylabel('1- Normalized Flux')
#plt.xlim([-15000.,15000.])
plt.show()
