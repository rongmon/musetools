
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from musetools import io as io
from musetools import spec as s

import getpass


username=getpass.getuser()

if username == 'bordoloi':
	fitsfile = '/Users/bordoloi/Dropbox/MUSE/LensedArc/RCS0327_16mc_zap.fits'
else:
	fitsfile = '/home/ahmed/astro/data/RCS0327_16mc_zap.fits'



wave, data, var = io.open_muse_cube(fitsfile)

xcen = 121
ycen = 245
spec, err_spec = s.extract_square(xcen, ycen, wave, data, var, 5)
'''
minindex = 2100
maxindex = 2800 #for Mg II lines
'''
minindex = 1750
maxindex = 2100
wave = wave[minindex:maxindex]
spec = spec[minindex:maxindex]
err_spec = err_spec[minindex:maxindex]

fig = plt.figure()
#plt.subplot(2,1,1)
#plt.plot(wave,spec) Uncomment these two lines and modify the indices of the next
# subplots to view the total flux
'''
minw = 7530.
maxw = 7600.  These the wavelength limits for Mg II
'''
minw = 6967.
maxw = 7111.   # These are the wavelength limits for Fe lines
q = np.where(( wave > minw) & (wave < maxw))
wave_fit = np.delete(wave, q)
spec_fit = np.delete(spec, q)

cont = np.poly1d(np.polyfit(wave_fit, spec_fit, 3))
'''
The previous line does the polynomial fitting for the continuum
'''

#plt.plot(wave[2100:2600],spec[2100:2600], '-',wave[int_index:fin_index],p(wave[int_index:fin_index]),'--')
plt.subplot(2,1,1)
plt.step(wave,spec,'-',wave,cont(wave),'--',wave,err_spec,'r-')
plt.xlabel('Wavelength')
plt.ylabel('Flux')


plt.subplot(2,1,2)
plt.step(wave,spec/cont(wave))
#plt.ylim([-1,2])
plt.xlabel('Wavelength')
plt.ylabel(' Normalized Flux')


#plt.subplot(3,1,3)
#plt.plot(wave,1 - (spec/cont(wave)))
#plt.xlabel('Wavelength')
#plt.ylabel('1- Normalized Flux')
plt.show()


'''
In the next part of the code, I am caluclating the equivalent width for the emission lines.
The equivalent width is: W = Integral (1 - (Flux(lambda)/Continuum(lambda))) dlambda
'''

#dlambda = 1.25   # Given from the head of the fits file as CD3_3
# Initializing the equaivalent width with 0
#u = spec/cont(wave)
#for i in q:
#	dw = (1.- u(i))*dlambda
#	w = w + dw
#print(dw)
#print(w)

#print(np.sum(w))
#print(i)
