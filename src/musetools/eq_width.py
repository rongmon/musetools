
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from musetools import io as io
from musetools import spec as s


fitsfile = '/home/ahmed/astro/data/RCS0327_16mc_zap.fits'
wave, data, var = io.open_muse_cube(fitsfile)

xcen = 121
ycen = 245
spec, err_spec = s.extract_square(xcen, ycen, wave, data, var, 5)

#flux = data[:,ycen-halfbox:ycen+halfbox+1, xcen-halfbox:xcen+halfbox+1]
#spec = np.sum(flux, axis=(1,2))

fig = plt.figure()
#plt.subplot(2,1,1)
#plt.plot(wave,spec) Uncomment these two lines and modify the indices of the next
# subplots to view the total flux

int_index = 2200
fin_index = 2300
p = np.poly1d(np.polyfit(wave[int_index:fin_index],spec[int_index:fin_index],5))
'''
The previous line does the polynomial fitting for the continuum
'''

#plt.plot(wave[2100:2600],spec[2100:2600], '-',wave[int_index:fin_index],p(wave[int_index:fin_index]),'--')
plt.subplot(2,1,1)
plt.plot(wave[2100:2800],spec[2100:2800],'-',wave[2200:2300],p(wave[2200:2300]),'--')
plt.xlabel('Wavelength')
plt.ylabel('Flux')


plt.subplot(2,1,2)
plt.plot(wave[2100:2800],1-spec[2100:2800]/p(wave[2100:2800]))
plt.xlabel('Wavelength')
plt.ylabel('Flux')
plt.show()
