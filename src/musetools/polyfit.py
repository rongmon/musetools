import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from astropy.wcs import WCS

#%matplotlib inline
a=fits.open('/home/ahmed/astro/data/RCS0327_16mc_zap.fits')
data=a[1].data

hdu_hdr=a[1].header

crval3 = hdu_hdr['CRVAL3']  # This is the starting value of the wavelength
crpix3 = hdu_hdr['CRPIX3']  # The pixel that has the value of CRVAL3
cd3_3 = hdu_hdr['CD3_3']    # The range of wavelength difference delta_lambda
wavedim = hdu_hdr['NAXIS3'] # The dimension of the data axis 3 (Wavelength)
# Do it
wave = crval3 + (crpix3 + np.arange(0, wavedim, 1.0)) * cd3_3  # This array contains the wavelength

squaresize = 5
halfbox = (squaresize - 1)//2
xcen = 121
ycen = 245
flux = data[:,ycen-halfbox:ycen+halfbox+1, xcen-halfbox:xcen+halfbox+1]
spec = np.sum(flux, axis=(1,2))

fig = plt.figure()
plt.xlabel('Wavelength')
plt.ylabel('Flux')
int_index = 2200
fin_index = 2300
p = np.poly1d(np.polyfit(wave[int_index:fin_index],spec[int_index:fin_index],5))

#plt.plot(wave[2100:2600],spec[2100:2600], '-',wave[int_index:fin_index],p(wave[int_index:fin_index]),'--')
plt.plot(wave[2100:2500],spec[2100:2500]/p(wave[2100:2500]))
plt.show()
'''
np.random.seed(1
x = np.linspace(0,1,20)
y = np.cos(x) + 0.3*np.random.rand(20)
p = np.poly1d(np.polyfit(x,y,3))

t = np.linspace(0,1,200)
plt.plot(x,y, 'o', t, p(t), '-')
plt.show()
'''
