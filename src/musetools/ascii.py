from astropy.io import ascii
data = ascii.read('/home/ahmed/astro/data/rcs0327-knotE-allres-combwC1.txt')
import numpy as np
import matplotlib.pyplot as plt
wave = data[0][:]
### Transforming to the rest frame wavelength
zgal= 1.7037455
wrest = wave/(1.+zgal)
###
flux = data[1][:]
noise = data[2][:]
q = np.where(flux > 0)
wrest = wrest[q]
flux = flux[q]
noise = noise[q]
print(wrest.shape)
min_index = 8000
max_index = 11950


plt.figure()
plt.step(wrest[min_index:max_index],flux[min_index:max_index],wrest[min_index:max_index],noise[min_index:max_index])
plt.show()
plt.close()

wmin = 2400
wmax = 2550
f = np.where((wrest > wmin) & (wrest < wmax))
wrest_fit = np.delete(wrest, f)
spec_fit = np.delete(flux, f)
print(wrest_fit.shape)
print(spec_fit.shape)
cont = np.poly1d(np.polyfit(wrest_fit, spec_fit, 3))
continuum = cont(wrest)
plt.figure()
plt.step(wrest[min_index:max_index],flux[min_index:max_index])
plt.step(wrest[min_index:max_index],continuum[min_index:max_index])
plt.step(wrest[min_index:max_index],noise[min_index:max_index])
plt.show()
