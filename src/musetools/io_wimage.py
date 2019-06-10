# this is an input-output file, which takes the fits file as an input and gives us a white image of the data cube by suming all the values of the wavelength axis
# importing the required libraries
from musetools import io as io
import matplotlib.pyplot as plt
from musetools import spec as s
import numpy as np

import getpass
# TEST BLAH BLAH BLAH

username=getpass.getuser()

if username == 'bordoloi':
	fitsfile = '/Users/bordoloi/Dropbox/MUSE/LensedArc/RCS0327_16mc_zap.fits'
else:
	fitsfile = '/home/ahmed/astro/data/RCS0327_16mc_zap.fits'
#input("Enter the path to your file: ")


wave, data, var, header = io.open_muse_cube(fitsfile)
#w = WCS(header)
zgal= 1.7037455
wrest = wave/(1.+zgal)

spec, spec_err = s.extract_square(115, 237, wave, data, var, 5)
minwave = 7090.     #7090.       #7555.
maxwave = 7110.     #7110.       #7573.
ems_image = io.narrow_band(minwave, maxwave, wave, data,plot=True)
cont_min = 6930.
cont_max = 6950.
cont_image = io.narrow_band(cont_min, cont_max, wave, data,plot=True)
'''
plt.figure()
plt.step(wrest,spec,wrest,spec_err)
plt.show()
'''
#fig = plt.figure()
'''
The z stretching for the color map
'''
width_in = 10
fig=plt.figure(1, figsize=(width_in, 15))
ax = fig.add_subplot(111)
ax.imshow(np.log10(np.abs(cont_image - ems_image)), cmap = plt.get_cmap('viridis'), origin='lower')
ax.set_title('Emission of FeII 2626 Subtracted from the continuum')
#ax.set_ylim([205,300])
plt.show()
