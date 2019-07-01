# this is an input-output file, which takes the fits file as an input and gives us a white image of the data cube by suming all the values of the wavelength axis
# importing the required libraries
from musetools import io as io
import matplotlib.pyplot as plt
from musetools import spec as s
import numpy as np
import getpass

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

#spec, spec_err = s.extract_square(115, 237, wave, data, var, 5)
minwave = 7558.4#7090.     #7090.       #7555.
maxwave = 7563.7#7110.     #7110.       #7573.
ems_image = io.narrow_band(minwave, maxwave, wave, data,plot=False)
cont_min = 7645.#6930.
cont_max = 7650.3#6950.
cont_image = io.narrow_band(cont_min, cont_max, wave, data,plot=False)
#print(ems_image - cont_image)
np.savetxt('output.txt', np.nan_to_num(ems_image - cont_image))

zmin=-1
zmax=2


'''
The z stretching for the color map
'''
width_in = 10
fig=plt.figure(1, figsize=(width_in, 15))
ax = fig.add_subplot(311)
ax.imshow(np.log10(np.abs( ems_image-cont_image)), cmap = plt.get_cmap('viridis'), origin='lower',vmin=zmin, vmax=zmax)
ax.contour(np.nan_to_num(ems_image-cont_image),levels=[4],colors='white',alpha=0.6)
ax.set_title('Residual emission')
ax.set_ylim([205,300])
ax.set_xlim([40,270])


ax1 = fig.add_subplot(312)
ax1.imshow(np.log10(np.abs(cont_image )), cmap = plt.get_cmap('viridis'), origin='lower',vmin=zmin, vmax=zmax)
ax1.contour(cont_image,levels=[4],colors='white',alpha=0.6)
ax1.set_title('Continuum')
ax1.set_ylim([205,300])
ax1.set_xlim([40,270])




ax2 = fig.add_subplot(313)
ax2.imshow(np.log10(np.abs(ems_image)), cmap = plt.get_cmap('viridis'), origin='lower',vmin=zmin, vmax=zmax)
ax2.set_title('MgII emission with continuum')
ax2.set_ylim([205,300])
ax2.set_xlim([40,270])
plt.show()

#fig = plt.figure()
'''
The z stretching for the color map
'''
