# this is an input-output file, which takes the fits file as an input and gives us a white image of the data cube by suming all the values of the wavelength axis
# importing the required libraries
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from astropy.wcs import WCS

fitsfile = input("Enter the path to your file: ")
a = fits.open(fitsfile)

data = a[1].data  # the spectrum data are included in the data extension of the fits file
# data.shape   # you can use this command to know the dimensions of your data cube
hdu_hdr = a[1].header  # reading the header of the data into a variable called hdu_hdr

# We will transform the pixels' indexes into wavelength array
crval3 = hdu_hdr['CRVAL3']  # This is the starting value of the wavelength
crpix3 = hdu_hdr['CRPIX3']  # The pixel that has the value of CRVAL3
cd3_3 = hdu_hdr['CD3_3']    # The range of wavelength difference delta_lambda
wavedim = hdu_hdr['NAXIS3'] # The dimension of the data axis 3 (Wavelength)
# Do it
wave = crval3 + (crpix3 + np.arange(0, wavedim, 1.0)) * cd3_3 # This array contains the wavelength 


q=np.where(( wave > float(wave[0])) & (wave < float(wave[len(wave)-1])))  # Choosing specific wavelength interval to view (In my case, I used all the wavelength interval)
image = np.sum(data[q,:,:],axis=1)  # We now sum all the wavelength
# We will plot the image now using log10 scaling
fig = plt.figure()
ax = fig.add_subplot(111)
d = ax.imshow(np.log10(np.abs(image[0,:,:])), cmap = plt.get_cmap('viridis'), origin='lower')
plt.show()
