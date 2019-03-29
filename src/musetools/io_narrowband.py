# This is an input-ouput file, which takes the fits file as an input and gives us a narrow band image by summing all the wavelength within given small wavelength interval
# importing the required libraries
from musetools import io as io

fitsfile = '/home/ahmed/astro/data/RCS0327_16mc_zap.fits'
#input("Enter the path to your file: ")
wave, data, var = io.open_muse_cube(fitsfile)

minwave = float(input('Enter the minimum value for your wavelength: '))
maxwave = float(input('Enter the maximum value for your wavelength: '))

io.narrow_band(minwave, maxwave, wave, data)
