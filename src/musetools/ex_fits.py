### This file is used to extract the individual sepctra as fits files
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from musetools import io as io
from musetools import spec as s
import musetools.util as u
import musetools.modeling as m
from astropy.wcs import WCS
from lmfit import Model
import pdb

import getpass

username=getpass.getuser()

if username == 'bordoloi':
	fitsfile = '/Users/bordoloi/Dropbox/MUSE/LensedArc/RCS0327_16mc_zap.fits'
else:
	fitsfile = '/home/ahmed/astro/data/RCS0327_16mc_zap.fits'


'''
wave, data, var, header = io.open_muse_cube(fitsfile)
w = WCS(header)
zgal= 1.7037455
wrest = wave/(1.+zgal)
'''
#xcen = 121
#ycen = 245

xcen = [114,114,115,118]#,121,124,127,133,137,141,148,153,160,166,170,177,185,191,198,203,208,213,220,225,231,238,244,246,244,240,238,242]
ycen = [230,233,237,241]#,244,248,252,257,260,264,269,271,274,274,274,274,274,272,271,270,268,266,263,259,255,249,244,240,237,234,228,224]


for cx, cy in zip(xcen, ycen):
	wave, data, var, header = io.open_muse_cube(fitsfile)
	w = WCS(header)
	zgal= 1.7037455
	wrest = wave/(1.+zgal)
	spec, spec_err = s.extract_square(cx, cy, wave, data, var, 5)
	minindex = 1750
	maxindex = 2100
	wave = wave[minindex:maxindex]
	spec = spec[minindex:maxindex]
	spec_err = spec_err[minindex:maxindex]
	wrest= wrest[minindex:maxindex]
	minw = 6967.
	maxw = 7111.   # These are the wavelength limits for Fe lines
	q = np.where(( wave > minw) & (wave < maxw))
	wrest_fit = np.delete(wrest, q)
	spec_fit = np.delete(spec, q)
	cont = np.poly1d(np.polyfit(wrest_fit, spec_fit, 3))  # Defining my polynomial
	continuum = cont(wrest)
	lam_center = [2586.650,2600.173,2612.654,2626.451]
	norm_flx = spec/continuum
	norm_flx_err = spec_err/continuum
	vel1 = u.veldiff(wrest,lam_center[0])
	#vel2 = u.veldiff(wrest,lam_center[1])
	#vel3 = u.veldiff(wrest,lam_center[2])
	#vel4 = u.veldiff(wrest,lam_center[3])
	#from scipy.optimize import curve_fit
	#popt, pcov = curve_fit(m.modelFe, vel1, norm_flx, sigma=norm_flx_err)
	#print(popt)
	#print(pcov)
	gmodel = Model(m.modelFe)
	result = gmodel.fit(norm_flx,v=vel1, v1=0, tau1=0.7, tau3= 0.4, c1=1.1, c2=1.7,c3=1., sigma1=150, sigma2=100)#,sigma3=100,sigma4=95)
	print(result.fit_report())
	plt.step(vel1, norm_flx, label='Normalized Flux')
	plt.plot(vel1, result.best_fit, 'y-',label='Model')
	plt.step(vel1, norm_flx_err,'r',label='Error')
	plt.legend(loc=0)
	plt.title('Normalized Flux Vs Velocity for '+str(cx)+' & '+str(cy)+'')
	plt.xlabel('Velocity')
	plt.ylabel('Normalized Flux')
	plt.xlim([-2100,6200])
	plt.show()
