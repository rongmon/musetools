
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from musetools import io as io
from musetools import spec as s
from musetools import util as u
from astropy.wcs import WCS
import math as m
import pdb
from astropy.io import ascii
from astropy.table import Table
import getpass

username=getpass.getuser()

if username == 'bordoloi':
	fitsfile = '/Users/bordoloi/Dropbox/MUSE/LensedArc/RCS0327_16mc_zap.fits'
else:
	fitsfile = '/home/ahmed/astro/data/RCS0327_16mc_zap.fits'

wave, data, var, header = io.open_muse_cube(fitsfile)
w = WCS(header)
zgal= 1.7037455
wrest = wave/(1.+zgal)

xcen = [114,114,115,118,121,124,127,133,137,141,148,153,160,166,170,177,185,191,198,203,208,213,220,225,231,238,244,246,244,240,238,242]
ycen = [226,233,237,241,244,248,252,257,260,264,269,271,274,274,274,274,274,272,271,270,268,266,263,259,255,249,244,240,237,234,228,224]

def spectrum_Fe1(wave, data,wrest, cx, cy):
	flx, flx_er = s.extract_square(cx, cy, wave, data, var, 5)
	minindex = 1750
	maxindex = 2100
	wave = wave[minindex:maxindex]
	flx = flx[minindex:maxindex]
	flx_er = flx_er[minindex:maxindex]
	wrest= wrest[minindex:maxindex]
	minw = 6967.
	maxw = 7111.
	q = np.where((wave > minw) & (wave < maxw))
	wrest_fit = np.delete(wrest, q)
	flx_fit = np.delete(flx, q)
	cont = np.poly1d(np.polyfit(wrest_fit, flx_fit, 3))
	continuum = cont(wrest)
	flx_norm = flx / continuum
	flx_er_norm = flx_er / continuum
	dat = Table([wave, wrest, flx, flx_er, continuum, flx_norm, flx_er_norm], names=['wave', 'wrest', 'flx', 'flx_er', 'continuum', 'flx_norm', 'flx_er_norm'])
	ascii.write(dat, '/home/ahmed/astro/spectra/spectrum_fe_'+str(cx)+'_'+str(cy)+'.dat',overwrite=True)
	return

def spectrum_Mg(wave, data, wrest, cx, cy):
	flx, flx_er = s.extract_square(cx, cy, wave, data, var, 5)
	minindex = 2200
	maxindex = 2500
	wave = wave[minindex:maxindex]
	flx = flx[minindex:maxindex]
	flx_er = flx_er[minindex:maxindex]
	wrest= wrest[minindex:maxindex]
	minw = 7530.
	maxw = 7600.
	q = np.where((wave > minw) & (wave < maxw))
	wrest_fit = np.delete(wrest, q)
	flx_fit = np.delete(flx, q)
	cont = np.poly1d(np.polyfit(wrest_fit, flx_fit, 3))
	continuum = cont(wrest)
	flx_norm = flx / continuum
	flx_er_norm = flx_er / continuum
	dat = Table([wave, wrest, flx, flx_er, continuum, flx_norm, flx_er_norm], names=['wave', 'wrest', 'flx', 'flx_er', 'continuum', 'flx_norm', 'flx_er_norm'])
	ascii.write(dat, '/home/ahmed/astro/spectra/spectrum_mg_'+str(cx)+'_'+str(cy)+'.dat',overwrite=True)
	return

for cx, cy in zip(xcen, ycen):
    spectrum_Fe1(wave, wrest, cx, cy)
    spectrum_Mg(wave, wrest, cx, cy)
