
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from musetools import io as io
from musetools import spec as s
from musetools import util as u
from astropy.wcs import WCS
import pdb

import getpass
# TEST BLAH BLAH BLAH

username=getpass.getuser()

if username == 'bordoloi':
	fitsfile = '/Users/bordoloi/Dropbox/MUSE/LensedArc/RCS0327_16mc_zap.fits'
else:
	fitsfile = '/home/ahmed/astro/data/RCS0327_16mc_zap.fits'



wave, data, var, header = io.open_muse_cube(fitsfile)
w = WCS(header)
zgal= 1.7037455
wrest = wave/(1.+zgal)
#xcen = 121   # x-values: 121, 140, 156, 243
#ycen = 245   # y-values: 245, 262, 272, 238

xcen = [114,114,115,118,121,124,127,133,137,141,148,153,160,166,170,177,185,191,198,203,208,213,220,225,231,238,244,246,244,240,238,242]
ycen = [226,233,237,241,244,248,252,257,260,264,269,271,274,274,274,274,274,272,271,270,268,266,263,259,255,249,244,240,237,234,228,224]


#s=w.pixel_to_world(xcen,ycen,0)
#print(s)

#spec, spec_err = s.extract_square(xcen, ycen, wave, data, var, 5)


def compute_eqw(lam_center,wrest,spec,continuum,vmin,vmax):
    vel = u.veldiff(wrest,lam_center)
    l = np.where((vel < vmax) & (vel > vmin))
    w = np.trapz(1-spec[l]/continuum[l],x=wrest[l])
    return w

def plot_vel(vel,spec,spec_err,continuum,start_line,end_line):
	plt.figure()
	plt.subplot(2,1,1)
	plt.step(vel,spec,'-',vel,continuum,'--',vel,spec_err,'r-')
	plt.axvline(x=start_line, linewidth=0.5)
	plt.axvline(x=end_line, linewidth = 0.5)
	plt.xlabel('Velocity')
	plt.ylabel('Flux')

	plt.subplot(2,1,2)
	plt.step(vel,spec/continuum,'-',vel,spec_err/continuum,'r-')
	#plt.ylim([-1,2])
	plt.axvline(x=start_line, linewidth =0.5)
	plt.axvline(x=end_line,linewidth=0.5)
	plt.xlabel('Velocity')
	plt.ylabel(' Normalized Flux')

	plt.show()
def Fewidth(wave,wrest,cx,cy):
	spec, spec_err = s.extract_square(cx, cy, wave, data, var, 5)
	minindex = 1750
	maxindex = 2100
	wave = wave[minindex:maxindex]
	spec = spec[minindex:maxindex]
	spec_err = spec_err[minindex:maxindex]
	wrest= wrest[minindex:maxindex]
	#plt.subplot(2,1,1)
	#plt.plot(wave,spec) Uncomment these two lines and modify the indices of the next
	# subplots to view the total flux
	minw = 6967.
	maxw = 7111.   # These are the wavelength limits for Fe lines
	q = np.where(( wave > minw) & (wave < maxw))
	wrest_fit = np.delete(wrest, q)
	spec_fit = np.delete(spec, q)
	cont = np.poly1d(np.polyfit(wrest_fit, spec_fit, 3))
	continuum = cont(wrest)
	lam_center = [2586.650,2600.173,2612.654,2626.451]
	eqw = np.zeros((4,1))
	k = 0
	for i in lam_center:
		vel = u.veldiff(wrest,i)
		#plot_vel(vel,spec,spec_err,continuum,-1000,550)
		eqw[k] = compute_eqw(i,wrest,spec,continuum,-1000,550)
		k = k +1

	return eqw

line = input('Enter the element type (Fe or Mg): ')
if line == 'Fe':
	eqw_arc = np.zeros((4,1))
	for cx, cy in zip(xcen, ycen):
		c=w.pixel_to_world(cx,cy,0)
		print(c)
		eqw = Fewidth(wave,wrest,cx,cy)
		#print(eqw)
		eqw_arc = np.hstack((eqw_arc,eqw))
eqw_arc = eqw_arc[:,1:]
print(eqw_arc)
for j in range(4):
	plt.scatter(xcen, ycen, c=eqw_arc[j,:], cmap='viridis')
	plt.colorbar()
	plt.show()



if line =='Mg':
	minindex = 2200
	maxindex = 2500
	wave = wave[minindex:maxindex]
	spec = spec[minindex:maxindex]
	spec_err = spec_err[minindex:maxindex]
	wrest= wrest[minindex:maxindex]
	minw = 7530.
	maxw = 7600.  #These are the wavelength limits for Mg II
	q = np.where(( wave > minw) & (wave < maxw))
	wrest_fit = np.delete(wrest, q)
	spec_fit = np.delete(spec, q)
	cont = np.poly1d(np.polyfit(wrest_fit, spec_fit, 3))
	continuum = cont(wrest)
	lam_center = [2796.351,2803.528]
	for i in lam_center:
		print(i)
		vel = u.veldiff(wrest,i)
		plot_vel(vel,spec,spec_err,continuum,-900,70)
		w1 = compute_eqw(i,wrest,spec,continuum,-900,70)
		print(w1)
