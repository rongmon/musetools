
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from musetools import io as io
from musetools import spec as s
from musetools import util as u
from astropy.wcs import WCS
import math as m
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
    ew = np.trapz(1-spec[l]/continuum[l],x=wrest[l])

    return ew

def plot_vel(vel,spec,spec_err,continuum,start_line,end_line,xc,yc,lamda):
	fig, axs = plt.subplots(2, 1, constrained_layout=True)
	axs[0].step(vel,spec,'-',label='Flux')#,vel,continuum,'--',label='Continuum',vel,spec_err,'r-',label ='Error')
	axs[0].step(vel,continuum,'--',label='Continuum')
	axs[0].step(vel,spec_err,'r-',label='Error')
	axs[0].axvline(x=start_line, linewidth=0.5)
	axs[0].axvline(x=end_line,linewidth = 0.5 )
	axs[0].legend(loc = 'upper right',prop={'size': 5})
	axs[0].set_xlabel('Velocity')
	axs[0].set_ylabel('Flux')

	axs[1].step(vel,spec/continuum,'-',label='Normalized Flux')#,vel,spec_err/continuum,'r-',label='Normalized Error')
	axs[1].step(vel,spec_err/continuum,'r-',label='Normalized Error')
	axs[1].axvline(x=start_line, linewidth = 0.5)
	axs[1].axvline(x=end_line, linewidth = 0.5)
	axs[1].legend(loc='upper right',prop={'size': 5})
	axs[1].set_xlabel('Velocity')
	axs[1].set_ylabel('Normalized Flux')
	fig.suptitle('Spectrum of Fe line '+str(lamda)+' of the square with pix coordinates ('+str(xc)+','+str(yc)+')')
	#manager = plt.get_current_fig_manager()
	#manager.window.showMaximized()
	fig.savefig('/home/ahmed/astro/figures/spectra/Spectrum_'+str(xc)+'_'+str(yc)+'_'+str(lamda)+'.pdf', bbox_inches='tight')   # save the figure to file
	plt.close(fig)
	#plt.savefig('Spectrum_'+str(xc)+'_'+str(yc)+'_'+str(lamda)+'.pdf')

def Fewidth(wave,wrest,cx,cy):
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
	eqw = np.zeros((4,1))
	eqw_err = np.zeros((4,1))
	k = 0
	for i in lam_center:
		vel = u.veldiff(wrest,i)
		plot_vel(vel,spec,spec_err,continuum,-1000,550,cx,cy,i)
		lmts = [-1000,550]
		temp = u.compute_EW(wrest,spec/continuum,i,lmts,spec_err/continuum)
		eqw[k]=temp['ew_tot']
		eqw_err[k] = temp['err_ew_tot']
		k = k +1

	return eqw, eqw_err

line = input('Enter the element type (Fe or Mg): ')
xcen_cord = []
ycen_cord = []
if line == 'Fe':
	eqw_arc = np.zeros((4,1))
	eqw_err_arc = np.zeros((4,1))
	for cx, cy in zip(xcen, ycen):
		c=w.pixel_to_world_values(cx,cy,0)
		xcen_cord.append(c[0])
		ycen_cord.append(c[1])
		print(c[0],c[1])
		eqw, eqw_err = Fewidth(wave,wrest,cx,cy)
		#print(eqw)
		eqw_arc = np.hstack((eqw_arc,eqw))
		eqw_err_arc = np.hstack((eqw_err_arc,eqw_err))
eqw_arc = eqw_arc[:,1:]
eqw_err_arc = eqw_err_arc[:,1:]
print(eqw_arc)
print(eqw_err_arc)
#np.array(your_list,dtype=float)
xcen_cord = np.array(xcen_cord,dtype=float)
ycen_cord = np.array(ycen_cord,dtype=float)
print(xcen_cord)
print(ycen_cord)
image= io.narrow_band(7530.,7600.,wave,data)

lam_center = [2586.650,2600.173,2612.654,2626.451]


for j in range(4):
	width_in = 10
	fig=plt.figure(1, figsize=(width_in, 15))
	ax = fig.add_subplot(111)
	im=ax.imshow(np.log10(np.abs(image)), interpolation='nearest',cmap=plt.get_cmap('viridis'),origin="lower")
	sc = ax.scatter(xcen, ycen, s =5**2,c=eqw_arc[j,:],marker='s', cmap='Reds')
	plt.colorbar(sc)
	ax.set_title('Equivalent Width of Fe line with rest wavelength: '+str(lam_center[j])+'A')
	ax.set_ylabel('Declination')
	ax.set_xlabel('Right Ascention')

	ax.set_ylim([205,300])
	fig.savefig('/home/ahmed/astro/figures/equivalent_width/EW_Fe_'+str(lam_center[j])+'.pdf')
	plt.close(fig)

# Choosing a reference point in the middle of the arc: xcen = 170, ycen = 274
#
# I will calculate the angular seperation for the other points from this center reference point.
d   = np.zeros(len(xcen))
xc = 170
yc = 274
cen = w.pixel_to_world_values(xc,yc,0)
cen = np.array(cen,dtype=float)
# cen[0] is the right ascention of the center of the Arc
# cen[1] is the declination of the center of the Arc.
for i in range(len(xcen_cord)):
	d[i] = m.acos(m.sin(m.radians(cen[1])) * m.sin(m.radians(ycen_cord[i])) + m.cos(m.radians(cen[1])) * m.cos(m.radians(ycen_cord[i])) * m.cos(m.radians( cen[0] - xcen_cord[i] )))
	d[i] = 3600 * m.degrees(d[i])
	if i < 14:
		d[i] = - d[i]
print(d)
for i in range(4):
	plt.errorbar(d,eqw_arc[i,:],yerr=eqw_err_arc[i,:],fmt='o',markersize=5, capsize=4)
	plt.title('Equvalent Width Vs Seperation Angle for Fe line: '+str(lam_center[i])+' A')
	plt.xlabel('Seperation Angle (Arc Seconds)')
	plt.ylabel('EW (A)')
	manager = plt.get_current_fig_manager()
	manager.window.showMaximized()
	plt.savefig('/home/ahmed/astro/figures/EW_Vs_Separation_Angle/EW_SepAng_Fe_'+str(lam_center[i])+'.pdf')
	plt.clf()

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
