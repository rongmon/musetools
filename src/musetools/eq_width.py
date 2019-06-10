
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

def plot_vel(vel,spec,spec_err,continuum,start_line,end_line,xc,yc,k,lamda,eqw,eqw_err):
	fig, axs = plt.subplots(2, 1, constrained_layout=True)
	axs[0].step(vel,spec,'-',label='Flux')#,vel,continuum,'--',label='Continuum',vel,spec_err,'r-',label ='Error')
	axs[0].step(vel,continuum,'--',label='Continuum')
	axs[0].step(vel,spec_err,'r-',label='Error')
	axs[0].axvline(x=start_line, linewidth=0.5)
	axs[0].axvline(x=end_line,linewidth = 0.5 )
	axs[0].legend(loc = 'upper right',prop={'size': 5})
	textstr = '\n'.join((
    r'$EW=%.9f$' % (eqw, ),
    r'$\sigma=%.9f$' % (eqw_err, )))
	axs[0].text(0.05, 0.95, textstr, transform=axs[0].transAxes, fontsize=5,
        verticalalignment='top')
	axs[0].set_xlabel('Velocity')
	axs[0].set_ylabel('Flux')

	axs[1].step(vel,spec/continuum,'-',label='Normalized Flux')#,vel,spec_err/continuum,'r-',label='Normalized Error')
	axs[1].step(vel,spec_err/continuum,'r-',label='Normalized Error')
	axs[1].axvline(x=start_line, linewidth = 0.5)
	axs[1].axvline(x=end_line, linewidth = 0.5)
	axs[1].legend(loc='upper right',prop={'size': 5})
	axs[1].set_xlabel('Velocity')
	axs[1].set_ylabel('Normalized Flux')
	fig.suptitle('Spectrum of Fe '+str(lamda)+' of the square '+str(k)+' with pix coordinates ('+str(xc)+','+str(yc)+')')
	fig.savefig('/home/ahmed/astro/figures/spectra/Spectrum_'+str(xc)+'_'+str(yc)+'_'+str(lamda)+'.png', bbox_inches='tight')   # save the figure to file
	plt.close(fig)

def Fewidth(wave,wrest,cx,cy,n):
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
	v_avg = []
	for i in lam_center:
		vel = u.veldiff(wrest,i)
		#plot_vel(vel,spec,spec_err,continuum,-1000,550,cx,cy,i)
		lmts = [-1000,550]
		temp = u.compute_EW(wrest,spec/continuum,i,lmts,spec_err/continuum)
		eqw[k]=temp['ew_tot']
		eqw_err[k] = temp['err_ew_tot']
		plot_vel(vel,spec,spec_err,continuum,-1000,550,cx,cy,n,i,eqw[k],eqw_err[k])
		k = k +1
		if i < 2612:
			l = np.where((vel < 550) & (vel > -1000))
			norm = np.trapz((1-spec[l]),x=vel[l])
			v_avg.append(np.trapz((1/norm)*vel[l]*(1-spec[l]),x=vel[l]))
	return eqw, eqw_err,v_avg#, spec, spec_err

line = input('Enter the element type (Fe or Mg): ')
xcen_cord = []
ycen_cord = []
if line == 'Fe':
	eqw_arc = np.zeros((4,1))
	eqw_err_arc = np.zeros((4,1))
	v_avg_arc = np.zeros((1,2))
	n= 1
	for cx, cy in zip(xcen, ycen):
		c=w.pixel_to_world_values(cx,cy,0)
		xcen_cord.append(c[0])
		ycen_cord.append(c[1])
		#print(c[0],c[1])
		eqw, eqw_err, v_avg = Fewidth(wave,wrest,cx,cy,n)
		v_avg = np.array(v_avg)
		n = n + 1
		#print(spec/spec_err)
		eqw_arc = np.hstack((eqw_arc,eqw))
		eqw_err_arc = np.hstack((eqw_err_arc,eqw_err))
		v_avg_arc = np.vstack((v_avg_arc,v_avg))
		#print('v_avg '+str(cx)+' & '+str(cy)+'')
		#print(v_avg)
	eqw_arc = eqw_arc[:,1:]
	eqw_err_arc = eqw_err_arc[:,1:]
	v_avg_arc = v_avg_arc[1:,:]
#print(eqw_arc)
#print(eqw_err_arc)
#print(v_avg_arc)
#np.array(your_list,dtype=float)

	xcen_cord = np.array(xcen_cord,dtype=float)
	ycen_cord = np.array(ycen_cord,dtype=float)
#print(xcen_cord)
#print(ycen_cord)
	image= io.narrow_band(7530.,7600.,wave,data)

	lam_center = [2586.650,2600.173,2612.654,2626.451]


	for j in range(4):
		width_in = 5
		fig=plt.figure()#1, figsize=(width_in, 7.5))
		ax = fig.add_subplot(111)
		im=ax.imshow(np.log10(np.abs(image)), interpolation='nearest',cmap=plt.get_cmap('viridis'),origin="lower")
		sc = ax.scatter(xcen, ycen, s =5**2,c=eqw_arc[j,:],marker='s', cmap='Reds')
		plt.colorbar(sc)
		ax.set_title('Equivalent Width of Fe line with rest wavelength: '+str(lam_center[j])+'A')
		ax.set_ylabel('Declination')
		ax.set_xlabel('Right Ascention')

		#ax.set_ylim([205,300])
		fig.savefig('/home/ahmed/astro/figures/equivalent_width/EW_FeII_'+str(lam_center[j])+'.png')
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
	#print(d)

	###### Checking the Signal to Noise Ratio for the Equivalent width
	snr = np.zeros((4,eqw_arc.shape[1]))
	lims = np.zeros((4,eqw_arc.shape[1]))
	for i in range(4):
		snr[i,:] = eqw_arc[i,:]/(3*eqw_err_arc[i,:])
		#print('SNR')
		#print(snr[i,:])

	det1 = np.where(np.abs(snr[0,:]) > 1)   # The detection for the Fe line 2586.650
	det2 = np.where(np.abs(snr[1,:]) > 1)   # The detection for the Fe line 2600.173
	det3 = np.where(np.abs(snr[2,:]) > 1)   # The detection for the Fe line 2612.654
	det4 = np.where(np.abs(snr[3,:]) > 1)   # The detection for the Fe line 2626.541

	non_det1 = np.where(np.abs(snr[0,:]) < 1)
	non_det2 = np.where(np.abs(snr[1,:]) < 1)
	non_det3 = np.where(np.abs(snr[2,:]) < 1)
	non_det4 = np.where(np.abs(snr[3,:]) < 1)

	def error_plot(d,eqw_arc,eqw_err_arc,i,det,non_det,mark,size):
		ew = eqw_arc[i,:]
		ew_err = eqw_err_arc[i,:]
		lam_center = [2586.650,2600.173,2612.654,2626.451]
		plt.errorbar(d[det], ew[det], yerr=ew_err[det], fmt='o', markersize=mark, capsize=size)
		if i <2:
			#plt.errorbar(d[non_det], 2*ew_err[non_det], yerr=2*ew_err[non_det], uplims=True, fmt='o',markersize=mark, capsize=size)
			plt.errorbar(d[non_det], 2*ew_err[non_det], yerr=1., uplims=True, fmt='o',markersize=mark, capsize=size)

		else:
			plt.errorbar(d[non_det], 2*ew_err[non_det], yerr=2*ew_err[non_det], lolims=True, fmt='o',markersize=mark, capsize=size)
		plt.title('EW Vs $\Delta \Theta$ for Fe line: '+str(lam_center[i])+' A')
		plt.xlabel('$\Delta \Theta$ (Arc Seconds)')
		plt.ylabel('EW (A)')
		manager = plt.get_current_fig_manager()
		manager.window.showMaximized()
		plt.savefig('/home/ahmed/astro/figures/EW_Vs_Separation_Angle/EW_SepAng_Fe_'+str(lam_center[i])+'.png')
		plt.clf()

	######

	#for i in range(4):
	error_plot(d,eqw_arc,eqw_err_arc,0,det1,non_det1,5,5)
	error_plot(d,eqw_arc,eqw_err_arc,1,det2,non_det2,5,5)
	error_plot(d,eqw_arc,eqw_err_arc,2,det3,non_det3,5,5)
	error_plot(d,eqw_arc,eqw_err_arc,3,det4,non_det4,5,5)

	plt.plot(d,eqw_arc[1,:]/eqw_arc[0,:],'o',markersize=5)
	plt.title('The Ratio of EW of Fe lines 2600 & 2586.650 Vs $\Delta \Theta$')
	plt.xlabel('$\Theta$')
	plt.ylabel('EW2/EW1')
	mng = plt.get_current_fig_manager()
	mng.window.showMaximized()
	plt.savefig('/home/ahmed/astro/figures/EW_Vs_Separation_Angle/EW_SepAng_Fe_2600_2586.png')
	plt.clf()


	plt.plot(d,v_avg_arc[:,0],'o',markersize=5)
	plt.title('Average Velocity of Fe 2586 Vs $\Delta \Theta$')
	plt.xlabel('$\Delta \Theta$')
	plt.ylabel('Avg Velocity')
	mg = plt.get_current_fig_manager()
	mg.window.showMaximized()
	plt.savefig('/home/ahmed/astro/figures/velocity/vel_vs_Sep_Ang_Fe_2586.png')
	plt.clf()

	plt.plot(d,v_avg_arc[:,1],'o',markersize=5)
	plt.title('Average Velocity of Fe 2600 Vs $\Delta \Theta$')
	plt.xlabel('$\Delta \Theta$')
	plt.ylabel('Avg Velocity')
	mg = plt.get_current_fig_manager()
	mg.window.showMaximized()
	plt.savefig('/home/ahmed/astro/figures/velocity/vel_vs_Sep_Ang_Fe_2600.png')
	plt.clf()

if line =='Mg':
	for cx, cy in zip(xcen, ycen):
		spec, spec_err = s.extract_square(cx, cy, wave, data, var, 5)
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
		print(wrest_fit)
		print(spec_fit)
		cont = np.poly1d(np.polyfit(wrest_fit, spec_fit, 5))
		continuum = cont(wrest)
		print(continuum)
		lam_center = [2796.351,2803.528]
		for i in lam_center:
			#print(i)
			vel = u.veldiff(wrest,i)
			#plot_vel(vel,spec,spec_err,continuum,-900,70)
			#w1 = compute_eqw(i,wrest,spec,continuum,-900,70)
			#print(w1)
			fig, axs = plt.subplots(2, 1, constrained_layout=True)
			start_line = -900
			end_line = 70
			axs[0].step(vel,spec,'-',label='Flux')#,vel,continuum,'--',label='Continuum',vel,spec_err,'r-',label ='Error')
			axs[0].step(vel,continuum,'--',label='Continuum')
			axs[0].step(vel,spec_err,'r-',label='Error')
			axs[0].axvline(x=start_line, linewidth=0.5)
			axs[0].axvline(x=end_line,linewidth = 0.5 )
			axs[0].legend(loc =0,prop={'size': 5})
			axs[0].set_xlabel('Velocity')
			axs[0].set_ylabel('Flux')
			axs[1].step(vel,spec/continuum,'-',label='Normalized Flux')#,vel,spec_err/continuum,'r-',label='Normalized Error')
			axs[1].step(vel,spec_err/continuum,'r-',label='Normalized Error')
			axs[1].axvline(x=start_line, linewidth = 0.5)
			axs[1].axvline(x=end_line, linewidth = 0.5)
			axs[1].legend(loc=0,prop={'size': 5})
			axs[1].set_xlabel('Velocity')
			axs[1].set_ylabel('Normalized Flux')
