
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from musetools import io as io
from musetools import spec as s
from musetools import util as u

import getpass


username=getpass.getuser()

if username == 'bordoloi':
	fitsfile = '/Users/bordoloi/Dropbox/MUSE/LensedArc/RCS0327_16mc_zap.fits'
else:
	fitsfile = '/home/ahmed/astro/data/RCS0327_16mc_zap.fits'



wave, data, var = io.open_muse_cube(fitsfile)
zgal= 1.7037455
wave_rest = wave/(1.+zgal)
xcen = 121   #  values: 121, 140, 156, 243
ycen = 245   #  values: 245, 262, 272, 238
spec, err_spec = s.extract_square(xcen, ycen, wave, data, var, 5)


#####
def plot_wave(wave,spec,err_spec,cont_wave,start_line,end_line):
	plt.figure()
	plt.subplot(2,1,1)
	plt.step(wave,spec,'-',wave,cont_wave,'--',wave,err_spec,'r-')
	plt.axvline(x=start_line, linewidth = 0.5)
	plt.axvline(x=end_line, linewidth = 0.5)
	plt.xlabel('Wavelength')
	plt.ylabel('Flux')

	plt.subplot(2,1,2)
	plt.step(wave,spec/cont_wave,'-',wave,err_spec/cont_wave,'r-')
	#plt.ylim([-1,2])
	plt.axvline(x=start_line, linewidth= 0.5)
	plt.axvline(x=end_line, linewidth = 0.5)
	plt.xlabel('Wavelength')
	plt.ylabel(' Normalized Flux')

	plt.show()

def plot_vel(vel,spec,err_spec,cont_velocity,start_line,end_line):
	plt.figure()
	plt.subplot(2,1,1)
	plt.step(vel,spec,'-',vel,cont_velocity,'--',vel,err_spec,'r-')
	plt.axvline(x=start_line, linewidth=0.5)
	plt.axvline(x=end_line, linewidth = 0.5)
	plt.xlabel('Velocity')
	plt.ylabel('Flux')

	plt.subplot(2,1,2)
	plt.step(vel,spec/cont_velocity,'-',vel,err_spec/cont_velocity,'r-')
	#plt.ylim([-1,2])
	plt.axvline(x=start_line, linewidth =0.5)
	plt.axvline(x=end_line,linewidth=0.5)
	plt.xlabel('Velocity')
	plt.ylabel(' Normalized Flux')

	plt.show()
#####
line = input('Enter the element type (Fe or Mg): ')

if line == 'Fe':
	minindex = 1750
	maxindex = 2100
	wave = wave[minindex:maxindex]
	spec = spec[minindex:maxindex]
	err_spec = err_spec[minindex:maxindex]
	wave_rest= wave_rest[minindex:maxindex]

	#plt.subplot(2,1,1)
	#plt.plot(wave,spec) Uncomment these two lines and modify the indices of the next
	# subplots to view the total flux

	minw = 6967.
	maxw = 7111.   # These are the wavelength limits for Fe lines
	q = np.where(( wave > minw) & (wave < maxw))
	wave_fit = np.delete(wave, q)
	wave_rest_fit = np.delete(wave_rest, q)
	spec_fit = np.delete(spec, q)

	cont = np.poly1d(np.polyfit(wave_fit, spec_fit, 3))
	cont_rest = np.poly1d(np.polyfit(wave_rest_fit,spec_fit, 3))
	'''
	The previous line does the polynomial fitting for the continuum
	'''
	# Do not forget to uncoment this part to show the figures

	#plt.plot(wave[2100:2600],spec[2100:2600], '-',wave[int_index:fin_index],p(wave[int_index:fin_index]),'--')
	graph_type = input('Enter the Graph type (w for wavelength or v for velocity): ')
	if graph_type == 'w':
		cont_wave = cont(wave)
		plot_wave(wave,spec,err_spec,cont_wave,7086.,7108.)


	elif graph_type == 'v':
		line_num = input('Enter which line do you want to be your reference v (choose 1, 2, 3, or 4): ')
		if line_num == '1':
			lam_center = 2586.650
			vel = u.veldiff(wave_rest,lam_center)
			vel_fit = np.delete(vel, q)
			cont_vel = np.poly1d(np.polyfit(vel_fit, spec_fit, 3))
			#cont_vel = np.poly1d(np.polynomial.legendre.legfit(vel_fit, spec_fit, 30))
			plot_vel(vel,spec,err_spec,cont_vel(vel),-1000.,600.)
			l1 = np.where((vel < 600.) & (vel > -1000.))
			w1 = np.trapz((1-spec[l1]/cont_rest(wave_rest[l1])))

			#u.compute_EW(wave,spec/cont(wave),wave_rest,[-1000.,600.],err_spec/cont(wave),False,zabs = zgal)
			print(w1)
		elif  line_num == '2':
			lam_center = 2600.173
			vel = u.veldiff(wave_rest,lam_center)
			vel_fit = np.delete(vel, q)
			cont_vel = np.poly1d(np.polyfit(vel_fit, spec_fit, 3))
			plot_vel(vel,spec,err_spec,cont_vel(vel),-1000.,600.)
			l2 = np.where((vel < 600.) & (vel > -1000.))
			w2 = np.trapz((1-spec[l2]/cont_rest(wave_rest[l2])))
			#u.compute_EW(wave,spec/cont(wave),wave_rest,[-1000.,600.],err_spec/cont(wave),False,zabs = zgal)
			print(w2)
		elif line_num == '3':
			lam_center = 2612.654
			vel = u.veldiff(wave_rest,lam_center)
			vel_fit = np.delete(vel, q)
			cont_vel = np.poly1d(np.polyfit(vel_fit, spec_fit, 3))
			plot_vel(vel,spec,err_spec,cont_vel(vel),-700.,400.)
			l3 = np.where((vel < 400.) & (vel > -700.))
			w3 = np.trapz((1-spec[l3]/cont_rest(wave_rest[l3])))
			print(w3)
		elif line_num == '4':
			lam_center = 2626.451
			vel = u.veldiff(wave_rest,lam_center)
			vel_fit = np.delete(vel, q)
			cont_vel = np.poly1d(np.polyfit(vel_fit, spec_fit, 3))
			plot_vel(vel,spec,err_spec,cont_vel(vel),-700.,380.)
			l4 = np.where((vel < 380.) & (vel > -700.))
			w4 = np.trapz((1-spec[l4]/cont_rest(wave_rest[l4])))
			print(w4)
		else:
			print('Unexpected Input !!!!')

	else:
		print('Unexpected Input !!!!')

	'''
	In the next part of the code, I am caluclating the equivalent width for the emission lines.
	The equivalent width is: W = Integral (1 - (Flux(lambda)/Continuum(lambda))) dlambda
	'''


elif line == 'Mg':
	minindex = 2200
	maxindex = 2500 #for Mg II lines

	wave = wave[minindex:maxindex]
	spec = spec[minindex:maxindex]
	err_spec = err_spec[minindex:maxindex]
	wave_rest = wave_rest[minindex:maxindex]

	minw = 7530.
	maxw = 7600.  #These are the wavelength limits for Mg II
	q = np.where(( wave > minw) & (wave < maxw))
	wave_fit = np.delete(wave, q)
	spec_fit = np.delete(spec, q)
	wave_rest_fit = np.delete(wave_rest, q)
	cont = np.poly1d(np.polyfit(wave_fit, spec_fit, 3))
	cont_rest = np.poly1d(np.polyfit(wave_rest_fit,spec_fit, 3))


	# Do not forget to uncoment this part to show the figures

	#plt.plot(wave[2100:2600],spec[2100:2600], '-',wave[int_index:fin_index],p(wave[int_index:fin_index]),'--')
	graph_type = input('Enter the Graph type (w for wavelength or v for velocity): ')
	if graph_type == 'w':
		cont_wave = cont(wave)
		plot_wave(wave,spec,err_spec,cont_wave,7555.75,7572.74)
	elif graph_type =='v':
		line_num = input('Enter the number of the line (Choose 1 or 2): ')
		if line_num == '1':
			lam_center = 2796.351
			vel = u.veldiff(wave_rest,lam_center)
			vel_fit = np.delete(vel, q)
			cont_vel = np.poly1d(np.polyfit(vel_fit, spec_fit, 3))
			plot_vel(vel,spec,err_spec,cont_vel(vel),-1000,600)
			l1 = np.where((vel < 600.) & (vel > -1000.))
			w1 = np.trapz((1-spec[l1]/cont_rest(wave_rest[l1])))
			print(w1)
		elif line_num == '2':
			lam_center = 2803.528
			vel = u.veldiff(wave_rest,lam_center)
			vel_fit = np.delete(vel, q)
			cont_vel = np.poly1d(np.polyfit(vel_fit, spec_fit, 3))
			plot_vel(vel,spec,err_spec,cont_vel(vel),-1000.,600.)
			l2 = np.where((vel < 600.) & (vel > -1000.))
			w2 = np.trapz((1-spec[l2]/cont_rest(wave_rest[l2])))
			print(w2)
	else:
		print('Unexpected Input !!!')

else:
	print('Unexpected Input !!!')
