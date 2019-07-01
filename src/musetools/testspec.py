from astropy.io import ascii
import matplotlib.pyplot as plt
data1 = ascii.read('/home/ahmed/astro/spectra/spectrum_tot_114_226.dat')
wave = data1[0][:]
flx = data1[1][:]
flx_er = data1[2][:]
minindex = 200
maxindex = 650
wave = wave[minindex:maxindex]
flx = flx[minindex:maxindex]
flx_er = flx_er[minindex:maxindex]
plt.step(wave,flx)
plt.step(wave,flx_er)
plt.show()
