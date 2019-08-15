from astropy.io import fits
from astropy.io import ascii
from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt
from musetools import io as io
from musetools import spec as s
import musetools.util as u
import musetools.modeling as m
from scipy.optimize import curve_fit
import corner
import getpass
import emcee
from multiprocessing import Pool
username=getpass.getuser()

if username == 'bordoloi':
	fitsfile = '/Users/bordoloi/Dropbox/MUSE/LensedArc/RCS0327_16mc_zap.fits'
else:
	fitsfile = '/home/ahmed/Gdrive/astro/data/RCS0327_16mc_zap.fits'

# Defining our functions
def lnlike(theta, model, wave, flux, flux_err):
    l = -0.5 * (np.sum( ((flux - model(wave,*theta))/flux_err) **2. ))
    return l


def lnprior(theta,lower,upper):
    # theta: is the array that contain my parameters
    # upper: upper bounds on my parameters
    # lower: lower bounds on my parameters
    for i in range(len(theta)):
        if ((theta[i] < lower[i]) or (theta[i] > upper[i])):
            return -np.inf
            break

    return 0.0

def lnprob(theta, model, wave, flux, flux_err, lower, upper):
    lp = lnprior(theta, lower, upper)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, model, wave,flux, flux_err)


import importlib
importlib.reload(m)
info = ascii.read('/home/ahmed/Gdrive/astro/spectra/redshift_OII_vac_mcmc.dat')
xcen = info[0][:]
ycen = info[1][:]
zgal = info[4][:]

data = ascii.read('/home/ahmed/Gdrive/astro/spectra/stacked_spectrum_OII.dat')
wave = data['wave']
wave = u.airtovac(wave)
flx_norm = data['flx_norm']
flx_er_norm = data['flx_er_norm']

p0 =    [ 1.7,  1.,   50.]
upper = [1.72, 1.5, 1000.]
lower = [ 1.7,  0.,    0.]

popt, pcov = curve_fit(m.model_OII, wave, flx_norm,p0,sigma=flx_er_norm,bounds=(lower,upper))

# EMCEE code
ndim, nwalkers = int(len(p0)), 100
pos = [p0 + 1e-3 * np.random.randn(ndim) for i in range(nwalkers)]
with Pool() as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool, args=(m.model_OII, wave, flx_norm, flx_er_norm, lower, upper))
    n = 1500
    sampler.run_mcmc(pos, n, progress=True)

samples = sampler.chain[:,int(0.2*n):,:].reshape((-1,ndim))
np.save('/home/ahmed/Gdrive/astro/samples_emcee/OII/samples/samples_OII_stacked.npy',samples)

flat_samples =sampler.get_chain(discard=int(0.2*n), flat=True)

# Plotting the walkers
fig, ax = plt.subplots(3)
fig.set_figheight(15)
fig.set_figwidth(30)

ax[0].plot(sampler.chain[:,:,0].T,'k',alpha=0.3)
ax[0].set_ylabel('$z$')

ax[1].plot(sampler.chain[:,:,1].T,'k',alpha=0.3)
ax[1].set_ylabel('$tau$')

ax[2].plot(sampler.chain[:,:,2].T,'k',alpha=0.3)
ax[2].set_ylabel('$\sigma_{\lambda}$')
ax[2].set_xlabel('Number of Steps')
plt.ioff()
fig.savefig('/home/ahmed/Gdrive/astro/samples_emcee/OII/walkers/walkers_OII_stacked.pdf',overwrite=True)
plt.close()

p_opt = [];  up_sig = [];  lw_sig = []
for i in range(ndim):
    mcmc = np.percentile(flat_samples[:,i],[34,50,68])
    q = np.diff(mcmc)
    p_opt.append(mcmc[1])
    up_sig.append(q[1])
    lw_sig.append(q[0])

## Corner Plotting
figure = corner.corner(samples, labels=["z","$tau$","$\sigma_{\lambda}$"], quantiles=[0.34, 0.5, 0.68], show_titles=True,title_kwargs={"fontsize":12})
# Loop over the diagonal
axes = np.array(figure.axes).reshape((ndim,ndim))
for i in range(ndim):
    ax = axes[i,i]
    ax.axvline(p_opt[i],color="b",alpha=0.5)
for yi in range(ndim):
    for xi in range(yi):
        ax = axes[yi, xi]
        ax.axvline(p_opt[xi], color="b", alpha=0.5)
        ax.axvline(p_opt[yi], color="b", alpha=0.5)
        ax.plot(p_opt[xi],p_opt[yi],"sb")
plt.ioff()
figure.savefig('/home/ahmed/Gdrive/astro/samples_emcee/OII/corner/corner_OII_stacked.pdf',overwrite=True)
plt.close


fig, ax = plt.subplots()
ax.step(wave,flx_norm,label='Normalized Flux')
ax.step(wave,m.model_OII(wave,*p_opt),label='Model')
ax.set_title('Fitting the OII doublet for the Stacked Spectrum')
ax.set_xlabel('$Wavelength \lambda [\AA]$')
ax.set_ylabel('Normalized Flux')
ax.set_xlim([6640.,6750.])
ax.legend(loc=0)
fig.savefig('/home/ahmed/Gdrive/astro/samples_emcee/OII/fitting/fit_OII_stacked_spectrum.pdf',overwrite=True)

info = Table([[p_opt[0]],[lw_sig[0]],[up_sig[0]]], names = ['z','z_low','z_high'])
ascii.write(info,'/home/ahmed/Gdrive/astro/samples_emcee/OII/redshift.dat',overwrite=True)
