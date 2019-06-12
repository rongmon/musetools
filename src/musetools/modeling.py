import numpy as np
import matplotlib.pyplot as plt
from astropy.convolution import convolve, Gaussian1DKernel

def modelFe(v,v1,tau1,tau3,c1,c2,c3,sigma1,sigma2):#,sigma3,sigma4):
    v2 = v1 + 1563.2173499656212    # This is the average velocity for the 2nd absorption line 2600
    Fabs = 1 - tau1 * np.exp(-(v - v1)**2. / ( 2. * sigma1**2.)) - c1 * tau1 * np.exp(-(v - v2)**2. / (2. * sigma1**2.))
    v3 = v1 + 2998.7121643443234    # This is the average velocity for the 1st emission line 2612
    v4 = v1 + 4577.445992391543     # This is the average velocity for the 2nd emission line 2626.
    v5 = v1 + 5222.289150706484                      # This is the average velocity for the third emission line 2632
    Fems = 1 + tau3 * np.exp(-(v - v3)**2. / (2. * sigma2**2.)) + c2 * tau3 * np.exp(-(v - v4)**2. / (2 * sigma2**2.)) +c3 * tau3 * np.exp(-(v - v5)**2. / (2.* sigma2**2.))
    muse_kernel = (2.57398611619 / 2.355)
    g = Gaussian1DKernel(stddev=muse_kernel)
    # Convolve data
    F = Fabs * Fems
    fmodel = convolve(F, g,boundary='extend')
    return fmodel
'''
v1, tau1, c1, sigma1 = 0.0, 0.7, 1.1, 150
tau3, c2, sigma2 = 0.5, 1.4, 100

v = np.linspace(-2000.,6500.,1000)
F , Fabs, Fems= modelFe(v,v1,tau1,tau3,c1,c2,sigma1,sigma2)

plt.figure()
plt.plot(v,F,v,Fabs,v,Fems)
plt.xlabel('v')
plt.ylabel('Flux')
plt.ylim(0.0,2.0)
plt.show()
'''
def modelMg(v,v1,tau1,c1,sigma1):
    v2 = v1 + 768.4476528376828
    F = 1 - tau1 * np.exp(-(v - v1)**2 / ( 2 * sigma1**2)) - c1 * tau1 * np.exp(-(v - v2)**2 / (2 * sigma1**2))
    muse_kernel = (6. / 2.355)
    g = Gaussian1DKernel(stddev=muse_kernel)
    fmodel = convolve(F, g, boundary='extend')
    return fmodel
'''
v = np.linspace(-1000.,2000.,1000)
Fmg = modelMg(v,v1,tau1,c1,sigma1)

plt.figure()
plt.plot(v,Fmg)
plt.xlabel('v')
plt.ylabel('Flux')
plt.ylim(0.0,3.0)
plt.show()
'''
