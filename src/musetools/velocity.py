from __future__ import division
def vel(lam_offset, lam_galaxy):
    # lam_offset : represents the observed wavelength of the gas from the circumgalactic medium around the galaxy.
    # lam_galaxy : represents the observed wavelength of the gas from the galaxy itself
    z = (lam_offset / lam_galaxy) - 1
    c = 299792.458  # Speed of light in km/s
    beta = ((z + 1)**2 - 1)/((z+1)**2 + 1)
    del_v = beta * c
    print("The velocity of the gas w.r.t. the rest frame of the galaxy is: "+str(del_v)+" km/s ")
    
lam_offset = float(input('Enter the observed wavelength value of the emission line: '))
lam_galaxy = float(input('Enter the wavelength value of the emission line from the galaxy: ')) # This is calculated using the redshift z
# which you got using z of the galaxy: '))

vel(lam_offset, lam_galaxy)
