import numpy as np
import matplotlib.pyplot as plt
from astropy.convolution import convolve, Gaussian1DKernel
import musetools.util as u

def model_Fe(v,vabs,vems,sig1,sig2,sig3,tau1,tau2,tau3,c1,c2,c3,c4,c5,c6,c7,c8):
    spec_res1=  2.62798855#2.88145487703    # For the lines set around 2300-2400 A
    spec_res2=  2.55817449#2.57398611619    # For the lines set around 2600.
    '''
    vabs               # Absorption Velocity
    vems               # Emission Velocity
    sig1               # Standard Deviation in the Absorption Velocity
    sig2               # Standard Deviation in the Emission Velocity
    sig3               # Standard Deviation in the velocity of the systematic component
    tau1               # Optical Depth of the first Absorption line 2586
    tau2               # The Beak of the Emission line 2612
    tau3               # Optical Depth of the Systematic Component
    c1                 # The line ratio 2600/2586
    c2                 # The line ratio 2344/2586
    c3                 # The line ratio 2374/2586
    c4                 # The line ratio 2382/2586
    c5                 # The line ratio 2626/2612
    c6                 # The line ratio 2632/2612
    c7                 # The line ratio 2365/2612
    c8                 # The line ratio 2396/2612
    '''
    lam_cen = [2586.650,2600.173,2612.654,2626.451,2632.1081,2344.212,2365.552,2374.460,2382.764,2396.355]
    '''
    Absorption lines: 2344.212, 2374.460, 2382.764, 2586.650, 2600.173
    Emission lines: 2365.552, 2396.355, 2612.654, 2626.451, 2632.1081
    '''
    # Absorption Lines Velocities
    v2 = vabs + u.veldiff(lam_cen[1],lam_cen[0])           # The Velocity of the Absorption line 2600
    v3 = vabs + u.veldiff(lam_cen[5],lam_cen[0])           # The Velocity of the Absorption line 2344
    v4 = vabs + u.veldiff(lam_cen[7],lam_cen[0])           # The Velocity of the Absorption line 2374
    v5 = vabs + u.veldiff(lam_cen[8],lam_cen[0])           # The Velocity of the Absorption line 2382
    # Emission Lines Velocities
    v6 = vems + u.veldiff(lam_cen[3],lam_cen[0])           # The Velocity of the Emission line 2626
    v7 = vems + u.veldiff(lam_cen[4],lam_cen[0])           # The Velocity of the Emission line 2632
    v8 = vems + u.veldiff(lam_cen[6],lam_cen[0])           # The Velocity of the Emission line 2365
    v9 = vems + u.veldiff(lam_cen[9],lam_cen[0])           # The Velocity of the Emission line 2396

    #lam_cen = [2586.650,2600.173,2612.654,2626.451,2632.1081]
    # The Velocity of the Emission line 2632
    # The Absorption component of the model
    Fabs = 1. - (tau1 * np.exp(-0.5*((v - vabs)/sig1)**2.) * np.piecewise(v,[v < 0.0, v >= 0.0], [1., 0.]) +
            c1 * tau1 * np.exp(-0.5*((v - v2)/sig1)**2.) * np.piecewise(v,[v < (v2 - vabs), v >= (v2 - vabs)], [1., 0.]) +
            c2 * tau1 * np.exp(-0.5*((v - v3)/sig1)**2.) * np.piecewise(v,[v < (v3 - vabs), v >= (v3 - vabs)], [1., 0.]) +
            c3 * tau1 * np.exp(-0.5*((v - v4)/sig1)**2.) * np.piecewise(v,[v < (v4 - vabs), v >= (v4 - vabs)], [1., 0.]) +
            c4 * tau1 * np.exp(-0.5*((v - v5)/sig1)**2.) * np.piecewise(v,[v < (v5 - vabs), v >= (v5 - vabs)], [1., 0.]) )
    """
    Fabs = 1. - (tau1 * np.exp(-0.5*((v - vabs)/sig1)**2.) * np.piecewise(v,[v<vabs, v>= vabs], [1., 0.]) +
            c1 * tau1 * np.exp(-0.5*((v - v2)/sig1)**2.) * np.piecewise(v,[v<v2, v>= v2], [1., 0.]) +
            c2 * tau1 * np.exp(-0.5*((v - v3)/sig1)**2.) * np.piecewise(v,[v<v3, v>= v3], [1., 0.]) +
            c3 * tau1 * np.exp(-0.5*((v - v4)/sig1)**2.) * np.piecewise(v,[v<v4, v>= v4], [1., 0.]) +
            c4 * tau1 * np.exp(-0.5*((v - v5)/sig1)**2.) * np.piecewise(v,[v<v5, v>= v5], [1., 0.]) )
    """
    # The Emission Component of the model
    Fems = 1. + (tau2 * np.exp(-0.5*((v - (vems+u.veldiff(lam_cen[2],lam_cen[0])))/sig2)**2.) +
            c5 * tau2 * np.exp(-0.5*((v - v6)/sig2)**2.) +
            c6 * tau2 * np.exp(-0.5*((v - v7)/sig2)**2.) +
            c7 * tau2 * np.exp(-0.5*((v - v8)/sig2)**2.) +
            c8 * tau2 * np.exp(-0.5*((v - v9)/sig2)**2.) )
    # The systematic Component of the model
    dv_abs1 = u.veldiff(lam_cen[1],lam_cen[0])
    dv_abs2 = u.veldiff(lam_cen[5],lam_cen[0])
    dv_abs3 = u.veldiff(lam_cen[7],lam_cen[0])
    dv_abs4 = u.veldiff(lam_cen[8],lam_cen[0])
    Fsys = 1. - (tau3 * np.exp(-0.5*(v/sig3)**2.)  +
            tau3 * np.exp(-0.5*((v - dv_abs1)/sig3)**2.)  +
            tau3 * np.exp(-0.5*((v - dv_abs2)/sig3)**2.)  +
            tau3 * np.exp(-0.5*((v - dv_abs3)/sig3)**2.)  +
            tau3 * np.exp(-0.5*((v - dv_abs4)/sig3)**2.) )
    # The total unconvolved model
    F =  Fabs * Fems * Fsys
    # 6570./( 1. + 1.7039798441343401) = 2429.7518394052004
    # This wavelength corresponds to the end of the wavelength interval of the first few FeII lines
    q1 = np.where(v > u.veldiff(2430.,lam_cen[0]))
    q2 = np.where(v < u.veldiff(2430.,lam_cen[0]))
    F_1     = np.array(F);   F_2     = np.array(F)
    F_1[q1] = 1. ;           F_2[q2] = 1.
    # convolving the model
    muse_kernel1 = ((spec_res1/1.25) / 2.355);          muse_kernel2 = ((spec_res2/1.25) / 2.355)
    g1 = Gaussian1DKernel(stddev=muse_kernel1);         g2 = Gaussian1DKernel(stddev=muse_kernel2)
    F_conv1 = convolve(F_1, g1, boundary='extend');     F_conv2 = convolve(F_2, g2, boundary='extend')

    F_conv = F_conv1 * F_conv2

    return F_conv

def model_Fe_comps(v,vabs,vems,sig1,sig2,sig3,tau1,tau2,tau3,c1,c2,c3,c4,c5,c6,c7,c8):
    spec_res1=  2.62798855#2.88145487703    # For the lines set around 2300-2400 A
    spec_res2=  2.55817449#2.57398611619    # For the lines set around 2600.
    '''
    vabs               # Absorption Velocity
    vems               # Emission Velocity
    sig1               # Standard Deviation in the Absorption Velocity
    sig2               # Standard Deviation in the Emission Velocity
    sig3               # Standard Deviation in the velocity of the systematic component
    tau1               # Optical Depth of the first Absorption line 2586
    tau2               # The Beak of the Emission line 2612
    tau3               # Optical Depth of the Systematic Component
    c1                 # The line ratio 2600/2586
    c2                 # The line ratio 2344/2586
    c3                 # The line ratio 2374/2586
    c4                 # The line ratio 2382/2586
    c5                 # The line ratio 2626/2612
    c6                 # The line ratio 2632/2612
    c7                 # The line ratio 2365/2612
    c8                 # The line ratio 2396/2612
    '''
    lam_cen = [2586.650,2600.173,2612.654,2626.451,2632.1081,2344.212,2365.552,2374.460,2382.764,2396.355]
    '''
    Absorption lines: 2344.212, 2374.460, 2382.764, 2586.650, 2600.173
    Emission lines: 2365.552, 2396.355, 2612.654, 2626.451, 2632.1081
    '''
    # Absorption Lines Velocities
    v2 = vabs + u.veldiff(lam_cen[1],lam_cen[0])           # The Velocity of the Absorption line 2600
    v3 = vabs + u.veldiff(lam_cen[5],lam_cen[0])           # The Velocity of the Absorption line 2344
    v4 = vabs + u.veldiff(lam_cen[7],lam_cen[0])           # The Velocity of the Absorption line 2374
    v5 = vabs + u.veldiff(lam_cen[8],lam_cen[0])           # The Velocity of the Absorption line 2382
    # Emission Lines Velocities
    v6 = vems + u.veldiff(lam_cen[3],lam_cen[0])           # The Velocity of the Emission line 2626
    v7 = vems + u.veldiff(lam_cen[4],lam_cen[0])           # The Velocity of the Emission line 2632
    v8 = vems + u.veldiff(lam_cen[6],lam_cen[0])           # The Velocity of the Emission line 2365
    v9 = vems + u.veldiff(lam_cen[9],lam_cen[0])           # The Velocity of the Emission line 2396

    #lam_cen = [2586.650,2600.173,2612.654,2626.451,2632.1081]
    # The Velocity of the Emission line 2632
    # The Absorption component of the model
    Fabs = 1. - (tau1 * np.exp(-0.5*((v - vabs)/sig1)**2.) * np.piecewise(v,[v < 0.0, v >= 0.0], [1., 0.]) +
            c1 * tau1 * np.exp(-0.5*((v - v2)/sig1)**2.) * np.piecewise(v,[v < (v2 - vabs), v >= (v2 - vabs)], [1., 0.]) +
            c2 * tau1 * np.exp(-0.5*((v - v3)/sig1)**2.) * np.piecewise(v,[v < (v3 - vabs), v >= (v3 - vabs)], [1., 0.]) +
            c3 * tau1 * np.exp(-0.5*((v - v4)/sig1)**2.) * np.piecewise(v,[v < (v4 - vabs), v >= (v4 - vabs)], [1., 0.]) +
            c4 * tau1 * np.exp(-0.5*((v - v5)/sig1)**2.) * np.piecewise(v,[v < (v5 - vabs), v >= (v5 - vabs)], [1., 0.]) )

    Fabs1 = 1. - tau1 * np.exp(-0.5*((v - vabs)/sig1)**2.) * np.piecewise(v,[v < 0.0, v >= 0.0], [1., 0.]) #2586
    Fabs2 = 1. - c1 * tau1 * np.exp(-0.5*((v - v2)/sig1)**2.) * np.piecewise(v,[v < (v2 - vabs), v >= (v2 - vabs)], [1., 0.])  #2600
    Fabs3 = 1. - c2 * tau1 * np.exp(-0.5*((v - v3)/sig1)**2.) * np.piecewise(v,[v < (v3 - vabs), v >= (v3 - vabs)], [1., 0.])  #2344
    Fabs4 = 1. - c3 * tau1 * np.exp(-0.5*((v - v4)/sig1)**2.) * np.piecewise(v,[v < (v4 - vabs), v >= (v4 - vabs)], [1., 0.])  #2374
    Fabs5 = 1. - c4 * tau1 * np.exp(-0.5*((v - v5)/sig1)**2.) * np.piecewise(v,[v < (v5 - vabs), v >= (v5 - vabs)], [1., 0.])  #2382
    """
    Fabs = 1. - (tau1 * np.exp(-0.5*((v - vabs)/sig1)**2.) * np.piecewise(v,[v<vabs, v>= vabs], [1., 0.]) +
            c1 * tau1 * np.exp(-0.5*((v - v2)/sig1)**2.) * np.piecewise(v,[v<v2, v>= v2], [1., 0.]) +
            c2 * tau1 * np.exp(-0.5*((v - v3)/sig1)**2.) * np.piecewise(v,[v<v3, v>= v3], [1., 0.]) +
            c3 * tau1 * np.exp(-0.5*((v - v4)/sig1)**2.) * np.piecewise(v,[v<v4, v>= v4], [1., 0.]) +
            c4 * tau1 * np.exp(-0.5*((v - v5)/sig1)**2.) * np.piecewise(v,[v<v5, v>= v5], [1., 0.]) )
    Fabs1 = 1. - tau1 * np.exp(-0.5*((v - vabs)/sig1)**2.) * np.piecewise(v,[v<vabs, v>= vabs], [1., 0.]) #2586
    Fabs2 = 1. - c1 * tau1 * np.exp(-0.5*((v - v2)/sig1)**2.) * np.piecewise(v,[v<v2, v>= v2], [1., 0.])  #2600
    Fabs3 = 1. - c2 * tau1 * np.exp(-0.5*((v - v3)/sig1)**2.) * np.piecewise(v,[v<v3, v>= v3], [1., 0.])  #2344
    Fabs4 = 1. - c3 * tau1 * np.exp(-0.5*((v - v4)/sig1)**2.) * np.piecewise(v,[v<v4, v>= v4], [1., 0.])  #2374
    Fabs5 = 1. - c4 * tau1 * np.exp(-0.5*((v - v5)/sig1)**2.) * np.piecewise(v,[v<v5, v>= v5], [1., 0.])  #2382
    """
    # The Emission Component of the model
    Fems = 1. + (tau2 * np.exp(-0.5*((v - (vems+u.veldiff(lam_cen[2],lam_cen[0])))/sig2)**2.) +
            c5 * tau2 * np.exp(-0.5*((v - v6)/sig2)**2.) +
            c6 * tau2 * np.exp(-0.5*((v - v7)/sig2)**2.) +
            c7 * tau2 * np.exp(-0.5*((v - v8)/sig2)**2.) +
            c8 * tau2 * np.exp(-0.5*((v - v9)/sig2)**2.) )
    Fems1 = 1. + tau2 * np.exp(-0.5*((v - (vems+u.veldiff(lam_cen[2],lam_cen[0])))/sig2)**2.) #2612
    Fems2 = 1. + c5 * tau2 * np.exp(-0.5*((v - v6)/sig2)**2.)  #2626
    Fems3 = 1. + c6 * tau2 * np.exp(-0.5*((v - v7)/sig2)**2.)  #2632
    Fems4 = 1. + c7 * tau2 * np.exp(-0.5*((v - v8)/sig2)**2.)  #2365
    Fems5 = 1. + c8 * tau2 * np.exp(-0.5*((v - v9)/sig2)**2.)  #2396
    # The systematic Component of the model
    dv_abs1 = u.veldiff(lam_cen[1],lam_cen[0])
    dv_abs2 = u.veldiff(lam_cen[5],lam_cen[0])
    dv_abs3 = u.veldiff(lam_cen[7],lam_cen[0])
    dv_abs4 = u.veldiff(lam_cen[8],lam_cen[0])
    Fsys = 1. - (tau3 * np.exp(-0.5*(v/sig3)**2.)  +
            tau3 * np.exp(-0.5*((v - dv_abs1)/sig3)**2.)  +
            tau3 * np.exp(-0.5*((v - dv_abs2)/sig3)**2.)  +
            tau3 * np.exp(-0.5*((v - dv_abs3)/sig3)**2.)  +
            tau3 * np.exp(-0.5*((v - dv_abs4)/sig3)**2.)  )
    # The total unconvolved model
    F = Fabs * Fems * Fsys
    # 6570./( 1. + 1.7039798441343401) = 2429.7518394052004
    # This wavelength corresponds to the end of the wavelength interval of the first few FeII lines
    q1 = np.where(v > u.veldiff(2430.,lam_cen[0]))
    q2 = np.where(v < u.veldiff(2430.,lam_cen[0]))
    F_1     = np.array(F)  ; F_2     = np.array(F)
    F_1[q1] = 1. ; F_2[q2] = 1.
    # convolving the model
    muse_kernel1 = ((spec_res1/1.25) / 2.355);          muse_kernel2 = ((spec_res2/1.25) / 2.355 )
    g1 = Gaussian1DKernel(stddev=muse_kernel1);         g2 = Gaussian1DKernel(stddev=muse_kernel2)
    F_conv1 = convolve(F_1, g1, boundary='extend');     F_conv2 = convolve(F_2, g2, boundary='extend')

    F_conv = F_conv1 * F_conv2

    return F_conv, F, Fabs, Fabs1, Fabs2, Fabs3, Fabs4, Fabs5, Fems, Fems1, Fems2, Fems3, Fems4, Fems5, Fsys


def Fout_2586(v, vabs, sig1, tau1):
    Fabs1 = 1. - tau1 * np.exp(-0.5*((v - vabs)/sig1)**2.) * np.piecewise(v,[v< 0.0, v>= 0.0], [1., 0.]) #2586
    return Fabs1

'''
def modelFe(v,v1,v3,tau1,tau3,c1,c2,c3,sigma1,sigma2):#,sigma3,sigma4):
    v2 = v1 + 1563.2173499656212    # This is the average velocity for the 2nd absorption line 2600
    Fabs = 1 - tau1 * np.exp(-(v - v1)**2. / ( 2. * sigma1**2.)) - c1 * tau1 * np.exp(-(v - v2)**2. / (2. * sigma1**2.))
    #v3 = v1 + 2998.7121643443234    # This is the average velocity for the 1st emission line 2612
    v4 = v3 + 1578.9749805940487
    #v1 + 4577.445992391543     # This is the average velocity for the 2nd emission line 2626.
    v5 = v3 + 2224.192279518938#v1 + 5222.289150706484                      # This is the average velocity for the third emission line 2632
    Fems = 1 + tau3 * np.exp(-(v - v3)**2. / (2. * sigma2**2.)) + c2 * tau3 * np.exp(-(v - v4)**2. / (2 * sigma2**2.)) +c3 * tau3 * np.exp(-(v - v5)**2. / (2.* sigma2**2.))
    muse_kernel = ((2.57398611619/1.25) / 2.355)
    g = Gaussian1DKernel(stddev=muse_kernel)
    # Convolve data
    F = Fabs * Fems
    fmodel = convolve(F, g,boundary='extend')
    return Fabs
'''
# For the MgII Lines
### Defint the MgII Model
def model_Mg(v,vabs,vems1,vems2,sig1,sig2,sig3,sig4,tau1,tau2,tau3,tau4,c1,c2,c3):
    # [2797.084, 2799.326, 2804.346, 2807.975]   corrected to vaccum
    # These are the emission line centroids as measured through QfitsView and the global red shift measured from the w stacked OII spectrum
    lam_cen = [2796.351, 2803.528, 2798.084, 2799.326, 2804.346, 2808.975]
    # The first two lines are the absorption lines
    # The next 4 are the main and secondary emission lines
    '''
    vabs: Outflow Velocity of the absorption lines
    vems1: Outflow Velocity of the two main emission lines
    vems2: Outflow Velocity of the secondary emission lines
    tau1: Optical Depth of the absorption line 2796
    tau2: Height of the Main Emission line 2797
    tau3: Height of the Secondary Emission line 2808
    tau4: Optical Depth of the systematic component
    sig1: the standard deviation in the absorption velocity
    sig2: the standard deviation in the emission velocity vems1
    sig3: the standard deviation in the emission velocity vems2
    sig4: the standard deviation in the velocity of the systemtic component
    c1:   the line ratio between 2803/2796
    c2:   the line ratio between 2804/2797
    c3:   the line ratio between 2799/2808
    '''
    # Absorption Line Velocity
    dv = u.veldiff(lam_cen[1],lam_cen[0])
    v2 = vabs + dv  #u.veldiff(lam_cen[1],lam_cen[0])
    # Emission Lines Velocities
    #v3 = vems1 + u.veldiff(lam_cen[2],lam_cen[0])
    v4 = vems1 + dv
    #v5 = vems2 + u.veldiff(lam_cen[3],lam_cen[0])
    v6 = vems2 + dv#u.veldiff(lam_cen[5],lam_cen[2])
    # Absorption Component
    Fabs = 1. - (tau1 * np.exp(-0.5*((v - vabs)/sig1)**2.) * np.piecewise(v,[v< 0.0, v>= 0.0], [1., 0.]) +
            c1 * tau1 * np.exp(-0.5*((v - v2)/sig1)**2.) * np.piecewise(v,[v< (v2 - vabs), v>= (v2 - vabs)], [1., 0.]) )
    """
    Fabs = 1. - (tau1 * np.exp(-0.5*((v - vabs)/sig1)**2.) * np.piecewise(v,[v<vabs, v>=vabs], [1., 0.]) +
            c1 * tau1 * np.exp(-0.5*((v - v2)/sig1)**2.) * np.piecewise(v,[v<v2, v>=v2], [1., 0.]) )
    """
    # The Emission Component
    Fems = 1. + (tau2 * np.exp(-0.5*((v - vems1)/sig2)**2.) + c2 * tau2 * np.exp(-0.5*((v - v4)/sig2)**2.) +
            tau3 * np.exp(-0.5*((v - vems2)/sig3)**2.) + c3 * tau3 * np.exp(-0.5*((v - v6)/sig3)**2.) )
    # The Systematic Component
    Fsys = 1. - (tau4 * np.exp(-0.5*(v/sig4)**2.) + tau4 * np.exp(-0.5*((v - dv)/sig4)**2.))
    # for the convloution
    spec_res = 2.52001483#2.542857
    muse_kernel = ((spec_res/1.25)/2.355)
    F = Fabs * Fems * Fsys
    g = Gaussian1DKernel(stddev=muse_kernel)
    F_conv = convolve(F, g, boundary='extend')
    return F_conv

def model_Mg_comps(v,vabs,vems1,vems2,sig1,sig2,sig3,sig4,tau1,tau2,tau3,tau4,c1,c2,c3):
    # [2797.084, 2799.326, 2804.346, 2807.975]   corrected to vaccum
    # These are the emission line centroids as measured through QfitsView and the global red shift measured from the w stacked OII spectrum
    lam_cen = [2796.351, 2803.528, 2797.084, 2799.326, 2804.346, 2808.975]
    # The first two lines are the absorption lines
    # The next 4 are the main and secondary emission lines
    '''
    vabs: Outflow Velocity of the absorption lines
    vems1: Outflow Velocity of the two main emission lines
    vems2: Outflow Velocity of the secondary emission lines
    tau1: Optical Depth of the absorption line 2796
    tau2: Height of the Main Emission line 2797
    tau3: Height of the Secondary Emission line 2808
    tau4: Optical Depth of the systematic component
    sig1: the standard deviation in the absorption velocity
    sig2: the standard deviation in the emission velocity vems1
    sig3: the standard deviation in the emission velocity vems2
    sig4: the standard deviation in the velocity of the systemtic component
    c1:   the line ratio between 2803/2796
    c2:   the line ratio between 2804/2797
    c3:   the line ratio between 2799/2808
    '''
    # Absorption Line Velocity
    dv = u.veldiff(lam_cen[1],lam_cen[0])
    v2 = vabs + dv  #u.veldiff(lam_cen[1],lam_cen[0])
    # Emission Lines Velocities
    #v3 = vems1 + u.veldiff(lam_cen[2],lam_cen[0])
    v4 = vems1 + dv
    #v5 = vems2 + u.veldiff(lam_cen[3],lam_cen[0])
    v6 = vems2 + dv#u.veldiff(lam_cen[5],lam_cen[2])
    # Absorption Component
    Fabs = 1. - (tau1 * np.exp(-0.5*((v - vabs)/sig1)**2.) * np.piecewise(v,[v< 0.0, v>= 0.0], [1., 0.]) +
            c1 * tau1 * np.exp(-0.5*((v - v2)/sig1)**2.) * np.piecewise(v,[v< (v2 - vabs), v>= (v2 - vabs)], [1., 0.]) )

    Fabs1 = 1. - tau1 * np.exp(-0.5*((v - vabs)/sig1)**2.) * np.piecewise(v,[v < 0.0, v >= 0.0], [1., 0.])
    Fabs2 = 1. - c1 * tau1 * np.exp(-0.5*((v - v2)/sig1)**2.) * np.piecewise(v,[v< (v2 - vabs), v >= (v2 - vabs)], [1., 0.])
    """
    Fabs = 1. - (tau1 * np.exp(-0.5*((v - vabs)/sig1)**2.) * np.piecewise(v,[v<vabs, v>=vabs], [1., 0.]) +
            c1 * tau1 * np.exp(-0.5*((v - v2)/sig1)**2.) * np.piecewise(v,[v<v2, v>=v2], [1., 0.]) )

    Fabs1 = 1. - tau1 * np.exp(-0.5*((v - vabs)/sig1)**2.) * np.piecewise(v,[v<vabs, v>=vabs], [1., 0.])
    Fabs2 = 1. - c1 * tau1 * np.exp(-0.5*((v - v2)/sig1)**2.) * np.piecewise(v,[v<v2, v>=v2], [1., 0.])
    """
    # The Emission Component
    Fems = 1. + (tau2 * np.exp(-0.5*((v - vems1)/sig2)**2.) + c2 * tau2 * np.exp(-0.5*((v - v4)/sig2)**2.) +
            tau3 * np.exp(-0.5*((v - vems2)/sig3)**2.) + c3 * tau3 * np.exp(-0.5*((v - v6)/sig3)**2.) )

    Fems1 = 1. + tau2 * np.exp(-0.5*((v - vems1)/sig2)**2.)
    Fems2 = 1. + c2 * tau2 * np.exp(-0.5*((v - v4)/sig2)**2.)
    Fems3 = 1. + tau3 * np.exp(-0.5*((v - vems2)/sig3)**2.)
    Fems4 = 1. + c3 * tau3 * np.exp(-0.5*((v - v6)/sig3)**2.)
    # The Systematic Component
    Fsys = 1. - (tau4 * np.exp(-0.5*(v/sig4)**2.) + tau4 * np.exp(-0.5*((v - dv)/sig4)**2.))

    '''
    Fabs1 = 1. - tau1 * np.exp(-0.5*((v - vabs)/sig1)**2.) * np.piecewise(v,[v<vabs, v>=vabs], [1., 0.])
    Fabs2 = 1. - c1 * tau1 * np.exp(-0.5*((v - v2)/sig1)**2.) * np.piecewise(v,[v<v2, v>=v2], [1., 0.])
    # The Emission Component
    Fems = 1. + (tau2 * np.exp(-0.5*((v - v3)/sig2)**2.) + c2 * tau2 * np.exp(-0.5*((v - v4)/sig2)**2.) +
            tau3 * np.exp(-0.5*((v - v6)/sig3)**2.) + c3 * tau3 * np.exp(-0.5*((v - v5)/sig3)**2.) )
    Fems1 = 1. + tau2 * np.exp(-0.5*((v - v3)/sig2)**2.)
    Fems2 = 1. + c2 * tau2 * np.exp(-0.5*((v - v4)/sig2)**2.)

    Fems3 = 1. + tau3 * np.exp(-0.5*((v - v6)/sig3)**2.)
    Fems4 = 1. + c3 * tau3 * np.exp(-0.5*((v - v5)/sig3)**2.)
    # The Systematic Component
    Fsys = 1. - (tau4 * np.exp(-0.5*(v/sig4)**2.) + tau4 * np.exp(-0.5*((v - dv_abs)/sig4)**2.))
    '''
    # for the convloution
    spec_res = 2.52001483#2.542857
    muse_kernel = ((spec_res/1.25)/2.355)
    F = Fabs * Fems * Fsys
    g = Gaussian1DKernel(stddev=muse_kernel)
    F_conv = convolve(F, g, boundary='extend')
    return F_conv, F, Fabs, Fabs1, Fabs2, Fems, Fems1, Fems2, Fems3, Fems4, Fsys

def Fout_2796(v, vabs, sig1, tau1):
    Fabs1 = 1. - tau1 * np.exp(-0.5*((v - vabs)/sig1)**2.) * np.piecewise(v,[v < 0.0, v>= 0.0], [1., 0.])
    return Fabs1

'''
def modelMg(v,v1,v3,tau1,tau2,c1,c2,c3,sigma1,sigma2):
    v2 = v1 + 768.4476528376828
    #F = 1 - tau1 * np.exp(-(v - v1)**2 / ( 2 * sigma1**2)) - c1 * tau1 * np.exp(-(v - v2)**2 / (2 * sigma1**2))
    Fabs = 1. - tau1 * np.exp(-(v - v1)**2. / (2. * sigma1**2.)) - c1 * tau1 * np.exp(-(v - v2)**2. / (2. * sigma1**2.))
    v4 = v3 + 745.2286364478183
    v5 = v3 + 1231.5324
    Fems = 1. + tau2 * np.exp(-(v - v3)**2. / (2. * sigma2**2.)) + c2 * tau2 * np.exp(-(v - v4)**2. / (2. * sigma2**2.)) + c3 * tau2 * np.exp(-(v - v5)**2. / (2. * sigma2**2.))
    muse_kernel = ((2.542857/1.25 )/ 2.355)
    F = Fabs*Fems
    g = Gaussian1DKernel(stddev=muse_kernel)
    fmodel = convolve(F, g, boundary='extend')
    return fmodel
'''


def model_OII(wave,z,tau,sigma):
    lam1 = 2470.97 #2324.21;
    lam1_obs = (1. + z) * lam1
    lam2 = 2471.09 #2325.40;
    lam2_obs = (1. + z) * lam2
    #wave = u.airtovac(wave)
    c = 1. # Get the Doublet ratio
    F = 1. + c* tau * np.exp(- (wave - lam1_obs)**2. / (2. * sigma**2.)) + tau * np.exp(-(wave - lam2_obs)**2. / (2. * sigma**2.))
    spec_res = 2.59392978 #2.68871980676
    muse_kernel = ((spec_res/1.25 )/ 2.355)#FWHM_avg = 2.50161030596 # Check the kernel for this part of the spectrum
    g = Gaussian1DKernel(stddev=muse_kernel)
    fmodel = convolve(F, g, boundary='extend')
    return fmodel

def model_CIII(wave,z,tau,sigma):
    lam1 = 1906.683 #2324.21;
    lam1_obs = (1. + z) * lam1
    lam2 = 1908.734 #2325.40;
    lam2_obs = (1. + z) * lam2
    c = 1.55 # Get the Doublet ratio
    F = 1. + c* tau * np.exp(- (wave - lam1_obs)**2. / (2. * sigma**2.)) + tau * np.exp(-(wave - lam2_obs)**2. / (2. * sigma**2.))
    spec_res =  2.83127078#2.60364004044
    muse_kernel = ((spec_res/1.25 )/ 2.355)#FWHM_avg = 2.50161030596 # Check the kernel for this part of the spectrum
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
