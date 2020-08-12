import numpy as np
import matplotlib.pyplot as plt
from astropy.convolution import convolve, Gaussian1DKernel
import musetools.util as u
from copy import deepcopy


###################################################################################
#                      Old Models
###################################################################################
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
    lam_cen = [2796.351, 2803.528]#, 2797.084, 2799.326, 2804.346, 2808.975]
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

def model_MgI(v, vabs, sig1, sig2, tau1, tau2):
    """
    This is the model for the MgI absorption line 2852 A
    The parameters for this model are
    vabs: centroid velocity of the outflowing component
    sig1: is the standard deviation in velocity for the outflowing component
    sig2: is the standard deviation in velocity for the systemic component
    tau1: is the optical depth for the outflowing component
    tau2: is the optical depth for the systemic  component
    """
    lam_cen = [2852.9634198]


    Fout = 1. - tau1 *  np.exp(-0.5*((v - vabs)/sig1)**2.) * np.piecewise(v,[v< 0.0, v>= 0.0], [1., 0.])
    Fsys = 1. - tau2 * np.exp(-0.5*(v / sig2)**2.)

    spec_res = 2.5176005677733357
    muse_kernel = ((spec_res/1.25)/2.355)
    F = np.asarray(Fout * Fsys)
    g = np.asarray(Gaussian1DKernel(stddev=muse_kernel) )
    F_conv = convolve(F, g, boundary='extend')

    return F_conv

def model_MgI_comps(v, vabs, sig1, sig2, tau1, tau2):
    """
    This is the model for the MgI absorption line 2852 A
    The parameters for this model are
    vabs: centroid velocity of the outflowing component
    sig1: is the standard deviation in velocity for the outflowing component
    sig2: is the standard deviation in velocity for the systemic component
    tau1: is the optical depth for the outflowing component
    tau2: is the optical depth for the systemic  component
    """
    lam_cen = [2852.9634198]


    Fout = 1. - tau1 *  np.exp(-0.5*((v - vabs)/sig1)**2.) * np.piecewise(v,[v< 0.0, v>= 0.0], [1., 0.])
    Fsys = 1. - tau2 * np.exp(-0.5*(v / sig2)**2.)

    spec_res = 2.5176005677733357
    muse_kernel = ((spec_res/1.25)/2.355)
    F = Fout * Fsys
    g = Gaussian1DKernel(stddev=muse_kernel)
    F_conv = convolve(F, g, boundary='extend')

    return F_conv, Fout, Fsys



def model_AlIII(v, vout, sig1, sig2, tau1, tau2):
    lam_cen = [1854.71829, 1862.79113]
    dv = u.veldiff(lam_cen[1],lam_cen[0])
    v2 = vout + dv
    Fout = 1. - tau1 * ( np.exp(-0.5*((v - vout)/sig1)**2.) * np.piecewise(v,[v< 0.0, v>= 0.0], [1., 0.]) + c1 *  np.exp(-0.5*((v - vout)/sig1)**2.) * np.piecewise(v,[v< (v2 - vout), v>= (v2-vout)], [1., 0.]) )
    Fsys  = 1. - tau2 * ( np.exp(-0.5*(v / sig2)**2.) + np.exp(-0.5*((v - dv)/ sig2)**2.) )
    spec_res, v_res = u.spectral_res(6250.4006373)
    muse_kernel = ((spec_res/1.25)/2.355)
    F = Fout * Fsys
    g = Gaussian1DKernel(stddev=muse_kernel)
    F_conv = convolve(F, g, boundary='extend')
    return F_conv

def model_AlIII_comps(v, vout, sig1, sig2, tau1, tau2):
    lam_cen = [1854.71829, 1862.79113]
    dv = u.veldiff(lam_cen[1],lam_cen[0])
    v2 = vout + dv
    Fout1 = 1. - tau1 * np.exp(-0.5*((v - vout)/sig1)**2.) * np.piecewise(v,[v< 0.0, v>= 0.0], [1., 0.])
    Fout2 = 1. - tau1 *  c1 *  np.exp(-0.5*((v - vout)/sig1)**2.) * np.piecewise(v,[v< (v2 - vout), v>= (v2-vout)], [1., 0.])
    Fsys  = 1. - tau2 * ( np.exp(-0.5*(v / sig2)**2.) + np.exp(-0.5*((v - dv)/ sig2)**2.) )
    spec_res, v_res = u.spectral_res(6250.4006373)
    muse_kernel = ((spec_res/1.25)/2.355)
    F = Fout * Fsys
    g = Gaussian1DKernel(stddev=muse_kernel)
    F_conv = convolve(F, g, boundary='extend')
    return F_conv, F, Fout1, Fout2, Fsys

def model_SiII_1526(v, vout, sig1, sig2, tau1, tau2):
    lam_cen = [1526.70698]
    Fout = 1. - tau1 *  np.exp(-0.5*((v - vout)/sig1)**2.) * np.piecewise(v,[v< 0.0, v>= 0.0], [1., 0.])
    Fsys = 1. - tau2 * np.exp(-0.5*(v / sig2)**2.)

    spec_res, v_res = u.spectral_res(5145.0022192999995)
    muse_kernel = ((spec_res/1.25)/2.355)
    F = Fout * Fsys
    g = Gaussian1DKernel(stddev=muse_kernel)
    F_conv = convolve(F, g, boundary='extend')

    return F_conv

def model_SiII_1526_comps(v, vout, sig1, sig2, tau1, tau2):
    lam_cen = [1526.70698]
    Fout = 1. - tau1 *  np.exp(-0.5*((v - vout)/sig1)**2.) * np.piecewise(v,[v< 0.0, v>= 0.0], [1., 0.])
    Fsys = 1. - tau2 * np.exp(-0.5*(v / sig2)**2.)

    spec_res, v_res = u.spectral_res(5145.0022192999995)
    muse_kernel = ((spec_res/1.25)/2.355)
    F = Fout * Fsys
    g = Gaussian1DKernel(stddev=muse_kernel)
    F_conv = convolve(F, g, boundary='extend')

    return F_conv, F, Fout, Fsys

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


def model_OII_comps(wave, z, tau, sigma):
    lam1 = 2470.97 #2324.21;
    lam1_obs = (1. + z) * lam1
    lam2 = 2471.09 #2325.40;
    lam2_obs = (1. + z) * lam2
    #wave = u.airtovac(wave)
    c = 1. # Get the Doublet ratio
    F = 1. + c * tau * np.exp(- (wave - lam1_obs)**2. / (2. * sigma**2.)) + tau * np.exp(-(wave - lam2_obs)**2. / (2. * sigma**2.))
    F1 = 1. + c * tau * np.exp(- (wave - lam1_obs)**2. / (2. * sigma**2.))
    F2 = 1. + tau * np.exp(-(wave - lam2_obs)**2. / (2. * sigma**2.))
    spec_res = 2.59392978 #2.68871980676
    muse_kernel = ((spec_res/1.25 )/ 2.355)#FWHM_avg = 2.50161030596 # Check the kernel for this part of the spectrum
    g = Gaussian1DKernel(stddev=muse_kernel)
    fmodel = convolve(F, g, boundary='extend')

    return fmodel, F1, F2

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


def model_CIII_comps(wave,z,tau,sigma):
    lam1 = 1906.683 #2324.21;
    lam1_obs = (1. + z) * lam1
    lam2 = 1908.734 #2325.40;
    lam2_obs = (1. + z) * lam2
    c = 1.55 # Get the Doublet ratio
    F = 1. + c * tau * np.exp(- (wave - lam1_obs)**2. / (2. * sigma**2.)) + tau * np.exp(-(wave - lam2_obs)**2. / (2. * sigma**2.))
    F1 = 1. + c * tau * np.exp(- (wave - lam1_obs)**2. / (2. * sigma**2.))
    F2 = 1. + tau * np.exp(-(wave - lam2_obs)**2. / (2. * sigma**2.))
    spec_res =  2.83127078#2.60364004044
    muse_kernel = ((spec_res/1.25 )/ 2.355)#FWHM_avg = 2.50161030596 # Check the kernel for this part of the spectrum
    g = Gaussian1DKernel(stddev=muse_kernel)
    fmodel = convolve(F, g, boundary='extend')
    return fmodel, F1, F2



##########################################################################################
"""New Modeling from Rubin et al 2014"""
#########################################################################################
def tau_lambda_fn(wrest, lam_cen, f0, lam_out, N_out, N_sys, b_D_out, b_D_sys):
    """
    Given inputs:
    wrest: rest-frame wavelength array
    lam_cen: central rest-frame wavelength of the transition
    f0: oscillator strength of the transition

    Free parameters:
    lam_out: the central wavelength of the gaussian descrbing the optical depth of the outflow component
    N_out: Column density of the outflowing component
    N_sys: Column density of the systemic component
    b_D_out: Doppler velocity width for the outflowing component
    b_D_sys: Doppler velocity width for the systemic component

    Outputs:
    tau_out: optical depth of the outflowing component
    tau_sys: optical depth of the systemic component
    """
    c = 299792.458  # Speed of light in km/s

    tau_out = (1.497e-15 * lam_cen * f0) * (N_out/ b_D_out) * np.exp(- ( (wrest - lam_out) / (lam_out* (b_D_out/c) ) )**2. ) * np.piecewise(wrest,[wrest < lam_cen, wrest >= lam_cen], [1., 0.]) # tau for the Outflow at 2586
    tau_sys = (1.497e-15 * lam_cen * f0) * (N_sys/ b_D_sys) * np.exp(- ( (wrest - lam_cen) / (lam_cen* (b_D_sys/c) ) )**2. )           # tau for the Systemic at 2586
    return tau_out, tau_sys


def model_FeII_wave_cf_sum_tau(wrest, lam_out, lam_ems, Cf_out, Cf_sys, b_D_out, b_D_sys, b_D_ems, logN_out, logN_sys, A, c_ems1, c_ems2, c_ems3, c_ems4):
    #Cf_sys = 1.0
    #Cf_out = 1.0
    z_r = 1.7039397365102
    N_out = 10.**logN_out
    N_sys = 10.**logN_sys
    c = 299792.458  # Speed of light in km/s

    lam_cen_ems = [2612.654, 2626.451, 2632.1081, 2365.552, 2396.355]
    lam_cen_abs = [2586.650, 2600.173, 2344.213, 2374.460, 2382.764]
    f0 =          [0.069125,   0.2394,   0.1142,   0.0313, 0.320]

    dlam_abs1 = lam_cen_abs[1] - lam_cen_abs[0]
    dlam_abs2 = lam_cen_abs[2] - lam_cen_abs[0]
    dlam_abs3 = lam_cen_abs[3] - lam_cen_abs[0]
    dlam_abs4 = lam_cen_abs[4] - lam_cen_abs[0]

    # (wrest, lam_cen, f0, lam_out, N_out, N_sys, b_D_out, b_D_sys)
    tau_out0, tau_sys0 = tau_lambda_fn(wrest, lam_cen_abs[0], f0[0], lam_out, N_out, N_sys, b_D_out, b_D_sys)
    tau_out1, tau_sys1 = tau_lambda_fn(wrest, lam_cen_abs[1], f0[1], lam_out + dlam_abs1, N_out, N_sys, b_D_out, b_D_sys)
    tau_out2, tau_sys2 = tau_lambda_fn(wrest, lam_cen_abs[2], f0[2], lam_out + dlam_abs2, N_out, N_sys, b_D_out, b_D_sys)
    tau_out3, tau_sys3 = tau_lambda_fn(wrest, lam_cen_abs[3], f0[3], lam_out + dlam_abs3, N_out, N_sys, b_D_out, b_D_sys)
    tau_out4, tau_sys4 = tau_lambda_fn(wrest, lam_cen_abs[4], f0[4], lam_out + dlam_abs4, N_out, N_sys, b_D_out, b_D_sys)


    F_out = 1.0 - Cf_out + Cf_out * np.exp(-(tau_out0 + tau_out1 + tau_out2 + tau_out3 + tau_out4) )



    F_sys = 1.0 - Cf_sys + Cf_sys * np.exp(-(tau_sys0 + tau_sys1 + tau_sys2 + tau_sys3 + tau_sys4) )


    dlam_ems0 = lam_cen_ems[0] - lam_cen_abs[0]   # The wavelength difference between Fe II emission 2612 and absorption 2586
    dlam_ems1 = lam_cen_ems[1] - lam_cen_abs[0]   # The wavelength difference between Fe II emission 2626 and absorption 2586
    dlam_ems2 = lam_cen_ems[2] - lam_cen_abs[0]   # The wavelength difference between Fe II emission 2632 and absorption 2586
    dlam_ems3 = lam_cen_ems[3] - lam_cen_abs[0]   # The wavelength difference between Fe II emission 2365 and absorption 2586
    dlam_ems4 = lam_cen_ems[4] - lam_cen_abs[0]   # The wavelength difference between Fe II emission 2396 and absorption 2586

    lam_ems0 = lam_ems + dlam_ems0
    lam_ems1 = lam_ems + dlam_ems1
    lam_ems2 = lam_ems + dlam_ems2
    lam_ems3 = lam_ems + dlam_ems3
    lam_ems4 = lam_ems + dlam_ems4

    F_ems = 1. + (A * np.exp(- ( (wrest - lam_ems0 ) / (lam_ems0* (b_D_ems/c)) )**2.)\
                 + c_ems1 * A * np.exp(- ( (wrest - lam_ems1) / (lam_ems1 * (b_D_ems/c)) )**2.)\
                 + c_ems2 * A * np.exp(- ( (wrest - lam_ems2) / (lam_ems2 * (b_D_ems/c)) )**2.)\
                 + c_ems3 * A * np.exp(- ( (wrest - lam_ems3) / (lam_ems3 * (b_D_ems/c)) )**2.)\
                 + c_ems4 * A * np.exp(- ( (wrest - lam_ems4) / (lam_ems4 * (b_D_ems/c)) )**2.) )

    spec_res, v_res = u.spectral_res( 2586.650 * (1. + z_r) )
    muse_kernel = ((spec_res/1.25)/2.355)
    g = Gaussian1DKernel(stddev=muse_kernel)
    F = F_out * F_sys * F_ems
    F_conv = convolve(F, g, boundary='extend')

    return F_conv

def model_FeII_wave_cf_sum_tau_comps(wrest, lam_out, lam_ems, Cf_out, Cf_sys, b_D_out, b_D_sys, b_D_ems, logN_out, logN_sys, A, c_ems1, c_ems2, c_ems3, c_ems4):
    #Cf_sys = 1.0
    #Cf_out = 1.0
    z_r = 1.7039397365102
    N_out = 10.**logN_out
    N_sys = 10.**logN_sys
    c = 299792.458  # Speed of light in km/s

    lam_cen_ems = [2612.654, 2626.451, 2632.1081, 2365.552, 2396.355]
    lam_cen_abs = [2586.650, 2600.173, 2344.213, 2374.460, 2382.764]
    f0 =          [0.069125,   0.2394,   0.1142,   0.0313, 0.320]

    dlam_abs1 = lam_cen_abs[1] - lam_cen_abs[0]
    dlam_abs2 = lam_cen_abs[2] - lam_cen_abs[0]
    dlam_abs3 = lam_cen_abs[3] - lam_cen_abs[0]
    dlam_abs4 = lam_cen_abs[4] - lam_cen_abs[0]

    # (wrest, lam_cen, f0, lam_out, N_out, N_sys, b_D_out, b_D_sys)
    tau_out0, tau_sys0 = tau_lambda_fn(wrest, lam_cen_abs[0], f0[0], lam_out, N_out, N_sys, b_D_out, b_D_sys)
    tau_out1, tau_sys1 = tau_lambda_fn(wrest, lam_cen_abs[1], f0[1], lam_out + dlam_abs1, N_out, N_sys, b_D_out, b_D_sys)
    tau_out2, tau_sys2 = tau_lambda_fn(wrest, lam_cen_abs[2], f0[2], lam_out + dlam_abs2, N_out, N_sys, b_D_out, b_D_sys)
    tau_out3, tau_sys3 = tau_lambda_fn(wrest, lam_cen_abs[3], f0[3], lam_out + dlam_abs3, N_out, N_sys, b_D_out, b_D_sys)
    tau_out4, tau_sys4 = tau_lambda_fn(wrest, lam_cen_abs[4], f0[4], lam_out + dlam_abs4, N_out, N_sys, b_D_out, b_D_sys)



    F_out = 1.0 - Cf_out + Cf_out * np.exp(-(tau_out0 + tau_out1 + tau_out2 + tau_out3 + tau_out4) )

    F_out0 = (1.0 + (- Cf_out + Cf_out * np.exp(-tau_out0) )  )
    F_out1 = (1.0 + (- Cf_out + Cf_out * np.exp(-tau_out1) )  )
    F_out2 = (1.0 + (- Cf_out + Cf_out * np.exp(-tau_out2) )  )
    F_out3 = (1.0 + (- Cf_out + Cf_out * np.exp(-tau_out3) )  )
    F_out4 = (1.0 + (- Cf_out + Cf_out * np.exp(-tau_out4) )  )


    F_sys = 1.0 - Cf_sys + Cf_sys * np.exp(-(tau_sys0 + tau_sys1 + tau_sys2 + tau_sys3 + tau_sys4) )



    dlam_ems0 = lam_cen_ems[0] - lam_cen_abs[0]   # The wavelength difference between Fe II emission 2612 and absorption 2586
    dlam_ems1 = lam_cen_ems[1] - lam_cen_abs[0]   # The wavelength difference between Fe II emission 2626 and absorption 2586
    dlam_ems2 = lam_cen_ems[2] - lam_cen_abs[0]   # The wavelength difference between Fe II emission 2632 and absorption 2586
    dlam_ems3 = lam_cen_ems[3] - lam_cen_abs[0]   # The wavelength difference between Fe II emission 2365 and absorption 2586
    dlam_ems4 = lam_cen_ems[4] - lam_cen_abs[0]   # The wavelength difference between Fe II emission 2396 and absorption 2586

    lam_ems0 = lam_ems + dlam_ems0
    lam_ems1 = lam_ems + dlam_ems1
    lam_ems2 = lam_ems + dlam_ems2
    lam_ems3 = lam_ems + dlam_ems3
    lam_ems4 = lam_ems + dlam_ems4

    F_ems = 1. + (A * np.exp(- ( (wrest - lam_ems0 ) / (lam_ems0* (b_D_ems/c)) )**2.)\
                 + c_ems1 * A * np.exp(- ( (wrest - lam_ems1) / (lam_ems1 * (b_D_ems/c)) )**2.)\
                 + c_ems2 * A * np.exp(- ( (wrest - lam_ems2) / (lam_ems2 * (b_D_ems/c)) )**2.)\
                 + c_ems3 * A * np.exp(- ( (wrest - lam_ems3) / (lam_ems3 * (b_D_ems/c)) )**2.)\
                 + c_ems4 * A * np.exp(- ( (wrest - lam_ems4) / (lam_ems4 * (b_D_ems/c)) )**2.) )

    F_ems0 = 1. + A * np.exp(- ( (wrest - lam_ems0 ) / (lam_ems0* (b_D_ems/c)) )**2.)
    F_ems1 = 1. + c_ems1 * A * np.exp(- ( (wrest - lam_ems1) / (lam_ems1 * (b_D_ems/c)) )**2.)
    F_ems2 = 1. + c_ems2 * A * np.exp(- ( (wrest - lam_ems2) / (lam_ems2 * (b_D_ems/c)) )**2.)
    F_ems3 = 1. + c_ems3 * A * np.exp(- ( (wrest - lam_ems3) / (lam_ems3 * (b_D_ems/c)) )**2.)
    F_ems4 = 1. + c_ems4 * A * np.exp(- ( (wrest - lam_ems4) / (lam_ems4 * (b_D_ems/c)) )**2.)

    spec_res, v_res = u.spectral_res( 2586.650 * (1. + z_r) )
    muse_kernel = ((spec_res/1.25)/2.355)
    g = Gaussian1DKernel(stddev=muse_kernel)
    F = F_out * F_sys * F_ems
    F_conv = convolve(F, g, boundary='extend')

    return F_conv, F, F_out, F_sys, F_ems, F_out0, F_out1, F_out2, F_out3, F_out4, F_ems0, F_ems1, F_ems2, F_ems3, F_ems4



########################### Mg II Models ############################################################
def model_MgII_wave_Cf(wrest, lam_out, lam_ems1, lam_ems2, b_D_out, b_D_sys, b_D_ems1, b_D_ems2, Cf_out, Cf_sys,logN_out, logN_sys, A_1, A_2, c_ems1, c_ems2):
    #Cf_sys = 1.0
    #Cf_out = 1.0
    c = 299792.458  # Speed of light in km/s
    z_r = 1.7039397365102
    N_out = 10.**logN_out
    N_sys = 10.**logN_sys

    lam_cen = [2796.351, 2803.528]
    f0 =      [0.6155, 0.3058]


    #dv = u.veldiff(lam_cen[1], lam_cen[0])
    dlam = lam_cen[1] - lam_cen[0]
    #vout1 = vout0 + dv

    tau_out0, tau_sys0 = tau_lambda_fn(wrest, lam_cen[0], f0[0], lam_out, N_out, N_sys, b_D_out, b_D_sys)
    tau_out1, tau_sys1 = tau_lambda_fn(wrest, lam_cen[1], f0[1], lam_out + dlam, N_out, N_sys, b_D_out, b_D_sys)

    #F_out = (1.0 + (- Cf_out + Cf_out * np.exp(-tau_out0) ) \
    #         + (- Cf_out + Cf_out * np.exp(-tau_out1) )  )

    F_out = (1.0 + (- Cf_out + Cf_out * np.exp(-(tau_out0 + tau_out1) ) ) )

    #F_sys =  ( 1.0 + (- Cf_sys + Cf_sys * np.exp(-tau_sys0))+ (- Cf_sys + Cf_sys * np.exp(-tau_sys1)) )
    F_sys = 1.0 + (- Cf_sys + Cf_sys * np.exp(- (tau_sys0 + tau_sys1) ))
    # The Free parameters for the Emission Component are:
    # A_1:       is the normalized Flux amplitude of the Mg II 2796 in the 1st emission component
    # A_2:      is the normalized Flux amplitude for the Mg II 2796 in the 2nd emission component
    # vems1:    is the emission velocity for Mg II 2nd emission component
    # vems2:    is the emission velocity for Mg II 2nd emission component
    # b_D_ems1: is the Doppler velocity width for the first emission component
    # b_D_ems2: is the Doppler velocity width for the second emission component
    # c_ems1: is the line ratio between Mg II emission lines 2803/2796 for the first emission component
    # c_ems2: is the line ratio between Mg II emission lines 2803/2796 for the second emission component

    lam_ems12 = lam_ems1 + dlam
    lam_ems22 = lam_ems2 + dlam

    F_ems = 1. + (A_1 * np.exp(- ( (wrest - lam_ems1 ) / (lam_ems1 * (b_D_ems1/c)) )**2. )\
                  + c_ems1 * A_1 * np.exp(- ( (wrest - lam_ems12 ) / (lam_ems12 * (b_D_ems1/c)) )**2. )\
                  + A_2 * np.exp(- ( (wrest - lam_ems2 ) / (lam_ems2 * (b_D_ems2/c)) )**2.)\
                  + c_ems2 * A_2 * np.exp(- ( (wrest - lam_ems22 ) / (lam_ems22 * (b_D_ems2/c)) )**2.))


    spec_res, v_res = u.spectral_res( lam_cen[0] * (1. + z_r) )
    muse_kernel = ((spec_res/1.25)/2.355)
    g = Gaussian1DKernel(stddev=muse_kernel)
    F = F_out * F_sys * F_ems
    F_conv = convolve(F, g, boundary='extend')
    return F_conv


def model_MgII_wave_Cf_comps(wrest, lam_out, lam_ems1, lam_ems2, b_D_out, b_D_sys, b_D_ems1, b_D_ems2, Cf_out, Cf_sys, logN_out, logN_sys, A_1, A_2, c_ems1, c_ems2):
    #Cf_sys = 1.0
    #Cf_out = 1.0
    c = 299792.458  # Speed of light in km/s
    z_r = 1.7039397365102
    N_out = 10.**logN_out
    N_sys = 10.**logN_sys

    lam_cen = [2796.351, 2803.528]
    f0 =      [0.6155, 0.3058]


    #dv = u.veldiff(lam_cen[1], lam_cen[0])
    dlam = lam_cen[1] - lam_cen[0]
    #vout1 = vout0 + dv

    tau_out0, tau_sys0 = tau_lambda_fn(wrest, lam_cen[0], f0[0], lam_out, N_out, N_sys, b_D_out, b_D_sys)
    tau_out1, tau_sys1 = tau_lambda_fn(wrest, lam_cen[1], f0[1], lam_out + dlam, N_out, N_sys, b_D_out, b_D_sys)


    F_out = (1.0 + (- Cf_out + Cf_out * np.exp(-(tau_out0 + tau_out1) ) ) )
    #F_out = (1.0 + (- Cf_out + Cf_out * np.exp(-tau_out0) ) \
    #         + (- Cf_out + Cf_out * np.exp(-tau_out1) )  )
    F_out0 = 1.0 + (- Cf_out + Cf_out * np.exp(-tau_out0) )
    F_out1 = 1.0 + (- Cf_out + Cf_out * np.exp(-tau_out1) )

    #F_sys =  ( 1.0 + (- Cf_sys + Cf_sys * np.exp(-tau_sys0))+ (- Cf_sys + Cf_sys * np.exp(-tau_sys1)) )
    F_sys = 1.0 + (- Cf_sys + Cf_sys * np.exp(- (tau_sys0 + tau_sys1) ))
    # The Free parameters for the Emission Component are:
    # A_1:       is the normalized Flux amplitude of the Mg II 2796 in the 1st emission component
    # A_2:      is the normalized Flux amplitude for the Mg II 2796 in the 2nd emission component
    # vems1:    is the emission velocity for Mg II 2nd emission component
    # vems2:    is the emission velocity for Mg II 2nd emission component
    # b_D_ems1: is the Doppler velocity width for the first emission component
    # b_D_ems2: is the Doppler velocity width for the second emission component
    # c_ems1: is the line ratio between Mg II emission lines 2803/2796 for the first emission component
    # c_ems2: is the line ratio between Mg II emission lines 2803/2796 for the second emission component

    lam_ems12 = lam_ems1 + dlam
    lam_ems22 = lam_ems2 + dlam

    F_ems = 1. + (A_1 * np.exp(- ( (wrest - lam_ems1 ) / (lam_ems1 * (b_D_ems1/c)) )**2. )\
                  + c_ems1 * A_1 * np.exp(- ( (wrest - lam_ems12 ) / (lam_ems12 * (b_D_ems1/c)) )**2. )\
                  + A_2 * np.exp(- ( (wrest - lam_ems2 ) / (lam_ems2 * (b_D_ems2/c)) )**2.)\
                  + c_ems2 * A_2 * np.exp(- ( (wrest - lam_ems22 ) / (lam_ems22 * (b_D_ems2/c)) )**2.))

    F_ems1 = 1. + (A_1 * np.exp(- ( (wrest - lam_ems1 ) / (lam_ems1 * (b_D_ems1/c)) )**2. )\
                  + c_ems1 * A_1 * np.exp(- ( (wrest - lam_ems12 ) / (lam_ems12 * (b_D_ems1/c)) )**2. ) )

    F_ems2 = 1. + (A_2 * np.exp(- ( (wrest - lam_ems2 ) / (lam_ems2 * (b_D_ems2/c)) )**2.)\
                    + c_ems2 * A_2 * np.exp(- ( (wrest - lam_ems22 ) / (lam_ems22 * (b_D_ems2/c)) )**2.))

    F_ems11 = 1. + (A_1 * np.exp(- ( (wrest - lam_ems1 ) / (lam_ems1 * (b_D_ems1/c)) )**2. ) )
    F_ems12 = 1. + c_ems1 * A_1 * np.exp(- ( (wrest - lam_ems12 ) / (lam_ems12 * (b_D_ems1/c)) )**2. )
    F_ems21 = 1. + A_2 * np.exp(- ( (wrest - lam_ems2 ) / (lam_ems2 * (b_D_ems2/c)) )**2.)
    F_ems22 = 1. + c_ems2 * A_2 * np.exp(- ( (wrest - lam_ems22 ) / (lam_ems22 * (b_D_ems2/c)) )**2.)


    spec_res, v_res = u.spectral_res( lam_cen[0] * (1. + z_r) )
    muse_kernel = ((spec_res/1.25)/2.355)
    g = Gaussian1DKernel(stddev=muse_kernel)
    F = F_out * F_sys * F_ems
    F_conv = convolve(F, g, boundary='extend')
    return F_conv, F, F_out, F_sys, F_ems, F_out0, F_out1, F_ems1, F_ems2, F_ems11, F_ems12, F_ems21, F_ems22



###################################################################################################################
""" Same as those from Rubin et al. 2014 but we keep Cf_sys = 1"""
####################################################################################################################
def model_FeII_wave_full(wrest, lam_out, lam_ems, Cf_out, b_D_out, b_D_sys, b_D_ems, logN_out, logN_sys, A, c_ems1, c_ems2, c_ems3, c_ems4):
    """
    Input:
    wrest: restframe wavelength array.

    Parameters of the model:
    lam_out: central wavelength of the outflow component.
    lam_ems: central wavelength of the emission component.
    Cf_out: The covering fraction of the outflow component.
    b_D_out: Doppler velocity parameter for the outflow component.
    b_D_sys: Doppler velocity parameter for the systemic component.
    b_D_ems: Doppler velocity parameter for the emission component.
    logN_out: log10 of the column density of the outflow component.
    logN_sys: log10 of the column density of the systemic component.
    A: Flux amplitude of the emission compnent.
    c_ems1: is the line ratio between the Fe II emission lines 2626 and 2612.
    c_ems2: is the line ratio between the Fe II emission lines 2632 and 2612.
    c_ems3: is the line ratio between the Fe II emission lines 2365 and 2612.
    c_ems4: is the line ratio between the Fe II emission lines 2396 and 2612.

    Output:
    F_conv: the full convolved final profile.
    """

    Cf_sys = 1.0
    #Cf_out = 1.0
    z_r = 1.703974047833502 # redshift of the galaxy from the stacked spectrum.
    N_out = 10.**logN_out
    N_sys = 10.**logN_sys
    c = 299792.458  # Speed of light in km/s

    lam_cen_ems = [2612.654, 2626.451, 2632.1081, 2365.552, 2396.355]
    lam_cen_abs = [2586.650, 2600.173, 2344.213, 2374.460, 2382.764, 2249.8768, 2260.7805]
    f0 =          [0.069125,   0.2394,   0.1142,   0.0313,    0.320, 0.001821, 0.00244]

    vel = u.veldiff(wrest, lam_cen_abs[0])

    dlam_abs1 = lam_cen_abs[1] - lam_cen_abs[0]
    dlam_abs2 = lam_cen_abs[2] - lam_cen_abs[0]
    dlam_abs3 = lam_cen_abs[3] - lam_cen_abs[0]
    dlam_abs4 = lam_cen_abs[4] - lam_cen_abs[0]
    dlam_abs5 = lam_cen_abs[5] - lam_cen_abs[0]
    dlam_abs6 = lam_cen_abs[6] - lam_cen_abs[0]

    # (wrest, lam_cen, f0, lam_out, N_out, N_sys, b_D_out, b_D_sys)
    tau_out0, tau_sys0 = tau_lambda_fn(wrest, lam_cen_abs[0], f0[0], lam_out, N_out, N_sys, b_D_out, b_D_sys)
    tau_out1, tau_sys1 = tau_lambda_fn(wrest, lam_cen_abs[1], f0[1], lam_out + dlam_abs1, N_out, N_sys, b_D_out, b_D_sys)
    tau_out2, tau_sys2 = tau_lambda_fn(wrest, lam_cen_abs[2], f0[2], lam_out + dlam_abs2, N_out, N_sys, b_D_out, b_D_sys)
    tau_out3, tau_sys3 = tau_lambda_fn(wrest, lam_cen_abs[3], f0[3], lam_out + dlam_abs3, N_out, N_sys, b_D_out, b_D_sys)
    tau_out4, tau_sys4 = tau_lambda_fn(wrest, lam_cen_abs[4], f0[4], lam_out + dlam_abs4, N_out, N_sys, b_D_out, b_D_sys)
    tau_out5, tau_sys5 = tau_lambda_fn(wrest, lam_cen_abs[5], f0[5], lam_out + dlam_abs5, N_out, N_sys, b_D_out, b_D_sys)
    tau_out6, tau_sys6 = tau_lambda_fn(wrest, lam_cen_abs[6], f0[6], lam_out + dlam_abs6, N_out, N_sys, b_D_out, b_D_sys)



    F_out = 1.0 - Cf_out + Cf_out * np.exp(-(tau_out0 + tau_out1 + tau_out2 + tau_out3 + tau_out4 + tau_out5 + tau_out6) )



    F_sys = 1.0 - Cf_sys + Cf_sys * np.exp(-(tau_sys0 + tau_sys1 + tau_sys2 + tau_sys3 + tau_sys4 + tau_out5 + tau_out6) )


    dlam_ems0 = lam_cen_ems[0] - lam_cen_abs[0]   # The wavelength difference between Fe II emission 2612 and absorption 2586
    dlam_ems1 = lam_cen_ems[1] - lam_cen_abs[0]   # The wavelength difference between Fe II emission 2626 and absorption 2586
    dlam_ems2 = lam_cen_ems[2] - lam_cen_abs[0]   # The wavelength difference between Fe II emission 2632 and absorption 2586
    dlam_ems3 = lam_cen_ems[3] - lam_cen_abs[0]   # The wavelength difference between Fe II emission 2365 and absorption 2586
    dlam_ems4 = lam_cen_ems[4] - lam_cen_abs[0]   # The wavelength difference between Fe II emission 2396 and absorption 2586

    lam_ems0 = lam_ems + dlam_ems0
    lam_ems1 = lam_ems + dlam_ems1
    lam_ems2 = lam_ems + dlam_ems2
    lam_ems3 = lam_ems + dlam_ems3
    lam_ems4 = lam_ems + dlam_ems4

    F_ems = 1. + (A * np.exp(- ( (wrest - lam_ems0 ) / (lam_ems0* (b_D_ems/c)) )**2.)\
                 + c_ems1 * A * np.exp(- ( (wrest - lam_ems1) / (lam_ems1 * (b_D_ems/c)) )**2.)\
                 + c_ems2 * A * np.exp(- ( (wrest - lam_ems2) / (lam_ems2 * (b_D_ems/c)) )**2.)\
                 + c_ems3 * A * np.exp(- ( (wrest - lam_ems3) / (lam_ems3 * (b_D_ems/c)) )**2.)\
                 + c_ems4 * A * np.exp(- ( (wrest - lam_ems4) / (lam_ems4 * (b_D_ems/c)) )**2.) )


    spec_res1, v_res1 = u.spectral_res( lam_cen_abs[-1] * (1. + z_r) )
    spec_res2, v_res2 = u.spectral_res( lam_cen_abs[3] * (1. + z_r) )
    spec_res3, v_res3 = u.spectral_res( lam_cen_abs[1] * (1. + z_r) )

    F = F_out * F_sys * F_ems

    q1 = np.where( vel > u.veldiff(2270., lam_cen_abs[0]))
    q2 = np.where( (vel < u.veldiff(2330., lam_cen_abs[0])) | (vel > u.veldiff(2400., lam_cen_abs[0])) )
    q3 = np.where(vel < u.veldiff(2575., lam_cen_abs[0]))

    F_1 = deepcopy(F);  F_2 = deepcopy(F); F_3 = deepcopy(F)
    F_1[q1] = 1.0; F_2[q2] = 1.0; F_3[q3] = 1.0
    # convolving the model
    muse_kernel1 = ((spec_res1/1.25) / 2.355);
    g1 = Gaussian1DKernel(stddev=muse_kernel1);
    F_conv1 = convolve(F_1, g1, boundary='extend');

    muse_kernel2 = ((spec_res2/1.25) / 2.355)
    g2 = Gaussian1DKernel(stddev=muse_kernel2)
    F_conv2 = convolve(F_2, g2, boundary='extend')

    muse_kernel3 = ((spec_res3/1.25) / 2.355)
    g3 = Gaussian1DKernel(stddev=muse_kernel3)
    F_conv3 = convolve(F_3, g3, boundary='extend')


    F_conv = F_conv1 * F_conv2 * F_conv3

    return F_conv




def model_FeII_wave_full_comps(wrest, lam_out, lam_ems, Cf_out, b_D_out, b_D_sys, b_D_ems, logN_out, logN_sys, A, c_ems1, c_ems2, c_ems3, c_ems4):
    """
    Input:
    wrest: restframe wavelength array.

    Parameters of the model:
    lam_out: central wavelength of the outflow component.
    lam_ems: central wavelength of the emission component.
    Cf_out: The covering fraction of the outflow component.
    b_D_out: Doppler velocity parameter for the outflow component.
    b_D_sys: Doppler velocity parameter for the systemic component.
    b_D_ems: Doppler velocity parameter for the emission component.
    logN_out: log10 of the column density of the outflow component.
    logN_sys: log10 of the column density of the systemic component.
    A: Flux amplitude of the emission compnent.
    c_ems1: is the line ratio between the Fe II emission lines 2626 and 2612.
    c_ems2: is the line ratio between the Fe II emission lines 2632 and 2612.
    c_ems3: is the line ratio between the Fe II emission lines 2365 and 2612.
    c_ems4: is the line ratio between the Fe II emission lines 2396 and 2612.

    Output:
    The output is a dictionaray that contains the flux profiles:
    output['F_conv']: Final convolved flux profile.
    output['F_unconv']: Final unconvolved flux profile.
    output['F_out']: Full profile for the outflow component.
    output['F_ems']: Full profile for the emission component.
    output['F_sys']: Full profile for the systemic component.

    For the indvidual lines profiles:
    Use the integer central wave lengthes without rounding in lam_cen_abs for the outflow: e.g. output['F+lam_cen_abs[i]+_out']
    Use the integer central wave lengthes without rounding in lam_cen_abs for the outflow: e.g. output['F+lam_cen_ems[i]+_out']
    """
    Cf_sys = 1.0
    #Cf_out = 1.0
    z_r = 1.703974047833502 #1.7039397365102
    N_out = 10.**logN_out
    N_sys = 10.**logN_sys
    c = 299792.458  # Speed of light in km/s



    output = {}

    lam_cen_ems = [2612.654, 2626.451, 2632.1081, 2365.552, 2396.355]
    lam_cen_abs = [2586.650, 2600.173, 2344.213, 2374.460, 2382.764, 2249.8768, 2260.7805]
    f0 =          [0.069125,   0.2394,   0.1142,   0.0313,    0.320, 0.001821, 0.00244]

    vel = u.veldiff(wrest, lam_cen_abs[0])

    dlam_abs1 = lam_cen_abs[1] - lam_cen_abs[0]
    dlam_abs2 = lam_cen_abs[2] - lam_cen_abs[0]
    dlam_abs3 = lam_cen_abs[3] - lam_cen_abs[0]
    dlam_abs4 = lam_cen_abs[4] - lam_cen_abs[0]
    dlam_abs5 = lam_cen_abs[5] - lam_cen_abs[0]
    dlam_abs6 = lam_cen_abs[6] - lam_cen_abs[0]

    # (wrest, lam_cen, f0, lam_out, N_out, N_sys, b_D_out, b_D_sys)
    tau_out0, tau_sys0 = tau_lambda_fn(wrest, lam_cen_abs[0], f0[0], lam_out, N_out, N_sys, b_D_out, b_D_sys)
    tau_out1, tau_sys1 = tau_lambda_fn(wrest, lam_cen_abs[1], f0[1], lam_out + dlam_abs1, N_out, N_sys, b_D_out, b_D_sys)
    tau_out2, tau_sys2 = tau_lambda_fn(wrest, lam_cen_abs[2], f0[2], lam_out + dlam_abs2, N_out, N_sys, b_D_out, b_D_sys)
    tau_out3, tau_sys3 = tau_lambda_fn(wrest, lam_cen_abs[3], f0[3], lam_out + dlam_abs3, N_out, N_sys, b_D_out, b_D_sys)
    tau_out4, tau_sys4 = tau_lambda_fn(wrest, lam_cen_abs[4], f0[4], lam_out + dlam_abs4, N_out, N_sys, b_D_out, b_D_sys)
    tau_out5, tau_sys5 = tau_lambda_fn(wrest, lam_cen_abs[5], f0[5], lam_out + dlam_abs5, N_out, N_sys, b_D_out, b_D_sys)
    tau_out6, tau_sys6 = tau_lambda_fn(wrest, lam_cen_abs[6], f0[6], lam_out + dlam_abs6, N_out, N_sys, b_D_out, b_D_sys)



    F_out = 1.0 - Cf_out + Cf_out * np.exp(-(tau_out0 + tau_out1 + tau_out2 + tau_out3 + tau_out4 + tau_out5 + tau_out6) )
    output['F2586_out'] = (1.0 + (- Cf_out + Cf_out * np.exp(-tau_out0) )  )
    output['F2600_out'] = (1.0 + (- Cf_out + Cf_out * np.exp(-tau_out1) )  )
    output['F2344_out'] = (1.0 + (- Cf_out + Cf_out * np.exp(-tau_out2) )  )
    output['F2374_out'] = (1.0 + (- Cf_out + Cf_out * np.exp(-tau_out3) )  )
    output['F2382_out'] = (1.0 + (- Cf_out + Cf_out * np.exp(-tau_out4) )  )
    output['F2249_out'] = (1.0 + (- Cf_out + Cf_out * np.exp(-tau_out5) )  )
    output['F2260_out'] = (1.0 + (- Cf_out + Cf_out * np.exp(-tau_out6) )  )

    F_sys = 1.0 - Cf_sys + Cf_sys * np.exp(-(tau_sys0 + tau_sys1 + tau_sys2 + tau_sys3 + tau_sys4 + tau_out5 + tau_out6) )

    dlam_ems0 = lam_cen_ems[0] - lam_cen_abs[0]   # The wavelength difference between Fe II emission 2612 and absorption 2586
    dlam_ems1 = lam_cen_ems[1] - lam_cen_abs[0]   # The wavelength difference between Fe II emission 2626 and absorption 2586
    dlam_ems2 = lam_cen_ems[2] - lam_cen_abs[0]   # The wavelength difference between Fe II emission 2632 and absorption 2586
    dlam_ems3 = lam_cen_ems[3] - lam_cen_abs[0]   # The wavelength difference between Fe II emission 2365 and absorption 2586
    dlam_ems4 = lam_cen_ems[4] - lam_cen_abs[0]   # The wavelength difference between Fe II emission 2396 and absorption 2586

    lam_ems0 = lam_ems + dlam_ems0
    lam_ems1 = lam_ems + dlam_ems1
    lam_ems2 = lam_ems + dlam_ems2
    lam_ems3 = lam_ems + dlam_ems3
    lam_ems4 = lam_ems + dlam_ems4

    F_ems = 1. + (A * np.exp(- ( (wrest - lam_ems0 ) / (lam_ems0* (b_D_ems/c)) )**2.)\
                 + c_ems1 * A * np.exp(- ( (wrest - lam_ems1) / (lam_ems1 * (b_D_ems/c)) )**2.)\
                 + c_ems2 * A * np.exp(- ( (wrest - lam_ems2) / (lam_ems2 * (b_D_ems/c)) )**2.)\
                 + c_ems3 * A * np.exp(- ( (wrest - lam_ems3) / (lam_ems3 * (b_D_ems/c)) )**2.)\
                 + c_ems4 * A * np.exp(- ( (wrest - lam_ems4) / (lam_ems4 * (b_D_ems/c)) )**2.) )

    output['F2612_ems'] = 1. + A * np.exp(- ( (wrest - lam_ems0 ) / (lam_ems0* (b_D_ems/c)) )**2.)
    output['F2626_ems'] = 1. + c_ems1 * A * np.exp(- ( (wrest - lam_ems1) / (lam_ems1 * (b_D_ems/c)) )**2.)
    output['F2632_ems'] = 1. + c_ems2 * A * np.exp(- ( (wrest - lam_ems2) / (lam_ems2 * (b_D_ems/c)) )**2.)
    output['F2365_ems'] = 1. + c_ems3 * A * np.exp(- ( (wrest - lam_ems3) / (lam_ems3 * (b_D_ems/c)) )**2.)
    output['F2396_ems'] = 1. + c_ems4 * A * np.exp(- ( (wrest - lam_ems4) / (lam_ems4 * (b_D_ems/c)) )**2.)

    output['F_out'] = F_out; output['F_ems'] = F_ems; output['F_sys'] = F_sys;

    spec_res1, v_res1 = u.spectral_res( lam_cen_abs[-1] * (1. + z_r) )
    spec_res2, v_res2 = u.spectral_res( lam_cen_abs[3] * (1. + z_r) )
    spec_res3, v_res3 = u.spectral_res( lam_cen_abs[1] * (1. + z_r) )

    F = F_out * F_sys * F_ems
    output['F_unconv'] = F

    q1 = np.where( vel > u.veldiff(2270., lam_cen_abs[0]))
    q2 = np.where( (vel < u.veldiff(2330., lam_cen_abs[0])) | (vel > u.veldiff(2400., lam_cen_abs[0])) )
    q3 = np.where(vel < u.veldiff(2575., lam_cen_abs[0]))

    F_1 = deepcopy(F);  F_2 = deepcopy(F); F_3 = deepcopy(F)
    F_1[q1] = 1.0; F_2[q2] = 1.0; F_3[q3] = 1.0

    # convolving the model
    muse_kernel1 = ((spec_res1/1.25) / 2.355);
    g1 = Gaussian1DKernel(stddev=muse_kernel1);
    F_conv1 = convolve(F_1, g1, boundary='extend');

    muse_kernel2 = ((spec_res2/1.25) / 2.355)
    g2 = Gaussian1DKernel(stddev=muse_kernel2)
    F_conv2 = convolve(F_2, g2, boundary='extend')

    muse_kernel3 = ((spec_res3/1.25) / 2.355)
    g3 = Gaussian1DKernel(stddev=muse_kernel3)
    F_conv3 = convolve(F_3, g3, boundary='extend')


    F_conv = F_conv1 * F_conv2 * F_conv3
    output['F_conv'] = F_conv


    return output



#####################################################################################################################
#         The Fe II model for the data from MAGE
#####################################################################################################################

def model_FeII_wave_MagE(wrest, lam_out, lam_ems, Cf_out, b_D_out, b_D_sys, b_D_ems, logN_out, logN_sys, A, c_ems1, c_ems3, c_ems4):
    Cf_sys = 1.0
    #Cf_out = 1.0
    N_out = 10.**logN_out
    N_sys = 10.**logN_sys
    c = 299792.458  # Speed of light in km/s

    lam_cen_ems = [2612.654, 2626.451, 2632.1081, 2365.552, 2396.355]
    lam_cen_abs = [2586.650, 2600.173, 2344.213, 2374.460, 2382.764, 2249.877, 2260.781]
    f0 =          [0.069125,   0.2394,   0.1142,   0.0313,    0.320, 0.001821, 0.00244]

    vel = u.veldiff(wrest, lam_cen_abs[0])

    dlam_abs1 = lam_cen_abs[1] - lam_cen_abs[0]
    dlam_abs2 = lam_cen_abs[2] - lam_cen_abs[0]
    dlam_abs3 = lam_cen_abs[3] - lam_cen_abs[0]
    dlam_abs4 = lam_cen_abs[4] - lam_cen_abs[0]
    dlam_abs5 = lam_cen_abs[5] - lam_cen_abs[0]
    dlam_abs6 = lam_cen_abs[6] - lam_cen_abs[0]


    # (wrest, lam_cen, f0, lam_out, N_out, N_sys, b_D_out, b_D_sys)
    tau_out0, tau_sys0 = tau_lambda_fn(wrest, lam_cen_abs[0], f0[0], lam_out, N_out, N_sys, b_D_out, b_D_sys)
    tau_out1, tau_sys1 = tau_lambda_fn(wrest, lam_cen_abs[1], f0[1], lam_out + dlam_abs1, N_out, N_sys, b_D_out, b_D_sys)
    tau_out2, tau_sys2 = tau_lambda_fn(wrest, lam_cen_abs[2], f0[2], lam_out + dlam_abs2, N_out, N_sys, b_D_out, b_D_sys)
    tau_out3, tau_sys3 = tau_lambda_fn(wrest, lam_cen_abs[3], f0[3], lam_out + dlam_abs3, N_out, N_sys, b_D_out, b_D_sys)
    tau_out4, tau_sys4 = tau_lambda_fn(wrest, lam_cen_abs[4], f0[4], lam_out + dlam_abs4, N_out, N_sys, b_D_out, b_D_sys)
    tau_out5, tau_sys5 = tau_lambda_fn(wrest, lam_cen_abs[5], f0[5], lam_out + dlam_abs5, N_out, N_sys, b_D_out, b_D_sys)
    tau_out6, tau_sys6 = tau_lambda_fn(wrest, lam_cen_abs[6], f0[6], lam_out + dlam_abs6, N_out, N_sys, b_D_out, b_D_sys)



    F_out = 1.0 - Cf_out + Cf_out * np.exp(-(tau_out0 + tau_out1 + tau_out2 + tau_out3 + tau_out4 + tau_out5 + tau_out6) )


    F_sys = 1.0 - Cf_sys + Cf_sys * np.exp(-(tau_sys0 + tau_sys1 + tau_sys2 + tau_sys3 + tau_sys4 + tau_sys5 + tau_sys6) )


    dlam_ems0 = lam_cen_ems[0] - lam_cen_abs[0]   # The wavelength difference between Fe II emission 2612 and absorption 2586
    dlam_ems1 = lam_cen_ems[1] - lam_cen_abs[0]   # The wavelength difference between Fe II emission 2626 and absorption 2586
    #dlam_ems2 = lam_cen_ems[2] - lam_cen_abs[0]   # The wavelength difference between Fe II emission 2632 and absorption 2586
    dlam_ems3 = lam_cen_ems[3] - lam_cen_abs[0]   # The wavelength difference between Fe II emission 2365 and absorption 2586
    dlam_ems4 = lam_cen_ems[4] - lam_cen_abs[0]   # The wavelength difference between Fe II emission 2396 and absorption 2586

    lam_ems0 = lam_ems + dlam_ems0
    lam_ems1 = lam_ems + dlam_ems1
    #lam_ems2 = lam_ems + dlam_ems2
    lam_ems3 = lam_ems + dlam_ems3
    lam_ems4 = lam_ems + dlam_ems4

    F_ems = 1. + (A * np.exp(- ( (wrest - lam_ems0 ) / (lam_ems0* (b_D_ems/c)) )**2.)\
                 + c_ems1 * A * np.exp(- ( (wrest - lam_ems1) / (lam_ems1 * (b_D_ems/c)) )**2.)\
                 + c_ems3 * A * np.exp(- ( (wrest - lam_ems3) / (lam_ems3 * (b_D_ems/c)) )**2.)\
                 + c_ems4 * A * np.exp(- ( (wrest - lam_ems4) / (lam_ems4 * (b_D_ems/c)) )**2.) )
    #+ c_ems2 * A * np.exp(- ( (wrest - lam_ems2) / (lam_ems2 * (b_D_ems/c)) )**2.)\

    v_res = 108. # km/s
    mage_kernel = ((v_res/ (vel[2]-vel[1]) )/2.355)
    g = Gaussian1DKernel(stddev=mage_kernel)
    F = F_out * F_sys * F_ems
    F_conv = convolve(F, g, boundary='extend')

    return F_conv

def model_FeII_wave_MagE_comps(wrest, lam_out, lam_ems, Cf_out, b_D_out, b_D_sys, b_D_ems, logN_out, logN_sys, A, c_ems1, c_ems3, c_ems4):
    Cf_sys = 1.0
    #Cf_out = 1.0
    N_out = 10.**logN_out
    N_sys = 10.**logN_sys
    c = 299792.458  # Speed of light in km/s

    lam_cen_ems = [2612.654, 2626.451, 2632.1081, 2365.552, 2396.355]
    lam_cen_abs = [2586.650, 2600.173, 2344.213, 2374.460, 2382.764, 2249.877, 2260.781]
    f0 =          [0.069125,   0.2394,   0.1142,   0.0313,    0.320, 0.001821, 0.00244]

    vel = u.veldiff(wrest, lam_cen_abs[0])

    dlam_abs1 = lam_cen_abs[1] - lam_cen_abs[0]
    dlam_abs2 = lam_cen_abs[2] - lam_cen_abs[0]
    dlam_abs3 = lam_cen_abs[3] - lam_cen_abs[0]
    dlam_abs4 = lam_cen_abs[4] - lam_cen_abs[0]
    dlam_abs5 = lam_cen_abs[5] - lam_cen_abs[0]
    dlam_abs6 = lam_cen_abs[6] - lam_cen_abs[0]


    # (wrest, lam_cen, f0, lam_out, N_out, N_sys, b_D_out, b_D_sys)
    tau_out0, tau_sys0 = tau_lambda_fn(wrest, lam_cen_abs[0], f0[0], lam_out, N_out, N_sys, b_D_out, b_D_sys)
    tau_out1, tau_sys1 = tau_lambda_fn(wrest, lam_cen_abs[1], f0[1], lam_out + dlam_abs1, N_out, N_sys, b_D_out, b_D_sys)
    tau_out2, tau_sys2 = tau_lambda_fn(wrest, lam_cen_abs[2], f0[2], lam_out + dlam_abs2, N_out, N_sys, b_D_out, b_D_sys)
    tau_out3, tau_sys3 = tau_lambda_fn(wrest, lam_cen_abs[3], f0[3], lam_out + dlam_abs3, N_out, N_sys, b_D_out, b_D_sys)
    tau_out4, tau_sys4 = tau_lambda_fn(wrest, lam_cen_abs[4], f0[4], lam_out + dlam_abs4, N_out, N_sys, b_D_out, b_D_sys)
    tau_out5, tau_sys5 = tau_lambda_fn(wrest, lam_cen_abs[5], f0[5], lam_out + dlam_abs5, N_out, N_sys, b_D_out, b_D_sys)
    tau_out6, tau_sys6 = tau_lambda_fn(wrest, lam_cen_abs[6], f0[6], lam_out + dlam_abs6, N_out, N_sys, b_D_out, b_D_sys)

    output = {}

    F_out = 1.0 - Cf_out + Cf_out * np.exp(-(tau_out0 + tau_out1 + tau_out2 + tau_out3 + tau_out4 + tau_out5 + tau_out6) )
    output['F_out'] = F_out

    output['F2586_out'] = (1.0 + (- Cf_out + Cf_out * np.exp(-tau_out0) )  )
    output['F2600_out'] = (1.0 + (- Cf_out + Cf_out * np.exp(-tau_out1) )  )
    output['F2344_out'] = (1.0 + (- Cf_out + Cf_out * np.exp(-tau_out2) )  )
    output['F2374_out'] = (1.0 + (- Cf_out + Cf_out * np.exp(-tau_out3) )  )
    output['F2382_out'] = (1.0 + (- Cf_out + Cf_out * np.exp(-tau_out4) )  )
    output['F2249_out'] = (1.0 + (- Cf_out + Cf_out * np.exp(-tau_out5) )  )
    output['F2260_out'] = (1.0 + (- Cf_out + Cf_out * np.exp(-tau_out6) )  )


    F_sys = 1.0 - Cf_sys + Cf_sys * np.exp(-(tau_sys0 + tau_sys1 + tau_sys2 + tau_sys3 + tau_sys4 + tau_sys5 + tau_sys6) )
    output['F_sys'] = F_sys


    dlam_ems0 = lam_cen_ems[0] - lam_cen_abs[0]   # The wavelength difference between Fe II emission 2612 and absorption 2586
    dlam_ems1 = lam_cen_ems[1] - lam_cen_abs[0]   # The wavelength difference between Fe II emission 2626 and absorption 2586
    #dlam_ems2 = lam_cen_ems[2] - lam_cen_abs[0]   # The wavelength difference between Fe II emission 2632 and absorption 2586
    dlam_ems3 = lam_cen_ems[3] - lam_cen_abs[0]   # The wavelength difference between Fe II emission 2365 and absorption 2586
    dlam_ems4 = lam_cen_ems[4] - lam_cen_abs[0]   # The wavelength difference between Fe II emission 2396 and absorption 2586

    lam_ems0 = lam_ems + dlam_ems0
    lam_ems1 = lam_ems + dlam_ems1
    #lam_ems2 = lam_ems + dlam_ems2
    lam_ems3 = lam_ems + dlam_ems3
    lam_ems4 = lam_ems + dlam_ems4

    #     lam_cen_ems = [2612.654, 2626.451, 2632.1081, 2365.552, 2396.355]

    F_ems = 1. + (A * np.exp(- ( (wrest - lam_ems0 ) / (lam_ems0* (b_D_ems/c)) )**2.)\
                 + c_ems1 * A * np.exp(- ( (wrest - lam_ems1) / (lam_ems1 * (b_D_ems/c)) )**2.)\
                 + c_ems3 * A * np.exp(- ( (wrest - lam_ems3) / (lam_ems3 * (b_D_ems/c)) )**2.)\
                 + c_ems4 * A * np.exp(- ( (wrest - lam_ems4) / (lam_ems4 * (b_D_ems/c)) )**2.) )
    #+ c_ems2 * A * np.exp(- ( (wrest - lam_ems2) / (lam_ems2 * (b_D_ems/c)) )**2.)\
    output['F_ems'] = F_ems

    output['F2612_ems'] = 1. + A * np.exp(- ( (wrest - lam_ems0 ) / (lam_ems0* (b_D_ems/c)) )**2.)
    output['F2626_ems'] = 1. + c_ems1 * A * np.exp(- ( (wrest - lam_ems1) / (lam_ems1 * (b_D_ems/c)) )**2.)
    #F_ems2 = 1. + c_ems2 * A * np.exp(- ( (wrest - lam_ems2) / (lam_ems2 * (b_D_ems/c)) )**2.)
    output['F2365_ems'] = 1. + c_ems3 * A * np.exp(- ( (wrest - lam_ems3) / (lam_ems3 * (b_D_ems/c)) )**2.)
    output['F2396_ems'] = 1. + c_ems4 * A * np.exp(- ( (wrest - lam_ems4) / (lam_ems4 * (b_D_ems/c)) )**2.)

    v_res = 108. # km/s
    mage_kernel = ((v_res/ (vel[2]-vel[1]) )/2.355)
    g = Gaussian1DKernel(stddev=mage_kernel)
    F = F_out * F_sys * F_ems
    F_conv = convolve(F, g, boundary='extend')
    output['F_unconv'] = F; output['F_conv'] = F_conv

    return output

#########################################################################################################################
#     Mg II models
#########################################################################################################################
def model_MgII_wave_full(wrest, lam_out, lam_ems1, lam_ems2, b_D_out, b_D_sys, b_D_ems1, b_D_ems2, Cf_out,logN_out, logN_sys, A_1, A_2, c_ems1, c_ems2):
    """
    Input:
    wrest: restframe wavelength.

    The Free parameters for model are:
    lam_out: central wavelength of the outflow component.
    lam_ems1: central wavelength of the first emission component.
    lam_ems2: central wavelength of the second emission component.
    b_D_out: Doppler velocity parameter for the outflow component.
    b_D_sys: Doppler velocity parameter for the systemic component.
    b_D_ems1: Doppler velocity parameter for the 1st emission component.
    b_D_ems2: Doppler velocity parameter for the 2nd emission component.
    Cf_out: Covering fraction for the outflow component.
    logN_out: log10 of the column density of Mg II for the outflow component.
    logN_sys: log10 of the column density of Mg II for the outflow component.
    A_1: Normalized flux amplitude for the 1st emission component.
    A_2: Normalized flux amplitude for the 2nd emission component.
    c_ems1: the line ratio between Mg II resonant emission lines 2803 and 2796 for the 1st emission component.
    c_ems2: the line ratio between Mg II resonant emission lines 2803 and 2796 for the 1st emission component.


    Output:
    The output is a dictionary that contains the output compnents for the model using the keywords:
    F_out: The full outflow component for the Mg II absorption lines.
    F2796_out: The outflow component for the Mg II 2796 absorption line.
    F2803_out: The outflow component for the Mg II 2803 absorption line.
    F_sys: The full systemic component for the Mg II absprotion lines.
    F_ems: The full emission component for the Mg II resonant emission lines.
    F_ems1: The 1st emission component for the Mg II resonant emission lines.
    F_ems2: The 2nd emission component for the Mg II resonant emission lines.
    F2796_ems1: The Mg II 2796 emission line model in the 1st emission component.
    F2803_ems1: The Mg II 2803 emission line model in the 1st emission component.
    F2796_ems2: The Mg II 2796 emission line model in the 2nd emission component.
    F2803_ems2: The Mg II 2803 emission line model in the 2nd emission component.
    """
    Cf_sys = 1.0 # Covering fraction for the systemic component
    #Cf_out = 1.0
    c = 299792.458  # Speed of light in km/s
    z_r = 1.7039397365102
    N_out = 10.**logN_out
    N_sys = 10.**logN_sys

    lam_cen = [2796.351, 2803.528]
    f0 =      [0.6155, 0.3058]


    #dv = u.veldiff(lam_cen[1], lam_cen[0])
    dlam = lam_cen[1] - lam_cen[0]
    #vout1 = vout0 + dv

    tau_out0, tau_sys0 = tau_lambda_fn(wrest, lam_cen[0], f0[0], lam_out, N_out, N_sys, b_D_out, b_D_sys)
    tau_out1, tau_sys1 = tau_lambda_fn(wrest, lam_cen[1], f0[1], lam_out + dlam, N_out, N_sys, b_D_out, b_D_sys)

    #F_out = (1.0 + (- Cf_out + Cf_out * np.exp(-tau_out0) ) \
    #         + (- Cf_out + Cf_out * np.exp(-tau_out1) )  )

    F_out = (1.0 + (- Cf_out + Cf_out * np.exp(-(tau_out0 + tau_out1) ) ) )

    F_sys = 1.0 + (- Cf_sys + Cf_sys * np.exp(- (tau_sys0 + tau_sys1) ))

    lam_ems12 = lam_ems1 + dlam
    lam_ems22 = lam_ems2 + dlam

    F_ems = 1. + (A_1 * np.exp(- ( (wrest - lam_ems1 ) / (lam_ems1 * (b_D_ems1/c)) )**2. )\
                  + c_ems1 * A_1 * np.exp(- ( (wrest - lam_ems12 ) / (lam_ems12 * (b_D_ems1/c)) )**2. )\
                  + A_2 * np.exp(- ( (wrest - lam_ems2 ) / (lam_ems2 * (b_D_ems2/c)) )**2.)\
                  + c_ems2 * A_2 * np.exp(- ( (wrest - lam_ems22 ) / (lam_ems22 * (b_D_ems2/c)) )**2.))


    spec_res, v_res = u.spectral_res( lam_cen[0] * (1. + z_r) )
    muse_kernel = ((spec_res/1.25)/2.355)
    g = Gaussian1DKernel(stddev=muse_kernel)
    F = F_out * F_sys * F_ems
    F_conv = convolve(F, g, boundary='extend')
    return F_conv


def model_MgII_wave_full_comps(wrest, lam_out, lam_ems1, lam_ems2, b_D_out, b_D_sys, b_D_ems1, b_D_ems2, Cf_out, logN_out, logN_sys, A_1, A_2, c_ems1, c_ems2):
    """
    Input:
    wrest: restframe wavelength.

    The Free parameters for model are:
    lam_out: central wavelength of the outflow component.
    lam_ems1: central wavelength of the first emission component.
    lam_ems2: central wavelength of the second emission component.
    b_D_out: Doppler velocity parameter for the outflow component.
    b_D_sys: Doppler velocity parameter for the systemic component.
    b_D_ems1: Doppler velocity parameter for the 1st emission component.
    b_D_ems2: Doppler velocity parameter for the 2nd emission component.
    Cf_out: Covering fraction for the outflow component.
    logN_out: log10 of the column density of Mg II for the outflow component.
    logN_sys: log10 of the column density of Mg II for the outflow component.
    A_1: Normalized flux amplitude for the 1st emission component.
    A_2: Normalized flux amplitude for the 2nd emission component.
    c_ems1: the line ratio between Mg II resonant emission lines 2803 and 2796 for the 1st emission component.
    c_ems2: the line ratio between Mg II resonant emission lines 2803 and 2796 for the 1st emission component.


    Output:
    The output is a dictionary that contains the output compnents for the model using the keywords:
    F_out: The full outflow component for the Mg II absorption lines.
    F2796_out: The outflow component for the Mg II 2796 absorption line.
    F2803_out: The outflow component for the Mg II 2803 absorption line.
    F_sys: The full systemic component for the Mg II absprotion lines.
    F_ems: The full emission component for the Mg II resonant emission lines.
    F_ems1: The 1st emission component for the Mg II resonant emission lines.
    F_ems2: The 2nd emission component for the Mg II resonant emission lines.
    F2796_ems1: The Mg II 2796 emission line model in the 1st emission component.
    F2803_ems1: The Mg II 2803 emission line model in the 1st emission component.
    F2796_ems2: The Mg II 2796 emission line model in the 2nd emission component.
    F2803_ems2: The Mg II 2803 emission line model in the 2nd emission component.
    """

    Cf_sys = 1.0
    #Cf_out = 1.0
    c = 299792.458  # Speed of light in km/s
    z_r = 1.7039397365102
    N_out = 10.**logN_out
    N_sys = 10.**logN_sys

    lam_cen = [2796.351, 2803.528]
    f0 =      [0.6155, 0.3058]


    #dv = u.veldiff(lam_cen[1], lam_cen[0])
    dlam = lam_cen[1] - lam_cen[0]
    #vout1 = vout0 + dv

    tau_out0, tau_sys0 = tau_lambda_fn(wrest, lam_cen[0], f0[0], lam_out, N_out, N_sys, b_D_out, b_D_sys)
    tau_out1, tau_sys1 = tau_lambda_fn(wrest, lam_cen[1], f0[1], lam_out + dlam, N_out, N_sys, b_D_out, b_D_sys)

    output = {}

    F_out = (1.0 + (- Cf_out + Cf_out * np.exp(-(tau_out0 + tau_out1) ) ) )
    output["F_out"] = F_out

    output["F2796_out"] = 1.0 + (- Cf_out + Cf_out * np.exp(-tau_out0) )
    output["F2803_out"] = 1.0 + (- Cf_out + Cf_out * np.exp(-tau_out1) )

    #F_sys =  ( 1.0 + (- Cf_sys + Cf_sys * np.exp(-tau_sys0))+ (- Cf_sys + Cf_sys * np.exp(-tau_sys1)) )
    F_sys = 1.0 + (- Cf_sys + Cf_sys * np.exp(- (tau_sys0 + tau_sys1) ))
    output["F_sys"] = F_sys

    lam_ems12 = lam_ems1 + dlam
    lam_ems22 = lam_ems2 + dlam

    F_ems = 1. + (A_1 * np.exp(- ( (wrest - lam_ems1 ) / (lam_ems1 * (b_D_ems1/c)) )**2. )\
                  + c_ems1 * A_1 * np.exp(- ( (wrest - lam_ems12 ) / (lam_ems12 * (b_D_ems1/c)) )**2. )\
                  + A_2 * np.exp(- ( (wrest - lam_ems2 ) / (lam_ems2 * (b_D_ems2/c)) )**2.)\
                  + c_ems2 * A_2 * np.exp(- ( (wrest - lam_ems22 ) / (lam_ems22 * (b_D_ems2/c)) )**2.))

    output["F_ems"] = F_ems
    output["F_ems1"] = 1. + (A_1 * np.exp(- ( (wrest - lam_ems1 ) / (lam_ems1 * (b_D_ems1/c)) )**2. )\
                  + c_ems1 * A_1 * np.exp(- ( (wrest - lam_ems12 ) / (lam_ems12 * (b_D_ems1/c)) )**2. ) )

    output["F_ems2"] = 1. + (A_2 * np.exp(- ( (wrest - lam_ems2 ) / (lam_ems2 * (b_D_ems2/c)) )**2.)\
                    + c_ems2 * A_2 * np.exp(- ( (wrest - lam_ems22 ) / (lam_ems22 * (b_D_ems2/c)) )**2.))

    output["F2796_ems1"] = 1. + (A_1 * np.exp(- ( (wrest - lam_ems1 ) / (lam_ems1 * (b_D_ems1/c)) )**2. ) )
    output["F2803_ems1"] = 1. + c_ems1 * A_1 * np.exp(- ( (wrest - lam_ems12 ) / (lam_ems12 * (b_D_ems1/c)) )**2. )

    output["F2796_ems2"] = 1. + A_2 * np.exp(- ( (wrest - lam_ems2 ) / (lam_ems2 * (b_D_ems2/c)) )**2.)
    output["F2803_ems2"] = 1. + c_ems2 * A_2 * np.exp(- ( (wrest - lam_ems22 ) / (lam_ems22 * (b_D_ems2/c)) )**2.)


    spec_res, v_res = u.spectral_res( lam_cen[0] * (1. + z_r) )
    muse_kernel = ((spec_res/1.25)/2.355)
    g = Gaussian1DKernel(stddev=muse_kernel)
    F = F_out * F_sys * F_ems
    output["F_unconv"] = F
    F_conv = convolve(F, g, boundary='extend')
    output["F_conv"] = F_conv
    return output


def model_MgII_wave_MAGE_full(wrest, lam_out, lam_ems1, lam_ems2, b_D_out, b_D_sys, b_D_ems1, b_D_ems2, Cf_out,logN_out, logN_sys, A_1, A_2, c_ems1, c_ems2):
    """
    Input:
    wrest: restframe wavelength.

    The Free parameters for model are:
    lam_out: central wavelength of the outflow component.
    lam_ems1: central wavelength of the first emission component.
    lam_ems2: central wavelength of the second emission component.
    b_D_out: Doppler velocity parameter for the outflow component.
    b_D_sys: Doppler velocity parameter for the systemic component.
    b_D_ems1: Doppler velocity parameter for the 1st emission component.
    b_D_ems2: Doppler velocity parameter for the 2nd emission component.
    Cf_out: Covering fraction for the outflow component.
    logN_out: log10 of the column density of Mg II for the outflow component.
    logN_sys: log10 of the column density of Mg II for the outflow component.
    A_1: Normalized flux amplitude for the 1st emission component.
    A_2: Normalized flux amplitude for the 2nd emission component.
    c_ems1: the line ratio between Mg II resonant emission lines 2803 and 2796 for the 1st emission component.
    c_ems2: the line ratio between Mg II resonant emission lines 2803 and 2796 for the 1st emission component.


    Output:
    The output is a dictionary that contains the output compnents for the model using the keywords:
    F_out: The full outflow component for the Mg II absorption lines.
    F2796_out: The outflow component for the Mg II 2796 absorption line.
    F2803_out: The outflow component for the Mg II 2803 absorption line.
    F_sys: The full systemic component for the Mg II absprotion lines.
    F_ems: The full emission component for the Mg II resonant emission lines.
    F_ems1: The 1st emission component for the Mg II resonant emission lines.
    F_ems2: The 2nd emission component for the Mg II resonant emission lines.
    F2796_ems1: The Mg II 2796 emission line model in the 1st emission component.
    F2803_ems1: The Mg II 2803 emission line model in the 1st emission component.
    F2796_ems2: The Mg II 2796 emission line model in the 2nd emission component.
    F2803_ems2: The Mg II 2803 emission line model in the 2nd emission component.
    """
    Cf_sys = 1.0 # Covering fraction for the systemic component
    #Cf_out = 1.0
    c = 299792.458  # Speed of light in km/s
    z_r = 1.7039397365102
    N_out = 10.**logN_out
    N_sys = 10.**logN_sys

    lam_cen = [2796.351, 2803.528]
    f0 =      [0.6155, 0.3058]
    vel = u.veldiff(wrest, lam_cen[0])


    #dv = u.veldiff(lam_cen[1], lam_cen[0])
    dlam = lam_cen[1] - lam_cen[0]
    #vout1 = vout0 + dv

    tau_out0, tau_sys0 = tau_lambda_fn(wrest, lam_cen[0], f0[0], lam_out, N_out, N_sys, b_D_out, b_D_sys)
    tau_out1, tau_sys1 = tau_lambda_fn(wrest, lam_cen[1], f0[1], lam_out + dlam, N_out, N_sys, b_D_out, b_D_sys)

    #F_out = (1.0 + (- Cf_out + Cf_out * np.exp(-tau_out0) ) \
    #         + (- Cf_out + Cf_out * np.exp(-tau_out1) )  )

    F_out = (1.0 + (- Cf_out + Cf_out * np.exp(-(tau_out0 + tau_out1) ) ) )

    F_sys = 1.0 + (- Cf_sys + Cf_sys * np.exp(- (tau_sys0 + tau_sys1) ))

    lam_ems12 = lam_ems1 + dlam
    lam_ems22 = lam_ems2 + dlam

    F_ems = 1. + (A_1 * np.exp(- ( (wrest - lam_ems1 ) / (lam_ems1 * (b_D_ems1/c)) )**2. )\
                  + c_ems1 * A_1 * np.exp(- ( (wrest - lam_ems12 ) / (lam_ems12 * (b_D_ems1/c)) )**2. )\
                  + A_2 * np.exp(- ( (wrest - lam_ems2 ) / (lam_ems2 * (b_D_ems2/c)) )**2.)\
                  + c_ems2 * A_2 * np.exp(- ( (wrest - lam_ems22 ) / (lam_ems22 * (b_D_ems2/c)) )**2.))


    #spec_res, v_res = u.spectral_res( lam_cen[0] * (1. + z_r) )
    v_res = 108.
    mage_kernel = ((v_res/ (vel[1] - vel[0]) )/2.355)
    g = Gaussian1DKernel(stddev=mage_kernel)
    F = F_out * F_sys * F_ems
    F_conv = convolve(F, g, boundary='extend')
    return F_conv


def model_MgII_wave_MAGE_comps(wrest, lam_out, lam_ems1, lam_ems2, b_D_out, b_D_sys, b_D_ems1, b_D_ems2, Cf_out, logN_out, logN_sys, A_1, A_2, c_ems1, c_ems2):
    """
    Input:
    wrest: restframe wavelength.

    The Free parameters for model are:
    lam_out: central wavelength of the outflow component.
    lam_ems1: central wavelength of the first emission component.
    lam_ems2: central wavelength of the second emission component.
    b_D_out: Doppler velocity parameter for the outflow component.
    b_D_sys: Doppler velocity parameter for the systemic component.
    b_D_ems1: Doppler velocity parameter for the 1st emission component.
    b_D_ems2: Doppler velocity parameter for the 2nd emission component.
    Cf_out: Covering fraction for the outflow component.
    logN_out: log10 of the column density of Mg II for the outflow component.
    logN_sys: log10 of the column density of Mg II for the outflow component.
    A_1: Normalized flux amplitude for the 1st emission component.
    A_2: Normalized flux amplitude for the 2nd emission component.
    c_ems1: the line ratio between Mg II resonant emission lines 2803 and 2796 for the 1st emission component.
    c_ems2: the line ratio between Mg II resonant emission lines 2803 and 2796 for the 1st emission component.


    Output:
    The output is a dictionary that contains the output compnents for the model using the keywords:
    F_out: The full outflow component for the Mg II absorption lines.
    F2796_out: The outflow component for the Mg II 2796 absorption line.
    F2803_out: The outflow component for the Mg II 2803 absorption line.
    F_sys: The full systemic component for the Mg II absprotion lines.
    F_ems: The full emission component for the Mg II resonant emission lines.
    F_ems1: The 1st emission component for the Mg II resonant emission lines.
    F_ems2: The 2nd emission component for the Mg II resonant emission lines.
    F2796_ems1: The Mg II 2796 emission line model in the 1st emission component.
    F2803_ems1: The Mg II 2803 emission line model in the 1st emission component.
    F2796_ems2: The Mg II 2796 emission line model in the 2nd emission component.
    F2803_ems2: The Mg II 2803 emission line model in the 2nd emission component.
    """

    Cf_sys = 1.0
    #Cf_out = 1.0
    c = 299792.458  # Speed of light in km/s
    z_r = 1.7039397365102
    N_out = 10.**logN_out
    N_sys = 10.**logN_sys

    lam_cen = [2796.351, 2803.528]
    f0 =      [0.6155, 0.3058]
    wrest = np.asarray(wrest)
    vel = u.veldiff(wrest, lam_cen[0])


    #dv = u.veldiff(lam_cen[1], lam_cen[0])
    dlam = lam_cen[1] - lam_cen[0]
    #vout1 = vout0 + dv

    tau_out0, tau_sys0 = tau_lambda_fn(wrest, lam_cen[0], f0[0], lam_out, N_out, N_sys, b_D_out, b_D_sys)
    tau_out1, tau_sys1 = tau_lambda_fn(wrest, lam_cen[1], f0[1], lam_out + dlam, N_out, N_sys, b_D_out, b_D_sys)

    output = {}

    F_out = (1.0 + (- Cf_out + Cf_out * np.exp(-(tau_out0 + tau_out1) ) ) )
    output["F_out"] = F_out

    output["F2796_out"] = 1.0 + (- Cf_out + Cf_out * np.exp(-tau_out0) )
    output["F2803_out"] = 1.0 + (- Cf_out + Cf_out * np.exp(-tau_out1) )

    #F_sys =  ( 1.0 + (- Cf_sys + Cf_sys * np.exp(-tau_sys0))+ (- Cf_sys + Cf_sys * np.exp(-tau_sys1)) )
    F_sys = 1.0 + (- Cf_sys + Cf_sys * np.exp(- (tau_sys0 + tau_sys1) ))
    output["F_sys"] = F_sys

    lam_ems12 = lam_ems1 + dlam
    lam_ems22 = lam_ems2 + dlam

    F_ems = 1. + (A_1 * np.exp(- ( (wrest - lam_ems1 ) / (lam_ems1 * (b_D_ems1/c)) )**2. )\
                  + c_ems1 * A_1 * np.exp(- ( (wrest - lam_ems12 ) / (lam_ems12 * (b_D_ems1/c)) )**2. )\
                  + A_2 * np.exp(- ( (wrest - lam_ems2 ) / (lam_ems2 * (b_D_ems2/c)) )**2.)\
                  + c_ems2 * A_2 * np.exp(- ( (wrest - lam_ems22 ) / (lam_ems22 * (b_D_ems2/c)) )**2.))

    output["F_ems"] = F_ems
    output["F_ems1"] = 1. + (A_1 * np.exp(- ( (wrest - lam_ems1 ) / (lam_ems1 * (b_D_ems1/c)) )**2. )\
                  + c_ems1 * A_1 * np.exp(- ( (wrest - lam_ems12 ) / (lam_ems12 * (b_D_ems1/c)) )**2. ) )

    output["F_ems2"] = 1. + (A_2 * np.exp(- ( (wrest - lam_ems2 ) / (lam_ems2 * (b_D_ems2/c)) )**2.)\
                    + c_ems2 * A_2 * np.exp(- ( (wrest - lam_ems22 ) / (lam_ems22 * (b_D_ems2/c)) )**2.))

    output["F2796_ems1"] = 1. + (A_1 * np.exp(- ( (wrest - lam_ems1 ) / (lam_ems1 * (b_D_ems1/c)) )**2. ) )
    output["F2803_ems1"] = 1. + c_ems1 * A_1 * np.exp(- ( (wrest - lam_ems12 ) / (lam_ems12 * (b_D_ems1/c)) )**2. )

    output["F2796_ems2"] = 1. + A_2 * np.exp(- ( (wrest - lam_ems2 ) / (lam_ems2 * (b_D_ems2/c)) )**2.)
    output["F2803_ems2"] = 1. + c_ems2 * A_2 * np.exp(- ( (wrest - lam_ems22 ) / (lam_ems22 * (b_D_ems2/c)) )**2.)



    #spec_res, v_res = u.spectral_res( lam_cen[0] * (1. + z_r) )
    v_res = 108.
    mage_kernel = ( (v_res/ (vel[1] - vel[0]) )/2.355)
    g = Gaussian1DKernel(stddev=mage_kernel)
    F = F_out * F_sys * F_ems
    output["F_unconv"] = F
    F_conv = convolve(F, g, boundary='extend')
    output["F_conv"] = F_conv
    return output
