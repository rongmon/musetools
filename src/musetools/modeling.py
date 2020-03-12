import numpy as np
import matplotlib.pyplot as plt
from astropy.convolution import convolve, Gaussian1DKernel
import musetools.util as u



#####################################################################################
#                    New Models as described in  Rubin et al 2014
#####################################################################################

####################### Fe II Full Model ############################################
def model_FeII_full(v, vout0, vems, b_D, b_D_sys, b_D_ems, logN, logN_sys, A, c_ems1, c_ems2, c_ems3, c_ems4):
    """
    The output of this function is the full convolved profile for the Fe II: F_conv
    v: is the input velocity array of the spectrum with respect to 2586. use veldiff() to transform from rest-frame wavelength into velocity
    
    The free parameters for this model are:
    1- vout0: is the velocity of the centroid of tau of the Fe II outflowing component
    2- vems: is the centroid velocity for the Fe II emission component
    3- b_D: Doppler velocity width for tau the outflowing component
    4- b_D_sys: Doppler velocity width for tau of the systemic component
    5- b_D_ems: Doppler velocity width for the emisssion component
    6- logN: Column density of the Fe II outflowing component
    7- logN_sys: Column density of the Fe II systemic component
    8- A: is the normalized flux amplitude for the emission line 2612
    9- c_ems1: is the line ratio between Fe II emission lines 2626/2612
    10- c_ems2: is the line ratio between Fe II emission lines 2632/2612
    11- c_ems3: is the line ratio between Fe II emission lines 2365/2612
    12- c_ems4: is the line ratio between Fe II emission lines 2396/2612
    
    You can try to use the covering fractions as free parameters:
    Cf_sys: is the covering fraction for the systemic component
    Cf_out: is the covering fraction for the outflowing component
    
    Both of them are constant and equal to 1 in this version of the model: Cf_sys = Cf_out = 1.0
    
    """
    Cf_sys = 1.0
    Cf_out = 1.0
    
    
    z_r = 1.7039397365102  # The global redshift value for the galaxy, It is used to get the observed wavelength for Fe II lines, that will be used to get the muse resolution corresponding to this wavelength before convolving the model
    N = 10.**logN
    N_sys = 10.**logN_sys
    
    lam_cen_ems = [2612.654, 2626.451, 2632.1081, 2365.552, 2396.355]
    lam_cen_abs = [2586.650, 2600.173, 2344.213, 2374.460, 2382.764]
    f0 =          [0.069125,   0.2394,   0.1142,   0.0313, 0.320]

    '''
    Absorption lines: 2344.212, 2374.460, 2382.764, 2586.650, 2600.173
    Emission lines: 2365.552, 2396.355, 2612.654, 2626.451, 2632.1081
    '''
    # Absorption Lines Velocities with respect to the Fe II absorption line 2586  
    dv_abs1 = u.veldiff(lam_cen_abs[1],lam_cen_abs[0]); vout1 = vout0 + dv_abs1           # Velocity difference between the absorption lines 2600 and 2586
    dv_abs2 = u.veldiff(lam_cen_abs[2],lam_cen_abs[0]); vout2 = vout0 + dv_abs2           # Velocity difference between the absorption lines 2344 and 2586
    dv_abs3 = u.veldiff(lam_cen_abs[3],lam_cen_abs[0]); vout3 = vout0 + dv_abs3           # Velocity difference between the absorption lines 2374 and 2586
    dv_abs4 = u.veldiff(lam_cen_abs[4],lam_cen_abs[0]); vout4 = vout0 + dv_abs4           # Velocity difference between the absorption lines 2382 and 2586
    # Emission Lines Velocities with respect to the Fe II absorption line 2586
    dv_ems0 = u.veldiff(lam_cen_ems[0],lam_cen_abs[0]);  vems0 = vems + dv_ems0           # Velocity difference between the emission line 2612 and absorption line 2586 
    dv_ems1 = u.veldiff(lam_cen_ems[1],lam_cen_abs[0]);  vems1 = vems + dv_ems1           # Velocity difference between the emission line 2626 and absorption line 2586
    dv_ems2 = u.veldiff(lam_cen_ems[2],lam_cen_abs[0]);  vems2 = vems + dv_ems2           # Velocity difference between the emission line 2632 and absorption line 2586
    dv_ems3 = u.veldiff(lam_cen_ems[3],lam_cen_abs[0]);  vems3 = vems + dv_ems3           # Velocity difference between the emission line 2365 and absorption line 2586
    dv_ems4 = u.veldiff(lam_cen_ems[4],lam_cen_abs[0]);  vems4 = vems + dv_ems4           # Velocity difference between the emission line 2396 and absorption line 2586
    

    # Tau for the outflowing component for each absorption line
    tau_out0 = (1.497e-15 * lam_cen_abs[0] * f0[0]) * (N/ b_D) * np.exp(-8.*np.log(2.) * ( (v -vout0) / b_D )**2. ) * np.piecewise(v,[v< 0.0, v>= 0.0], [1., 0.]) # tau for the Outflow at 2586
    tau_out1 = (1.497e-15 * lam_cen_abs[1] * f0[1]) * (N/ b_D) * np.exp(-8.*np.log(2.) * ( (v -vout1) / b_D )**2. ) * np.piecewise(v,[v< dv_abs1, v>= dv_abs1], [1., 0.])  # tau for the Outflow at 2600 
    tau_out2 = (1.497e-15 * lam_cen_abs[2] * f0[2]) * (N/ b_D) * np.exp(-8.*np.log(2.) * ( (v -vout2) / b_D )**2. ) * np.piecewise(v,[v< dv_abs2, v>= dv_abs2], [1., 0.]) # tau for the Outflow at 2344
    tau_out3 = (1.497e-15 * lam_cen_abs[3] * f0[3]) * (N/ b_D) * np.exp(-8.*np.log(2.) * ( (v -vout3) / b_D )**2. ) * np.piecewise(v,[v< dv_abs3, v>= dv_abs3], [1., 0.]) # tau for the Outflow at 2374
    tau_out4 = (1.497e-15 * lam_cen_abs[4] * f0[4]) * (N/ b_D) * np.exp(-8.*np.log(2.) * ( (v -vout4) / b_D )**2. ) * np.piecewise(v,[v< dv_abs4, v>= dv_abs4], [1., 0.])# tau for the Outflow at 2382
    
    # Tau for the systemic component for each absorption line
    tau_sys0 = (1.497e-15 * lam_cen_abs[0] * f0[0]) * (N_sys/ b_D_sys) * np.exp(-8.*np.log(2.) * ( (v) / b_D_sys )**2. )           # tau for the Systemic at 2586
    tau_sys1 = (1.497e-15 * lam_cen_abs[1] * f0[1]) * (N_sys/ b_D_sys) * np.exp(-8.*np.log(2.) * ( (v - dv_abs1) / b_D_sys )**2. ) # tau for the Systemic at 2600
    tau_sys2 = (1.497e-15 * lam_cen_abs[2] * f0[2]) * (N_sys/ b_D_sys) * np.exp(-8.*np.log(2.) * ( (v - dv_abs2) / b_D_sys )**2. ) # tau for the Systemic at 2344
    tau_sys3 = (1.497e-15 * lam_cen_abs[3] * f0[3]) * (N_sys/ b_D_sys) * np.exp(-8.*np.log(2.) * ( (v - dv_abs3) / b_D_sys )**2. ) # tau for the Systemic at 2374
    tau_sys4 = (1.497e-15 * lam_cen_abs[4] * f0[4]) * (N_sys/ b_D_sys) * np.exp(-8.*np.log(2.) * ( (v - dv_abs4) / b_D_sys )**2. ) # tau for the Systemic at 2382
    
    # The Full outflowing component
    F_out = (1.0 + (- Cf_out + Cf_out * np.exp(-tau_out0) ) * np.piecewise(v,[v< 0.0, v>= 0.0], [1., 0.])\
             + (- Cf_out + Cf_out * np.exp(-tau_out1) )\
             + (- Cf_out + Cf_out * np.exp(-tau_out2) )\
             + (- Cf_out + Cf_out * np.exp(-tau_out3) )\
             + (- Cf_out + Cf_out * np.exp(-tau_out4) ) ) 
    
    # The full systemic component
    F_sys =  ( 1.0 + (- Cf_sys + Cf_sys * np.exp(-tau_sys0))\
             + (- Cf_sys + Cf_sys * np.exp(-tau_sys1))\
             + (- Cf_sys + Cf_sys * np.exp(-tau_sys2))\
             + (- Cf_sys + Cf_sys * np.exp(-tau_sys3))\
             + (- Cf_sys + Cf_sys * np.exp(-tau_sys4)))   
    
    # The full emission component
    F_ems = 1. + (A * np.exp(-8.*np.log(2.) * ( (v - vems0) / b_D_ems )**2.)\
                 + c_ems1 * A * np.exp(-8.*np.log(2.) * ( (v - vems1) / b_D_ems)**2.)\
                 + c_ems2 * A * np.exp(-8.*np.log(2.) * ( (v - vems2) / b_D_ems)**2.)\
                 + c_ems3 * A * np.exp(-8.*np.log(2.) * ( (v - vems3) / b_D_ems)**2.)\
                 + c_ems4 * A * np.exp(-8.*np.log(2.) * ( (v - vems4) / b_D_ems)**2.) )
    
    spec_res, v_res = u.spectral_res( 2586.650 * (1. + z_r) )  # wavelength resolution, and velocity resolution for muse at the observed wavelength of Fe II 2586
    muse_kernel = ((spec_res/1.25)/2.355)
    g = Gaussian1DKernel(stddev=muse_kernel)
    # The full unconvolved model
    F = F_out * F_sys * F_ems
    # The full convolved model
    F_conv = convolve(F, g, boundary='extend')
    return F_conv


########################## Fe II Full model & individual conmponents ############################################
def model_FeII_comps(v, vout0, vems, b_D, b_D_sys, b_D_ems, logN, logN_sys, A, c_ems1, c_ems2, c_ems3, c_ems4):
    """
    The output of this function is the full convolved profile for the Fe II plus the individual components:
    F_conv, F, F_out, F_sys, F_ems, F_out0, F_out1, F_out2, F_out3, F_out4, F_ems0, F_ems1, F_ems2, F_ems3, F_ems4
    
    v: is the input velocity array of the spectrum with respect to Fe II 2586. use veldiff() to transform from rest-frame wavelength into velocity
    
    The free parameters for this model are:
    1- vout0: is the velocity of the centroid of tau of the Fe II outflowing component
    2- vems: is the centroid velocity for the Fe II emission component
    3- b_D: Doppler velocity width for tau the outflowing component
    4- b_D_sys: Doppler velocity width for tau of the systemic component
    5- b_D_ems: Doppler velocity width for the emisssion component
    6- logN: Column density of the Fe II outflowing component
    7- logN_sys: Column density of the Fe II systemic component
    8- A: is the normalized flux amplitude for the emission line 2612
    9- c_ems1: is the line ratio between Fe II emission lines 2626/2612
    10- c_ems2: is the line ratio between Fe II emission lines 2632/2612
    11- c_ems3: is the line ratio between Fe II emission lines 2365/2612
    12- c_ems4: is the line ratio between Fe II emission lines 2396/2612
    
    You can try to use the covering fractions as free parameters:
    Cf_sys: is the covering fraction for the systemic component
    Cf_out: is the covering fraction for the outflowing component
    
    Both of them are constant and equal to 1 in this version of the model: Cf_sys = Cf_out = 1.0
    
    """

    Cf_sys = 1.0
    Cf_out = 1.0
    z_r = 1.7039397365102  # The global redshift value for the galaxy, It is used to get the observed wavelength for Fe II lines, that will be used to get the muse resolution corresponding to this wavelength before convolving the model
    N = 10.**logN
    N_sys = 10.**logN_sys
    
    lam_cen_ems = [2612.654, 2626.451, 2632.1081, 2365.552, 2396.355]
    lam_cen_abs = [2586.650, 2600.173, 2344.213, 2374.460, 2382.764]
    f0 =          [0.069125,   0.2394,   0.1142,   0.0313, 0.320]

    '''
    Absorption lines: 2344.212, 2374.460, 2382.764, 2586.650, 2600.173
    Emission lines: 2365.552, 2396.355, 2612.654, 2626.451, 2632.1081
    '''
    # Absorption Lines Velocities with respect to the Fe II absorption line 2586 
    dv_abs1 = u.veldiff(lam_cen_abs[1],lam_cen_abs[0]); vout1 = vout0 + dv_abs1           # Velocity difference between the absorption lines 2600 and 2586
    dv_abs2 = u.veldiff(lam_cen_abs[2],lam_cen_abs[0]); vout2 = vout0 + dv_abs2           # Velocity difference between the absorption lines 2344 and 2586
    dv_abs3 = u.veldiff(lam_cen_abs[3],lam_cen_abs[0]); vout3 = vout0 + dv_abs3           # Velocity difference between the absorption lines 2374 and 2586
    dv_abs4 = u.veldiff(lam_cen_abs[4],lam_cen_abs[0]); vout4 = vout0 + dv_abs4           # Velocity difference between the absorption lines 2382 and 2586
    # Emission Lines Velocities with respect to the Fe II absorption line 2586
    dv_ems0 = u.veldiff(lam_cen_ems[0],lam_cen_abs[0]);  vems0 = vems + dv_ems0           # Velocity difference between the emission line 2612 and absorption line 2586 
    dv_ems1 = u.veldiff(lam_cen_ems[1],lam_cen_abs[0]);  vems1 = vems + dv_ems1           # Velocity difference between the emission line 2626 and absorption line 2586 
    dv_ems2 = u.veldiff(lam_cen_ems[2],lam_cen_abs[0]);  vems2 = vems + dv_ems2           # Velocity difference between the emission line 2632 and absorption line 2586 
    dv_ems3 = u.veldiff(lam_cen_ems[3],lam_cen_abs[0]);  vems3 = vems + dv_ems3           # Velocity difference between the emission line 2365 and absorption line 2586 
    dv_ems4 = u.veldiff(lam_cen_ems[4],lam_cen_abs[0]);  vems4 = vems + dv_ems4           # Velocity difference between the emission line 2396 and absorption line 2586 

    # Tau for the outflowing component for each absorption line
    tau_out0 = (1.497e-15 * lam_cen_abs[0] * f0[0]) * (N/ b_D) * np.exp(-8.*np.log(2.) * ( (v -vout0) / b_D )**2. ) * np.piecewise(v,[v< 0.0, v>= 0.0], [1., 0.]) # tau for the Outflow at 2586
    tau_out1 = (1.497e-15 * lam_cen_abs[1] * f0[1]) * (N/ b_D) * np.exp(-8.*np.log(2.) * ( (v -vout1) / b_D )**2. ) * np.piecewise(v,[v< dv_abs1, v>= dv_abs1], [1., 0.]) # tau for the Outflow at 2600 
    tau_out2 = (1.497e-15 * lam_cen_abs[2] * f0[2]) * (N/ b_D) * np.exp(-8.*np.log(2.) * ( (v -vout2) / b_D )**2. ) * np.piecewise(v,[v< dv_abs2, v>= dv_abs2], [1., 0.]) # tau for the Outflow at 2344
    tau_out3 = (1.497e-15 * lam_cen_abs[3] * f0[3]) * (N/ b_D) * np.exp(-8.*np.log(2.) * ( (v -vout3) / b_D )**2. ) * np.piecewise(v,[v< dv_abs3, v>= dv_abs3], [1., 0.]) # tau for the Outflow at 2374
    tau_out4 = (1.497e-15 * lam_cen_abs[4] * f0[4]) * (N/ b_D) * np.exp(-8.*np.log(2.) * ( (v -vout4) / b_D )**2. ) * np.piecewise(v,[v< dv_abs4, v>= dv_abs4], [1., 0.]) # tau for the Outflow at 2382
    
    # Tau for the systemic component for each absorption line
    
    tau_sys0 = (1.497e-15 * lam_cen_abs[0] * f0[0]) * (N_sys/ b_D_sys) * np.exp(-8.*np.log(2.) * ( (v) / b_D_sys )**2. )           # tau for the Systemic at 2586
    tau_sys1 = (1.497e-15 * lam_cen_abs[1] * f0[1]) * (N_sys/ b_D_sys) * np.exp(-8.*np.log(2.) * ( (v - dv_abs1) / b_D_sys )**2. ) # tau for the Systemic at 2600
    tau_sys2 = (1.497e-15 * lam_cen_abs[2] * f0[2]) * (N_sys/ b_D_sys) * np.exp(-8.*np.log(2.) * ( (v - dv_abs2) / b_D_sys )**2. ) # tau for the Systemic at 2344
    tau_sys3 = (1.497e-15 * lam_cen_abs[3] * f0[3]) * (N_sys/ b_D_sys) * np.exp(-8.*np.log(2.) * ( (v - dv_abs3) / b_D_sys )**2. ) # tau for the Systemic at 2374
    tau_sys4 = (1.497e-15 * lam_cen_abs[4] * f0[4]) * (N_sys/ b_D_sys) * np.exp(-8.*np.log(2.) * ( (v - dv_abs4) / b_D_sys )**2. ) # tau for the Systemic at 2382
    
    # The Full outflowing component 
    F_out = (1.0 + (- Cf_out + Cf_out * np.exp(-tau_out0) ) \
             + (- Cf_out + Cf_out * np.exp(-tau_out1) ) \
             + (- Cf_out + Cf_out * np.exp(-tau_out2) ) \
             + (- Cf_out + Cf_out * np.exp(-tau_out3) ) \
             + (- Cf_out + Cf_out * np.exp(-tau_out4) ) ) 
    
    # The individual outflowing components
    F_out0 = (1.0 + (- Cf_out + Cf_out * np.exp(-tau_out0) )  )
    F_out1 = (1.0 + (- Cf_out + Cf_out * np.exp(-tau_out1) )  )
    F_out2 = (1.0 + (- Cf_out + Cf_out * np.exp(-tau_out2) )  )
    F_out3 = (1.0 + (- Cf_out + Cf_out * np.exp(-tau_out3) )  )
    F_out4 = (1.0 + (- Cf_out + Cf_out * np.exp(-tau_out4) )  )
    
    # The full systemic component
    F_sys =  ( 1.0 + (- Cf_sys + Cf_sys * np.exp(-tau_sys0))\
             + (- Cf_sys + Cf_sys * np.exp(-tau_sys1))\
             + (- Cf_sys + Cf_sys * np.exp(-tau_sys2))\
             + (- Cf_sys + Cf_sys * np.exp(-tau_sys3))\
             + (- Cf_sys + Cf_sys * np.exp(-tau_sys4)))   

    
    # The full emission component    
    F_ems = 1. + (A * np.exp(-8.*np.log(2.) * ( (v - vems0) / b_D_ems )**2.)\
                 + c_ems1 * A * np.exp(-8.*np.log(2.) * ( (v - vems1) / b_D_ems)**2.)\
                 + c_ems2 * A * np.exp(-8.*np.log(2.) * ( (v - vems2) / b_D_ems)**2.)\
                 + c_ems3 * A * np.exp(-8.*np.log(2.) * ( (v - vems3) / b_D_ems)**2.)\
                 + c_ems4 * A * np.exp(-8.*np.log(2.) * ( (v - vems4) / b_D_ems)**2.) )
    
    # The individual emission components
    F_ems0 = 1. + A * np.exp(-8.*np.log(2.) * ( (v - vems0) / b_D_ems )**2.) 
    F_ems1 = 1. + c_ems1 * A * np.exp(-8.*np.log(2.) * ( (v - vems1) / b_D_ems)**2.)
    F_ems2 = 1. + c_ems2 * A * np.exp(-8.*np.log(2.) * ( (v - vems2) / b_D_ems)**2.)
    F_ems3 = 1. + c_ems3 * A * np.exp(-8.*np.log(2.) * ( (v - vems3) / b_D_ems)**2.)
    F_ems4 = 1. + c_ems4 * A * np.exp(-8.*np.log(2.) * ( (v - vems4) / b_D_ems)**2.)
    
    spec_res, v_res = u.spectral_res( 2586.650 * (1. + z_r) )   # wavelength resolution, and velocity resolution for muse at the observed wavelength of Fe II 2586
    muse_kernel = ((spec_res/1.25)/2.355)
    g = Gaussian1DKernel(stddev=muse_kernel)
    # The full unconvolved model
    F = F_out * F_sys * F_ems
    # The full convolved model
    F_conv = convolve(F, g, boundary='extend')
    return F_conv, F, F_out, F_sys, F_ems, F_out0, F_out1, F_out2, F_out3, F_out4, F_ems0, F_ems1, F_ems2, F_ems3, F_ems4



####################### Mg II Full Model ######################################################
def model_MgII_full(v, vout0, vems1, vems2, b_D, b_D_sys, b_D_ems1, b_D_ems2, logN, logN_sys, A_1, A_2, c_ems1, c_ems2):
    """
    The output of this function is the full convolved Mg II model: F_conv
    - v: is the input velocity array of the spectrum withe respect to the Mg II 2796 line
    
    The Free parameters for the model are:
    1- vout0: centroid velocity of tau of the Mg II outflowing component
    2- vems1: emission velocity for Mg II 1st emission component
    3- vems2: emission velocity for Mg II 2nd emission component
    4- b_D: is the Doppler velocity width for the outflowing component
    5- b_D_sys: is the Doppler velocity width for the systemic component
    6- b_D_ems1: is the Doppler velocity width for the 1st emission component
    7- b_D_ems2: is the Doppler velocity width for the 2nd emission component
    8- logN: Column density of the outflowing component
    9- logN_sys: Column density of the systemic component
    10- A_1:      is the normalized Flux amplitude of the Mg II 2796 in the 1st emission component
    11- A_2:      is the normalized Flux amplitude of the Mg II 2796 in the 2nd emission component
    12- c_ems1: is the line ratio between Mg II emission lines 2803/2796 for the first emission component
    13- c_ems2: is the line ratio between Mg II emission lines 2803/2796 for the second emission component
    """
    Cf_sys = 1.0
    Cf_out = 1.0
    
    z_r = 1.7039397365102
    N = 10.**logN
    N_sys = 10.**logN_sys
    
    lam_cen = [2796.351, 2803.528]
    f0 =      [0.6155, 0.3058]
    
    dv = u.veldiff(lam_cen[1], lam_cen[0])  # Velocity difference between the Mg II lines 2803 and 2796
    vout1 = vout0 + dv

    # Tau for the outflowing component for each absorption line
    tau_out0 = (1.497e-15 * lam_cen[0] * f0[0]) * (N/ b_D) * np.exp(-8.*np.log(2.) * ( (v -vout0) / b_D )**2. ) * np.piecewise(v,[v< 0.0, v>= 0.0], [1., 0.]) # tau for the Outflow at 2796
    tau_out1 = (1.497e-15 * lam_cen[1] * f0[1]) * (N/ b_D) * np.exp(-8.*np.log(2.) * ( (v -vout1) / b_D )**2. ) * np.piecewise(v,[v< dv, v>= dv], [1., 0.]) # tau for the Outflow at 2803
    
    # Tau for the systemic component for each absorption line
    tau_sys0 = (1.497e-15 * lam_cen[0] * f0[0]) * (N_sys/ b_D_sys) * np.exp(-8.*np.log(2.) * ( (v) / b_D_sys )**2. )           # tau for the Systemic at 2796
    tau_sys1 = (1.497e-15 * lam_cen[1] * f0[1]) * (N_sys/ b_D_sys) * np.exp(-8.*np.log(2.) * ( (v - dv) / b_D_sys )**2. ) # tau for the Systemic at 2803
    
    # The full outflowing component
    F_out = (1.0 + (- Cf_out + Cf_out * np.exp(-tau_out0) ) \
             + (- Cf_out + Cf_out * np.exp(-tau_out1) )  )
    
    # The full systemic component
    F_sys =  ( 1.0 + (- Cf_sys + Cf_sys * np.exp(-tau_sys0))+ (- Cf_sys + Cf_sys * np.exp(-tau_sys1)) )
    
    
    
    vems11 = vems1 + dv
    vems22 = vems2 + dv
    
    # The full emission component
    F_ems = 1. + (A_1 * np.exp(-8.*np.log(2.) * ( (v - vems1) / b_D_ems1 )**2.)\
                  + c_ems1 * A_1 * np.exp(-8.*np.log(2.) * ( (v - vems11) / b_D_ems1)**2.)\
                  + A_2 * np.exp(-8. * np.log(2.) * ( (v - vems2) / b_D_ems2 )**2.)\
                  + c_ems2 * A_2 * np.exp(-8.*np.log(2.) * ( (v - vems22) / b_D_ems2)**2.))
    
    
    spec_res, v_res = u.spectral_res( lam_cen[0] * (1. + z_r) )   # wavelength resolution, and velocity resolution for muse at the observed wavelength of Mg II 2796
    muse_kernel = ((spec_res/1.25)/2.355)
    g = Gaussian1DKernel(stddev=muse_kernel)
    # The full unconvolved model
    F = F_out * F_sys * F_ems
    # The full convolved model
    F_conv = convolve(F, g, boundary='extend')
    return F_conv



########################## Mg II Full model and individual components ##################################################
def model_MgII_comps(v, vout0, vems1, vems2, b_D, b_D_sys, b_D_ems1, b_D_ems2, logN, logN_sys, A_1, A_2, c_ems1, c_ems2):
    """
    The output of this function is the full convolved Mg II model and individual components: 
    F_conv, F, F_out, F_sys, F_ems, F_out0, F_out1, F_ems1, F_ems2, F_ems11, F_ems12, F_ems21, F_ems22
    
    - v: is the input velocity array of the spectrum withe respect to the Mg II 2796 line
    
    The Free parameters for the model are:
    1- vout0: centroid velocity of tau of the Mg II outflowing component
    2- vems1: emission velocity for Mg II 1st emission component
    3- vems2: emission velocity for Mg II 2nd emission component
    4- b_D: is the Doppler velocity width for the outflowing component
    5- b_D_sys: is the Doppler velocity width for the systemic component
    6- b_D_ems1: is the Doppler velocity width for the 1st emission component
    7- b_D_ems2: is the Doppler velocity width for the 2nd emission component
    8- logN: Column density of the outflowing component
    9- logN_sys: Column density of the systemic component
    10- A_1:      is the normalized Flux amplitude of the Mg II 2796 in the 1st emission component
    11- A_2:      is the normalized Flux amplitude of the Mg II 2796 in the 2nd emission component
    12- c_ems1: is the line ratio between Mg II emission lines 2803/2796 for the first emission component
    13- c_ems2: is the line ratio between Mg II emission lines 2803/2796 for the second emission component
    """
    
    Cf_sys = 1.0
    Cf_out = 1.0
    
    z_r = 1.7039397365102
    N = 10.**logN
    N_sys = 10.**logN_sys
    
    lam_cen = [2796.351, 2803.528]

    dv = u.veldiff(lam_cen[1], lam_cen[0])
    vout1 = vout0 + dv

    
    tau_out0 = (1.497e-15 * lam_cen[0] * f0[0]) * (N/ b_D) * np.exp(-8.*np.log(2.) * ( (v -vout0) / b_D )**2. ) * np.piecewise(v,[v< 0.0, v>= 0.0], [1., 0.])  # tau for the Outflow at 2796
    tau_out1 = (1.497e-15 * lam_cen[1] * f0[1]) * (N/ b_D) * np.exp(-8.*np.log(2.) * ( (v -vout1) / b_D )**2. ) * np.piecewise(v,[v< dv, v>= dv], [1., 0.]) # tau for the Outflow at 2803
    
    
    tau_sys0 = (1.497e-15 * lam_cen[0] * f0[0]) * (N_sys/ b_D_sys) * np.exp(-8.*np.log(2.) * ( (v) / b_D_sys )**2. )           # tau for the Systemic at 2796
    tau_sys1 = (1.497e-15 * lam_cen[1] * f0[1]) * (N_sys/ b_D_sys) * np.exp(-8.*np.log(2.) * ( (v - dv) / b_D_sys )**2. ) # tau for the Systemic at 2803
    
    F_out = (1.0 + (- Cf_out + Cf_out * np.exp(-tau_out0) ) \
             + (- Cf_out + Cf_out * np.exp(-tau_out1) )  )
    
    F_out0 = 1.0 + (- Cf_out + Cf_out * np.exp(-tau_out0) ) 
    F_out1 = 1.0 + (- Cf_out + Cf_out * np.exp(-tau_out1) ) 
    
    
    
    F_sys =  ( 1.0 + (- Cf_sys + Cf_sys * np.exp(-tau_sys0))+ (- Cf_sys + Cf_sys * np.exp(-tau_sys1)) )

    vems11 = vems1 + dv
    vems22 = vems2 + dv
    # The full emission component
    F_ems = 1. + (A_1 * np.exp(-8.*np.log(2.) * ( (v - vems1) / b_D_ems1 )**2.)\
                  + c_ems1 * A_1 * np.exp(-8.*np.log(2.) * ( (v - vems11) / b_D_ems1)**2.)\
                  + A_2 * np.exp(-8. * np.log(2.) * ( (v - vems2) / b_D_ems2 )**2.)\
                  + c_ems2 * A_2 * np.exp(-8.*np.log(2.) * ( (v - vems22) / b_D_ems2)**2.))
    # The primary emission set
    F_ems1 = 1. + (A_1 * np.exp(-8.*np.log(2.) * ( (v - vems1) / b_D_ems1 )**2.)\
                   + c_ems1 * A_1 * np.exp(-8.*np.log(2.) * ( (v - vems11) / b_D_ems1)**2.) )
    # The secondary emission set
    F_ems2 = 1. + (A_2 * np.exp(-8. * np.log(2.) * ( (v - vems2) / b_D_ems2 )**2.)\
                   + c_ems2 * A_2 * np.exp(-8.*np.log(2.) * ( (v - vems22) / b_D_ems2)**2.))
    
    # Individual emission components for the primary emission set
    F_ems11 = 1. + A_1 * np.exp(-8.*np.log(2.) * ( (v - vems1) / b_D_ems1 )**2.)
    F_ems12 = 1. + c_ems1 * A_1 * np.exp(-8.*np.log(2.) * ( (v - vems11) / b_D_ems1)**2.)
    # Individual emission components for the secondary emission set
    F_ems21 = 1. + A_2 * np.exp(-8. * np.log(2.) * ( (v - vems2) / b_D_ems2 )**2.)
    F_ems22 = 1. + c_ems2 * A_2 * np.exp(-8.*np.log(2.) * ( (v - vems22) / b_D_ems2)**2.)
    
    spec_res, v_res = u.spectral_res( lam_cen[0] * (1. + z_r) ) # wavelength resolution, and velocity resolution for muse at the observed wavelength of Mg II 2796
    muse_kernel = ((spec_res/1.25)/2.355)
    g = Gaussian1DKernel(stddev=muse_kernel)
    # The full unconvolved model
    F = F_out * F_sys * F_ems   
    # The full convolved model         
    F_conv = convolve(F, g, boundary='extend')
    return F_conv, F, F_out, F_sys, F_ems, F_out0, F_out1, F_ems1, F_ems2, F_ems11, F_ems12, F_ems21, F_ems22






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

