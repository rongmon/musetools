import numpy as np
import musetools.modeling as m
import matplotlib.pyplot as plt
from astropy.io import ascii
from astropy.table import Table
import musetools.util as u


import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

info = ascii.read('/home/ahmed/Gdrive/astro/spectra/redshift_OII_vac_mcmc.dat')
xcen = info[0][:]
ycen = info[1][:]
zgal = info[4][:]


from time import sleep
import time
from tqdm import tqdm
start_time = time.time()
for cx, cy, z in zip(xcen[:11], ycen[:11], zgal[:11]):
    data = ascii.read('/home/ahmed/Gdrive/astro/spectra/spectrum_mg_'+str(cx)+'_'+str(cy)+'.dat')
    wave = data['wave']
    wave = u.airtovac(wave)
    wrest = wave/(1. + z)
    lam_cen = [2796.351, 2803.528, 2797.084, 2799.326, 2804.346, 2808.975]
    f0 = [0.6155, 0.3058]
    vel = u.veldiff(wrest,lam_cen[0])
    samples = np.load('/home/ahmed/Gdrive/astro/samples_emcee/MgII/samples/samples_MgII_'+str(cx)+'_'+str(cy)+'.npy')
    s = samples.shape
    ew2796_s = [];  vabs2796_s = [];  logN2796_s = []
    ew2803_s = [];  vabs2803_s = [];  logN2803_s = []

    ewems1_s = [];  vems1_s = []
    ewems2_s = [];  #vems2_s = []
    ewems3_s = [];  vems3_s = []
    ewems4_s = [];  #vems4_s = []

    print(''+str(cx)+'_'+str(cy)+'')

    for i in tqdm(range(s[0])):
        F_conv, F, Fabs, Fabs1, Fabs2, Fems, Fems1, Fems2, Fems3, Fems4, Fsys = m.model_Mg_comps(vel,*samples[i,:])
        #ew, vabs, logN = compute_abs(wrest,flx_norm, lam_center, tau, f0, sig, vmin,vmax)
        ew2796, vabs2796, logN2796 = u.compute_abs(wrest, Fabs1, lam_cen[0], samples[i,7], f0[0], samples[i,3], -1900., 20.)
        ew2803, vabs2803, logN2803 = u.compute_abs(wrest, Fabs2, lam_cen[1], samples[i,11]*samples[i,7], f0[1], samples[i,3], -1900., 20.)

        ew2796_s.append(ew2796);   vabs2796_s.append(vabs2796);  logN2796_s.append(logN2796)
        ew2803_s.append(ew2803);   vabs2803_s.append(vabs2803);  logN2803_s.append(logN2803)

        #compute_ems(wrest, flx_norm, lam_center, vmin, vmax)
        ewems1 = u.compute_ems(wrest, Fems1, lam_cen[2], -1000., 1000.)
        ewems2 = u.compute_ems(wrest, Fems2, lam_cen[4], -1000., 1000.)
        ewems3 = u.compute_ems(wrest, Fems3, lam_cen[3], -1000., 1000.)
        ewems4 = u.compute_ems(wrest, Fems4, lam_cen[5], -1000., 1000.)

        ewems1_s.append(ewems1);  vems1_s.append(samples[i,1])
        ewems2_s.append(ewems2);  #vems2_s.append(vems2)
        ewems3_s.append(ewems3);  vems3_s.append(samples[i,2])
        ewems4_s.append(ewems4);  #vems4_s.append(vems4)
        time.sleep(0.000000000000000000001)
    values = Table([ew2796_s, vabs2796_s, logN2796_s,
                    ew2803_s, vabs2803_s, logN2803_s,
                    ewems1_s, vems1_s,
                    ewems2_s,
                    ewems3_s, vems3_s,
                    ewems4_s],names=['ew2796', 'vabs2796', 'logN2796',
                                              'ew2803', 'vabs2803', 'logN2803',
                                              'ewems1', 'vems1',
                                              'ewems2',
                                              'ewems3', 'vems3',
                                              'ewems4',])
    ascii.write(values,'/home/ahmed/Gdrive/astro/samples_emcee/MgII/physical_quantities/values_MgII_'+str(cx)+'_'+str(cy)+'.dat',overwrite=True)

    p_ew2796 = np.percentile(ew2796_s,[34,50,68]);  q_ew2796 = np.diff(p_ew2796)
    p_ew2803 = np.percentile(ew2803_s,[34,50,68]);  q_ew2803 = np.diff(p_ew2803)

    p_ewems1 = np.percentile(ewems1_s,[34,50,68]);  q_ewems1 = np.diff(p_ewems1)
    p_ewems2 = np.percentile(ewems2_s,[34,50,68]);  q_ewems2 = np.diff(p_ewems2)
    p_ewems3 = np.percentile(ewems3_s,[34,50,68]);  q_ewems3 = np.diff(p_ewems3)
    p_ewems4 = np.percentile(ewems4_s,[34,50,68]);  q_ewems4 = np.diff(p_ewems4)

    p_vabs2796 = np.percentile(vabs2796_s,[34,50,68]);  q_vabs2796 = np.diff(p_vabs2796)
    p_vabs2803 = np.percentile(vabs2803_s,[34,50,68]);  q_vabs2803 = np.diff(p_vabs2803)

    p_vems1 = np.percentile(vems1_s,[34,50,68]);  q_vems1 = np.diff(p_vems1)
    #p_vems2 = np.percentile(vems2_s,[34,50,68]);  q_vems2 = np.diff(p_vems2)
    p_vems3 = np.percentile(vems3_s,[34,50,68]);  q_vems3 = np.diff(p_vems3)
    #p_vems4 = np.percentile(vems4_s,[34,50,68]);  q_vems4 = np.diff(p_vems4)

    p_logN2796 = np.percentile(logN2796_s,[34,50,68]);  q_logN2796 = np.diff(p_logN2796)
    p_logN2803 = np.percentile(logN2803_s,[34,50,68]);  q_logN2803 = np.diff(p_logN2803)

    print('For the Absorption Lines: ')
    print('EW 2796: '+str('%.5f' % p_ew2796[1])+', low:'+str('%.5f' % q_ew2796[0])+',  up:'+str('%.5f' % q_ew2796[1])+'')
    print('EW 2803: '+str('%.5f' % p_ew2803[1])+', low:'+str('%.5f' % q_ew2803[0])+',  up:'+str('%.5f' % q_ew2803[1])+'')
    print('- - - -')
    print('Vabs 2796: '+str('%.5f' % p_vabs2796[1])+', low:'+str('%.5f' % q_vabs2796[0])+', up:'+str('%.5f' % q_vabs2796[1])+'')
    print('Vabs 2803: '+str('%.5f' % p_vabs2803[1])+', low:'+str('%.5f' % q_vabs2803[0])+', up:'+str('%.5f' % q_vabs2803[1])+'')
    print('- - - -')
    print('log(N) 2796: '+str('%.5f' % p_logN2796[1])+', low:'+str('%.5f' % q_logN2796[0])+', up:'+str('%.5f' % q_logN2796[1])+'')
    print('log(N) 2803: '+str('%.5f' % p_logN2803[1])+', low:'+str('%.5f' % q_logN2803[0])+', up:'+str('%.5f' % q_logN2803[1])+'')
    print('- - - - - - - - - - - - - - - - - - - -')
    print('For the Emission Lines: ')
    print('EW ems1: '+str('%.5f' % p_ewems1[1])+', low:'+str('%.5f' % q_ewems1[0])+',  up:'+str('%.5f' % q_ewems1[1])+'')
    print('EW ems2: '+str('%.5f' % p_ewems2[1])+', low:'+str('%.5f' % q_ewems2[0])+',  up:'+str('%.5f' % q_ewems2[1])+'')
    print('EW ems3: '+str('%.5f' % p_ewems3[1])+', low:'+str('%.5f' % q_ewems3[0])+',  up:'+str('%.5f' % q_ewems3[1])+'')
    print('EW ems4: '+str('%.5f' % p_ewems4[1])+', low:'+str('%.5f' % q_ewems4[0])+',  up:'+str('%.5f' % q_ewems4[1])+'')
    print('- - - -')
    print('Vems1: '+str('%.5f' % p_vems1[1])+', low:'+str('%.5f' % q_vems1[0])+', up:'+str('%.5f' % q_vems1[1])+'')
    #print('Vems2: '+str('%.5f' % p_vems2[1])+', low:'+str('%.5f' % q_vems2[0])+', up:'+str('%.5f' % q_vems2[1])+'')
    print('Vems3: '+str('%.5f' % p_vems3[1])+', low:'+str('%.5f' % q_vems3[0])+', up:'+str('%.5f' % q_vems3[1])+'')
    #print('Vems4: '+str('%.5f' % p_vems4[1])+', low:'+str('%.5f' % q_vems4[0])+', up:'+str('%.5f' % q_vems4[1])+'')
    print('_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ ')


print("--- %s seconds ---" % (time.time() - start_time))
