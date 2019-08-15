import numpy as np
import musetools.modeling as m
import matplotlib.pyplot as plt
from astropy.io import ascii
from astropy.table import Table
import musetools.util as u
import matplotlib
import importlib
import time
importlib.reload(u)
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

info = ascii.read('/home/ahmed/Gdrive/astro/spectra/redshift_OII_vac_mcmc.dat')
xcen = info[0][:]
ycen = info[1][:]
zgal = info[4][:]

start_time = time.time()

for cx, cy, z in zip(xcen, ycen, zgal):
    data1 = ascii.read('/home/ahmed/Gdrive/astro/spectra/spec_lw_fe_'+str(cx)+'_'+str(cy)+'_norm.ascii')
    wave1 = data1['wave']

    data2 = ascii.read('/home/ahmed/Gdrive/astro/spectra/spectrum_fe_'+str(cx)+'_'+str(cy)+'.dat')
    wave2 = data2['wave']
    wave = u.airtovac(np.asarray([*wave1,*wave2]))
    wrest = wave/(1. + z)

    lam_cen = [2586.650,2600.173,2612.654,2626.451,2632.1081,2344.212,2365.552,2374.460,2382.764,2396.355]
    f0 =      [0.069125,   0.2394,   0.1142,   0.0313, 0.320]
    #startv = -960.;  endv = 100.
    vel = u.veldiff(wrest,lam_cen[0])
    samples = np.load('/home/ahmed/Gdrive/astro/samples_emcee/FeII/samples/samples_FeII_'+str(cx)+'_'+str(cy)+'.npy')
    s = samples.shape
    ew2586_s = [];  vabs2586_s = [];  logN2586_s = []
    ew2600_s = [];  vabs2600_s = [];  logN2600_s = []
    ew2344_s = [];  vabs2344_s = [];  logN2344_s = []
    ew2374_s = [];  vabs2374_s = [];  logN2374_s = []
    ew2382_s = [];  vabs2382_s = [];  logN2382_s = []

    ew2612_s = [];  vems2612_s = []
    ew2626_s = [];  vems2626_s = []
    ew2632_s = [];  vems2632_s = []
    ew2365_s = [];  vems2365_s = []
    ew2396_s = [];  vems2396_s = []
    print(''+str(cx)+'_'+str(cy)+'')
    for i in range(s[0]):
        F_conv, F, Fabs, Fabs1, Fabs2, Fabs3, Fabs4, Fabs5, Fems, Fems1, Fems2, Fems3, Fems4, Fems5, Fsys = m.model_Fe_comps(vel,*samples[i,:])
        ew2586, vabs2586, logN2586 = u.compute_abs(wrest, Fabs1, lam_cen[0], samples[i,5], f0[0],samples[i,2],-1000.,100.)
        ew2600, vabs2600, logN2600 = u.compute_abs(wrest, Fabs2, lam_cen[1], samples[i,8]*samples[i,5],  f0[1], samples[i,2], -1000., 100.)
        ew2344, vabs2344, logN2344 = u.compute_abs(wrest, Fabs3, lam_cen[5], samples[i,9]*samples[i,5],  f0[2], samples[i,2], -1000., 100.)
        ew2374, vabs2374, logN2374 = u.compute_abs(wrest, Fabs4, lam_cen[7], samples[i,10]*samples[i,5], f0[3], samples[i,2], -1000., 100.)
        ew2382, vabs2382, logN2382 = u.compute_abs(wrest, Fabs5, lam_cen[8], samples[i,11]*samples[i,5], f0[4], samples[i,2], -1000., 100.)

        ew2586_s.append(ew2586);  vabs2586_s.append(vabs2586);  logN2586_s.append(logN2586)
        ew2600_s.append(ew2600);  vabs2600_s.append(vabs2600);  logN2600_s.append(logN2600)
        ew2344_s.append(ew2344);  vabs2344_s.append(vabs2344);  logN2344_s.append(logN2344)
        ew2374_s.append(ew2374);  vabs2374_s.append(vabs2374);  logN2374_s.append(logN2374)
        ew2382_s.append(ew2382);  vabs2382_s.append(vabs2382);  logN2382_s.append(logN2382)

        ew2612 = u.compute_ems(wrest, Fems1, lam_cen[2], -1000., 1000.)
        ew2626 = u.compute_ems(wrest, Fems2, lam_cen[3], -1000., 1000.)
        ew2632 = u.compute_ems(wrest, Fems3, lam_cen[4], -1000., 1000.)
        ew2365 = u.compute_ems(wrest, Fems4, lam_cen[6], -1000., 1000.)
        ew2396 = u.compute_ems(wrest, Fems5, lam_cen[9], -1000., 1000.)

        ew2612_s.append(ew2612);   vems2612_s.append(samples[i,1])#vems2612)
        ew2626_s.append(ew2626);   vems2626_s.append(samples[i,1])#vems2626)
        ew2632_s.append(ew2632);   vems2632_s.append(samples[i,1])#vems2632)
        ew2365_s.append(ew2365);   vems2365_s.append(samples[i,1])#vems2365)
        ew2396_s.append(ew2396);   vems2396_s.append(samples[i,1])#vems2396)

    data = Table([ew2586_s,vabs2586_s,logN2586_s,
                  ew2600_s,vabs2600_s,logN2600_s,
                  ew2344_s,vabs2344_s,logN2344_s,
                  ew2374_s,vabs2374_s,logN2374_s,
                  ew2382_s,vabs2382_s,logN2382_s,
                  ew2612_s,vems2612_s,
                  ew2626_s,vems2626_s,
                  ew2632_s,vems2632_s,
                  ew2365_s,vems2365_s,
                  ew2396_s,vems2396_s], names=['ew2586','vabs2586','logN2586',
                                               'ew2600','vabs2600','logN2600',
                                               'ew2344','vabs2344','logN2344',
                                               'ew2374','vabs2374','logN2374',
                                               'ew2382','vabs2382','logN2382',
                                               'ew2612','vems2612',
                                               'ew2626','vems2626',
                                               'ew2632','vems2632',
                                               'ew2365','vems2365',
                                               'ew2396','vems2396'])
    ascii.write(data,'/home/ahmed/Gdrive/astro/samples_emcee/FeII/physical_quantities/values_FeII_'+str(cx)+'_'+str(cy)+'.dat',overwrite=True)

    p_ew2586 = np.percentile(ew2586_s,[34,50,68]);  q_ew2586 = np.diff(p_ew2586)
    p_ew2600 = np.percentile(ew2600_s,[34,50,68]);  q_ew2600 = np.diff(p_ew2600)
    p_ew2344 = np.percentile(ew2344_s,[34,50,68]);  q_ew2344 = np.diff(p_ew2344)
    p_ew2374 = np.percentile(ew2374_s,[34,50,68]);  q_ew2374 = np.diff(p_ew2374)
    p_ew2382 = np.percentile(ew2382_s,[34,50,68]);  q_ew2382 = np.diff(p_ew2382)

    p_ew2612 = np.percentile(ew2612_s,[34,50,68]);  q_ew2612 = np.diff(p_ew2612)
    p_ew2626 = np.percentile(ew2626_s,[34,50,68]);  q_ew2626 = np.diff(p_ew2626)
    p_ew2632 = np.percentile(ew2632_s,[34,50,68]);  q_ew2632 = np.diff(p_ew2632)
    p_ew2365 = np.percentile(ew2365_s,[34,50,68]);  q_ew2365 = np.diff(p_ew2365)
    p_ew2396 = np.percentile(ew2396_s,[34,50,68]);  q_ew2396 = np.diff(p_ew2396)

    p_vabs2586 = np.percentile(vabs2586_s,[34,50,68]);  q_vabs2586 = np.diff(p_vabs2586)
    p_vabs2600 = np.percentile(vabs2600_s,[34,50,68]);  q_vabs2600 = np.diff(p_vabs2600)
    p_vabs2344 = np.percentile(vabs2344_s,[34,50,68]);  q_vabs2344 = np.diff(p_vabs2344)
    p_vabs2374 = np.percentile(vabs2374_s,[34,50,68]);  q_vabs2374 = np.diff(p_vabs2374)
    p_vabs2382 = np.percentile(vabs2382_s,[34,50,68]);  q_vabs2382 = np.diff(p_vabs2382)

    p_vems2612 = np.percentile(vems2612_s,[34,50,68]);  q_vems2612 = np.diff(p_vems2612)
    p_vems2626 = np.percentile(vems2626_s,[34,50,68]);  q_vems2626 = np.diff(p_vems2626)
    p_vems2632 = np.percentile(vems2632_s,[34,50,68]);  q_vems2632 = np.diff(p_vems2632)
    p_vems2365 = np.percentile(vems2365_s,[34,50,68]);  q_vems2365 = np.diff(p_vems2365)
    p_vems2396 = np.percentile(vems2396_s,[34,50,68]);  q_vems2396 = np.diff(p_vems2396)

    p_logN2586 = np.percentile(logN2586_s,[34,50,68]);  q_logN2586 = np.diff(p_logN2586)
    p_logN2600 = np.percentile(logN2600_s,[34,50,68]);  q_logN2600 = np.diff(p_logN2600)
    p_logN2344 = np.percentile(logN2344_s,[34,50,68]);  q_logN2344 = np.diff(p_logN2344)
    p_logN2374 = np.percentile(logN2374_s,[34,50,68]);  q_logN2374 = np.diff(p_logN2374)
    p_logN2382 = np.percentile(logN2382_s,[34,50,68]);  q_logN2382 = np.diff(p_logN2382)


    print('For the Absorption Lines: ')
    print('EW 2586:'+str('%.5f' % p_ew2586[1])+', low:'+str('%.5f' % q_ew2586[0])+',  up:'+str('%.5f' % q_ew2586[1])+'')
    print('EW 2600:'+str('%.5f' % p_ew2600[1])+', low:'+str('%.5f' % q_ew2600[0])+',  up:'+str('%.5f' % q_ew2600[1])+'')
    print('EW 2344:'+str('%.5f' % p_ew2344[1])+', low:'+str('%.5f' % q_ew2344[0])+',  up:'+str('%.5f' % q_ew2344[1])+'')
    print('EW 2374:'+str('%.5f' % p_ew2374[1])+', low:'+str('%.5f' % q_ew2374[0])+',  up:'+str('%.5f' % q_ew2374[1])+'')
    print('EW 2382:'+str('%.5f' % p_ew2382[1])+', low:'+str('%.5f' % q_ew2382[0])+',  up:'+str('%.5f' % q_ew2382[1])+'')
    print('- - -')
    print('Vabs 2586:'+str('%.5f' % p_vabs2586[1])+', low:'+str('%.5f' % q_vabs2586[0])+', up:'+str('%.5f' % q_vabs2586[1])+'')
    print('Vabs 2600:'+str('%.5f' % p_vabs2600[1])+', low:'+str('%.5f' % q_vabs2600[0])+', up:'+str('%.5f' % q_vabs2600[1])+'')
    print('Vabs 2344:'+str('%.5f' % p_vabs2344[1])+', low:'+str('%.5f' % q_vabs2344[0])+', up:'+str('%.5f' % q_vabs2344[1])+'')
    print('Vabs 2374:'+str('%.5f' % p_vabs2374[1])+', low:'+str('%.5f' % q_vabs2374[0])+', up:'+str('%.5f' % q_vabs2374[1])+'')
    print('Vabs 2382:'+str('%.5f' % p_vabs2382[1])+', low:'+str('%.5f' % q_vabs2382[0])+', up:'+str('%.5f' % q_vabs2382[1])+'')
    print('- - -')
    print('log(N) 2586:'+str('%.5f' % p_logN2586[1])+', low:'+str('%.5f' % q_logN2586[0])+', up:'+str('%.5f' % q_logN2586[1])+'')
    print('log(N) 2600:'+str('%.5f' % p_logN2600[1])+', low:'+str('%.5f' % q_logN2600[0])+', up:'+str('%.5f' % q_logN2600[1])+'')
    print('log(N) 2344:'+str('%.5f' % p_logN2344[1])+', low:'+str('%.5f' % q_logN2344[0])+', up:'+str('%.5f' % q_logN2344[1])+'')
    print('log(N) 2374:'+str('%.5f' % p_logN2374[1])+', low:'+str('%.5f' % q_logN2374[0])+', up:'+str('%.5f' % q_logN2374[1])+'')
    print('log(N) 2382:'+str('%.5f' % p_logN2382[1])+', low:'+str('%.5f' % q_logN2382[0])+', up:'+str('%.5f' % q_logN2382[1])+'')
    print('- - - - - - - - - - - - - - - - - - - -')
    print('For the Emission Lines: ')
    print('EW 2612:'+str('%.5f' % p_ew2612[1])+', low:'+str('%.5f' % q_ew2612[0])+',  up:'+str('%.5f' % q_ew2612[1])+'')
    print('EW 2626:'+str('%.5f' % p_ew2626[1])+', low:'+str('%.5f' % q_ew2626[0])+',  up:'+str('%.5f' % q_ew2626[1])+'')
    print('EW 2632:'+str('%.5f' % p_ew2632[1])+', low:'+str('%.5f' % q_ew2632[0])+',  up:'+str('%.5f' % q_ew2632[1])+'')
    print('EW 2365:'+str('%.5f' % p_ew2365[1])+', low:'+str('%.5f' % q_ew2365[0])+',  up:'+str('%.5f' % q_ew2365[1])+'')
    print('EW 2396:'+str('%.5f' % p_ew2396[1])+', low:'+str('%.5f' % q_ew2396[0])+',  up:'+str('%.5f' % q_ew2396[1])+'')
    print('- - -')
    print('Vems 2612:'+str('%.5f' % p_vems2612[1])+', low:'+str('%.5f' % q_vems2612[0])+', up:'+str('%.5f' % q_vems2612[1])+'')
    print('Vems 2626:'+str('%.5f' % p_vems2626[1])+', low:'+str('%.5f' % q_vems2626[0])+', up:'+str('%.5f' % q_vems2626[1])+'')
    print('Vems 2632:'+str('%.5f' % p_vems2632[1])+', low:'+str('%.5f' % q_vems2632[0])+', up:'+str('%.5f' % q_vems2632[1])+'')
    print('Vems 2365:'+str('%.5f' % p_vems2365[1])+', low:'+str('%.5f' % q_vems2365[0])+', up:'+str('%.5f' % q_vems2365[1])+'')
    print('Vems 2396:'+str('%.5f' % p_vems2396[1])+', low:'+str('%.5f' % q_vems2396[0])+', up:'+str('%.5f' % q_vems2396[1])+'')
    print('_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ ')
print("--- %s seconds ---" % (time.time() - start_time))
