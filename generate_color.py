#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

'''
Given a catalog and a subset of this catalog (the hosts - 'nev' objects),
this program randomly gives to each entry of this catalog a magnitude (for example in the B band)
using a Gaussian distribution for hosts and Schechter function for other galaxies.

Please notice that the hosts are required to be the first nev entries of the catalog.
'''

def Schechter(M, Ms, alpha, gal_den):
    '''
    Schechter function
    '''
    tmp = 10**(-0.4*(M-Ms))
    return gal_den * tmp**(alpha+1.0)*np.exp(-tmp)

if __name__ == '__main__':

    data   = pd.read_csv('MDC1_universe.txt', sep = ' ')
    events = np.genfromtxt('2015_all_inj.txt', names = True)
    nev = events.shape[0]

    # Draw nev luminosities for hosts (gaussian distributed)
    mag_hosts  = np.random.normal(lmean, lsigma, nev)

    mag_others = np.zeros(len(data)-nev)
    # Draw other galaxies luminosity (Schechter function) with Von Neumann's accept-reject sampling method
    mag_bounds = [-17, -24]
    c = abs(mag_bounds[1]-mag_bounds[0])*Schechter(mag_bounds[0], Ms, alpha, gal_den)
    for i in range(len(mag_others)):
        flag = True
        while(flag):
            temp_mag = np.random.uniform(mag_bound[1],mag_bound[0])
            att      = np.random.uniform(0, c)
            bound    = Schechter(mag_bounds[0], Ms, alpha, gal_den)
            if(att < bound):
                mag_others[i] = temp_mag
                flag = False

    magnitudes  = np.concatenate([mag_hosts, mag_others])
    data['mag'] = magnitudes

    data.to_csv('m-catalog.txt', header=True, index=False, sep='\t')
