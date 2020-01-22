#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from schechter_function import *

'''
Given a catalog and a subset of this catalog (the hosts - 'nev' objects),
this program randomly gives to each entry of this catalog a magnitude (for example in the B band)
using a Gaussian distribution for hosts and Schechter function for other galaxies.

Please notice that the hosts are required to be the first nev entries of the catalog.
'''

if __name__ == '__main__':

    data   = pd.read_csv('MDC1_universe.txt', sep = ' ')
    events = np.genfromtxt('2015_all_inj.txt', names = True)
    nev = events.shape[0]

    # Schechter parameters as in O2 H0 paper (from Gehrels et al. (2016))
    alpha   = -1.07
    Mstar   = -20.47

    # Gaussian Parameters
    m_mean  = -20.
    m_sigma = 0.5

    # Draw nev luminosities for hosts (gaussian distributed)
    mag_hosts  = np.random.normal(m_mean, m_sigma, nev)

    mag_others = np.zeros(len(data)-nev)
    # Draw other galaxies luminosity (Schechter function) with Von Neumann's accept-reject sampling method
    mag_bounds = [-17, -24]

    Schechter = SchechterMagFunction(mag_bounds[1], mag_bounds[0])

    c = Schechter(mag_bounds[0])

    for i in range(len(mag_others)):
        flag = True
        while(flag):
            temp_mag = np.random.uniform(mag_bounds[1],mag_bounds[0])
            att      = np.random.uniform(0, c)
            bound    = Schechter(temp_mag)
            if(att < bound):
                mag_others[i] = temp_mag
                flag = False

    magnitudes  = np.concatenate([mag_hosts, mag_others])
    data['mag'] = magnitudes

    data.to_csv('m-catalog.txt', header=True, index=False, sep='\t')
