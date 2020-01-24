#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from schechter_function import *
import matplotlib.pyplot as plt

'''
Given a catalog and a subset of this catalog (the hosts - 'nev' objects),
this program randomly gives to each entry of this catalog a magnitude (for example in the B band)
using a Gaussian distribution for hosts and Schechter function for other galaxies.

Please notice that the hosts are required to be the first nev entries of the catalog.
'''
def gaussian(x,x0,sigma):
    return np.exp(-(x-x0)**2/(2*sigma**2))/(sigma*np.sqrt(2*np.pi))

cat = open('catalog50k.txt', 'w')
cat.write('evid\tra\tdec\tLD\tmag\n')

events     = np.genfromtxt('2015_all_inj.txt', names = True)
detections = np.genfromtxt('mdc_catalogs_gwcosmo_catalogs_mdc_counterparts.txt', names = True)

det_ID = detections['simulation_ID']

nev    = 250

# Schechter parameters as in O2 H0 paper (from Gehrels et al. (2016))
alpha   = -1.07
Mstar   = -20.47

# Gaussian Parameters
m_mean  = -20.
m_sigma = 0.5

# Draw nev luminosities for hosts (gaussian distributed)

mag_others = np.zeros(len(events))
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

magnitudes  = mag_others

for index, id in zip(range(len(magnitudes)), events['simid']):
    if id in det_ID:
        magnitudes[index] = np.random.normal(m_mean, m_sigma, 1)


for m, ra, dec, LD, id  in zip(magnitudes, events['RA'], events['DEC'], events['LD'], events['simid']):
    cat.write('%f\t%f\t%f\t%f\t%f\n' %(id, ra, dec, LD, m))
cat.close()

# Optional: check the output distribution
output = True
if(output):
    plt.figure(1)
    plt.subplot(211)
    count, mbin, patch = plt.hist(magnitudes, density = True, bins = int(np.sqrt(len(magnitudes))))
    app = np.linspace(min(mag_bounds),max(mag_bounds))
    plt.plot(app, Schechter(app), label = 'Schechter')
    plt.legend(loc=0)
    meanbin = np.zeros(len(mbin)-1)
    for i in range(len(mbin)-1):
        meanbin[i] = (mbin[i+1]+mbin[i])/2.
    plt.subplot(212)
    plt.plot(meanbin, count-Schechter(meanbin))
    plt.savefig('magnitude_histogram_50k.pdf')
