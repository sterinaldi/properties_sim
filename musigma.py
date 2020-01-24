#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier
from astroquery.simbad import Simbad
import astropy.units as u
import lal

from scipy import interpolate
from scipy.special import logsumexp
import cpnest, cpnest.model

folder = 'path/to/folder/'

def gaussian(x,x0,sigma):
    return np.exp(-(x-x0)**2/(2*sigma**2))/(sigma*np.sqrt(2*np.pi))

def Gaussexp(x, mu, sigma):
    return -(x-mu)**2/(2*sigma**2)-2.0*np.pi*sigma

class event:
    def __init__(self, mags, logposts):
        self.mag = mags
        self.GWlogpost = logposts

class mu_sigma(cpnest.model.Model):

    def __init__(self, events, mu_bounds = [-24,-17], sigma_bounds = [0.0001,2]):

        self.names  = ['mu', 'sigma']
        self.bounds = [mu_bounds, sigma_bounds]
        self.events = events

    def log_prior(self,x):
        if not(np.isfinite(super(ranking, self).log_prior(x))):
            return -np.inf
        # Flat prior is assumed
        return 0.


    def log_likelihood(self, x):
        logL = 0.
        mu    = x['mu']
        sigma = x['sigma']
        for e in self.events:
            logL  += logsumexp(e.GWlogpost+Gaussexp(e.mag, mu, sigma))
        return logL

if __name__ == '__main__' :

    mag_file = 'path/to/mags/'
    magnitudes = np.genfromtxt(mag_file, names = True)
    mags_array = magnitudes['mag']
    events = np.empty(250, dtype = event)

    for i in range(250):
        j = i+1
        path_to_data = folder + str(j)
        data = np.genfromtxt(path_to_data + '/galaxy_ranks.txt', names = True)
        events[i].mag       = mags_array
        events[i].GWlogpost = data['logposterior']

    W = mu_sigma(events)
    job = cpnest.CPNest(W, verbose=1, nthreads=4, nlive=1000, maxmcmc=1024)
    job.run()
