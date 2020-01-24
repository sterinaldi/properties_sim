#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import h5py
from os.path import splitext

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
from scipy.stats import gaussian_kde

from sklearn.mixture import GaussianMixture

def Gaussexp(x, mu, sigma):
    return -(x-mu)**2/(2*sigma**2)-2.0*np.pi*sigma

def HubbleLaw(D_L, omega): # Da rivedere: test solo 1 ordine
    return D_L*omega.h/(3e3) # Sicuro del numero?

def gaussian(x,x0,sigma):
    return np.exp(-(x-x0)**2/(2*sigma**2))/(sigma*np.sqrt(2*np.pi))
def LumDist(z, omega):
    return 3e3*(z + (1-omega.om +omega.ol)*z**2/2.)/omega.h

def dLumDist(z, omega):
    return 3e3*(1+(1-omega.om+omega.ol)*z)/omega.h

def RedshiftCalculation(LD, omega, zinit=0.3, limit = 0.001):
    '''
    Redshift given a certain luminosity, calculated by recursion.
    Limit is the less significative digit.
    '''
    LD_test = LumDist(zinit, omega)
    if abs(LD-LD_test) < limit :
        return zinit
    znew = zinit - (LD_test - LD)/dLumDist(zinit,omega)
    return RedshiftCalculation(LD, omega, zinit = znew)


def get_samples(file, names = ['ra','dec','luminosity_distance']):
    filename, ext = splitext(file)
    samples = {}

    if ext == '.json':
        with open(file, 'r') as f:
            data = json.load(f)

        post = np.array(data['posterior_samples']['SEOBNRv4pHM']['samples'])
        keys = data['posterior_samples']['SEOBNRv4pHM']['parameter_names']

        for name in names:
            index  = keys.index(name)
            samples[name] = post[:,index]

        return samples

    if ext == '.hdf5':
        f = h5py.File(file, 'r')
        data = f['lalinference_mcmc']['posterior_samples'][:]
        h5names = ['ra','dec','dist']

        for name, nameh5 in zip(names, h5names):
            samples[name] = data[nameh5]

        return samples

    if ext == '.dat':
        data = np.genfromtxt(file, names = True)
        dat_names = ['ra','dec','distance']

        for name, name_dat in zip(names, dat_names):
            samples[name] = data[dat_names]

        return samples



def pos_posterior(ra_s, dec_s, number = 2):
    func = GaussianMixture(n_components = number, covariance_type = 'full')
    samples = []
    for x,y in zip(ra_s, dec_s):
        samples.append(np.array([x,y]))
    func.fit_predict(samples)
    return func

def show_gaussian_mixture(ra_s, dec_s, mixture):
    x = np.linspace(min(ra_s), max(ra_s))
    y = np.linspace(min(dec_s), max(dec_s))
    X, Y = np.meshgrid(x,y)
    XX = np.array([X.ravel(), Y.ravel()]).T
    Z = -mixture.score_samples(XX)
    Z = Z.reshape(X.shape)
    CS = plt.contour(X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0),levels=np.logspace(0, 3, 10))
    CB = plt.colorbar(CS, shrink=0.8, extend='both')
    plt.scatter(ra_s,dec_s, 0.8)
    plt.show()

def LD_posterior(LD_s):
    return gaussian_kde(LD_s)


class ranking(cpnest.model.Model):

    def __init__(self, omega, z_bounds, catalog, id):
        self.names=['zgw']
        self.bounds=[z_bounds]
        self.omega   = omega
        self.catalog = catalog
        self.detection_id = id

    def dropgal(self):
        for i in self.catalog.index:
            if self.pLD(lal.LuminosityDistance(self.omega, self.catalog['z'][i])) < 0.0001:
                self.catalog = self.catalog.drop(i)

    def plot_outputs(self):
        plt.figure(1)
        plt.xlabel('ra')
        plt.ylabel('dec')
        plt.xlim([min(self.catalog['RAJ2000'])-0.1, max(self.catalog['RAJ2000'])+0.1])
        plt.ylim([min(self.catalog['DEJ2000'])-0.1, max(self.catalog['DEJ2000'])+0.1])
        S = plt.scatter(self.catalog['RAJ2000'], self.catalog['DEJ2000'], c = self.catalog['p'], marker = '+')
        bar = plt.colorbar(S)
        bar.set_label('p')
        plt.savefig('prob'+self.detection_id+'.pdf', bbox_inches = 'tight')

        plt.figure(2)
        S = plt.scatter(self.catalog['RAJ2000'], self.catalog['DEJ2000'], c = self.catalog['ppos'], marker = '+')
        bar = plt.colorbar(S)
        bar.set_label('p')
        plt.xlabel('ra')
        plt.ylabel('dec')
        plt.savefig('positionsprob'+self.detection_id+'.pdf', bbox_inches = 'tight')

    def log_prior(self,x):
        if not(np.isfinite(super(ranking, self).log_prior(x))):
            return -np.inf
        return 0.

    def log_likelihood(self, x):
        logL = 0.
        zgw = x['zgw']
        # Proper motion is here assumed to be gaussian (sigma ~10%)
        Lh = np.array([gaussian(zgw, zgi, zgi/10.0)*M.pLD(lal.LuminosityDistance(self.omega, zgi))*np.exp(M.p_pos.score_samples([[np.deg2rad(rai),np.deg2rad(di)]])[0])for zgi,rai,di in zip(self.catalog['z'],self.catalog['RA'],self.catalog['Dec'])])
        logL = np.log(Lh.sum())
        return logL

    def run(self, file, show_output = False, run_sampling = True):

        # posteriors GW calculation
        samples = get_samples(file = file)
        self.pLD = gaussian_kde(samples['luminosity_distance'])
        self.p_pos = pos_posterior(samples['ra'],samples['dec'], number = 1)
        probs = []
        for ra, dec in zip(self.catalog['RA'], self.catalog['Dec']):
            probs.append(np.exp(self.p_pos.score_samples([[ra,dec]]))[0])
        self.catalog['ppos'] = np.array(probs)
        # Dropping galaxies outside the confident volume
        # Position
        self.catalog = self.catalog[self.catalog['ppos'] > 0.01] # empirical!
        # Distance
        self.dropgal()
        # run
        job = cpnest.CPNest(self, verbose=1, nthreads=4, nlive=1000, maxmcmc=100)
        if run_sampling:
            job.run()
            posteriors = job.get_posterior_samples(filename = 'posterior.dat')
        # z posteriors calculation
        posteriors = np.genfromtxt('posterior_backup.dat', names = True)
        just_z = [post[0] for post in posteriors]
        self.pdfz = gaussian_kde(just_z)

        # Probability calculation and galaxy sorting
        prob = self.catalog['z'].apply(self.pdfz)
        prob = prob/prob.max()
        self.catalog['p'] = prob
        self.catalog = self.catalog.sort_values('p', ascending = False)
        self.catalog.to_csv('rank'+self.detection_id+'.txt', header=True, index=None, sep='&', mode='w')
        if show_output:
            self.plot_outputs()

if __name__ == '__main__':

    # Event selection

    # positions = 'posterior_samples.json'
    # positions = 'GW170817_GWTC-1.hdf5'
    # positions = 'posterior_samples_170817.dat'

    # To be fixed according to the expected luminosity distance
    z_bounds = [0.02,0.08]
    omega = lal.CreateCosmologicalParameters(0.7,0.3,0.7,-1.,0.,0.)
    M = ranking(omega, z_bounds)
    M.run(file = positions, run_sampling = False, show_output = True)
