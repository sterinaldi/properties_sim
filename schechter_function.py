"""
Module with Schechter magnitude function:
(C) Walter Del Pozzo (2014)

Modified by SR
"""
from numpy import *
from scipy.integrate import quad
import numpy as np

# format (Mstar_obs, alpha, mmin, mmax)
schechter_function_params = { 'B':(-20.457,-1.07),
                              'K':(-23.55,-1.02)}

class SchechterMagFunctionInternal(object):
    """
    Returns a Schechter magnitude function fort a given set of parameters

    Parameters
    ----------
    Mstar_obs : observed characteristic magnitude used to define
                Mstar = Mstar_obs + 5.*np.log10(H0/100.)
    alpha : observed characteristic slope.
    phistar : density (can be set to unity)
    """
    def __init__(self, Mstar, alpha, mmin, mmax, phistar=1.):
        self.Mstar = Mstar
        self.phistar = phistar
        self.alpha = alpha
        self.mmin = mmin
        self.mmax = mmax
        self.norm = None

    def evaluate(self, m):
        return 0.4*log(10.0)*self.phistar \
               * pow(10.0, -0.4*(self.alpha+1.0)*(m-self.Mstar)) \
               * exp(-pow(10, -0.4*(m-self.Mstar)))

    def normalise(self):
        if self.norm is None:
            self.norm = quad(self.evaluate, self.mmin, self.mmax)[0]

    def pdf(self, m):
        self.normalise()
        return self.evaluate(m)/self.norm


def SchechterMagFunction(mmin, mmax, H0=70., band='B'):
    """
    Returns a Schechter magnitude function fort a given set of parameters

    Parameters
    ----------
    H0 : Hubble parameter in km/s/Mpc (default=70.)
    band : Either B or K band magnitude to define SF params (default='B').

    Example usage
    -------------

    smf = SchechterMagFunction(H0=70., band='B')
    (integral, error) = scipy.integrate.quad(smf)
    """
    if band == 'constant': # Perform incompleteness correction using B-band SF for constant luminosity weights
        band = 'B'
    Mstar_obs, alpha = schechter_function_params[band]
    Mstar = Mstar_obs + 5.*np.log10(H0/100.)
    smf = SchechterMagFunctionInternal(Mstar, alpha, mmin, mmax)
    return smf.pdf


def M_Mobs(H0, M_obs):
    """
    Given an observed absolute magnitude, returns absolute magnitude
    """
    return M_obs + 5.*np.log10(H0/100.)
