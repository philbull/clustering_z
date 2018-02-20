#!/usr/bin/env python
"""
Calculate photo-z selection functions using a simple two-Gaussian model.
"""
import numpy as np
import pylab as P
from scipy.integrate import simps, quad, cumtrapz
from scipy.interpolate import interp1d
from clustering_z import zbins_lsst_alonso, dNdz_lsst

# Photo-z pdf settings
dzc = 0. #0.01
dzt = 0.2
sigt_fac = 1. #5.
ptail = 0. #0.05

def pdf_2gaussian(zp, zs, dzc, dzt, sigma_c, sigma_t, ptail):
    """
    Photo-z pdf.
    """
    pdf_c = np.exp(-0.5*((zp - zs + dzc)/sigma_c)**2.) \
          / np.sqrt(2.*np.pi) / sigma_c
    pdf_t = np.exp(-0.5*((zp - zs + dzt)/sigma_t)**2.) \
          / np.sqrt(2.*np.pi) / sigma_t
    return (1. - ptail) * pdf_c + ptail * pdf_t


def norm_selection_fn(zs, selfn):
    """
    Normalise a binned selection function to have an integral of unity.
    """
    norm = np.sum( np.atleast_2d(selfn) * (zs[1] - zs[0]), axis=1)
    return (selfn.T / norm).T


def binned_selection_fn(zs, zbins, selfn):
    """
    Calculate fiducial values for binned selection functions.
    """
    # Interpolate the cumulative integral of selection function
    cumul_selfn = interp1d(zs, cumtrapz(selfn, zs, initial=0.), kind='linear')
    
    # Get mean value of selection function in each redshift bin
    y = cumul_selfn(zbins)
    sel_bins = np.diff(y) / np.diff(zbins)
    
    #print("Warning: selfns.binned_selection_fn has interpolation issue.")
    return sel_bins


def calc_selection_fn(zs, zmin, zmax, pdf=pdf_2gaussian, sigma_z0=0.03, 
                      normed=False):
    """
    Calculate photo-z selection function by integrating photo-z pdf over z_phot.
    """
    # Photo-z redshifts and bin centre
    zp = np.linspace(zmin, zmax, 500)
    zc = 0.5 * (zmin + zmax)
    sigma_z = sigma_z0 * (1. + zc)
    
    # Set photo-z pdf function and parameters
    pz_params = {
        'dzc':      dzc, 
        'dzt':      dzt, 
        'sigma_c':  sigma_z, 
        'sigma_t':  sigt_fac*sigma_z, 
        'ptail':    ptail
    }
    
    # Perform integral over zp for each zs
    ZS, ZP = np.meshgrid(zs, zp)
    integ = pdf(ZP, ZS, **pz_params)
    selfn = simps(integ, zp, axis=0)
    
    # Apply normalisation
    if normed:
        norm = simps(selfn, zs)
        selfn /= norm
    return selfn


if __name__ == '__main__':

    # Define photo-z bins
    zmin, zmax = zbins_lsst_alonso(nbins=15, sigma_z0=0.03)
    
    # Define spectroscopic sampling
    zs = np.linspace(0., 3.6, 10000)

    # Calculate selection functions
    selfns = [ calc_selection_fn(zs, zmin[i], zmax[i], pdf=pdf_2gaussian) 
               for i in [8,] ]
    selfns = np.array(selfns)

    # Calculate binned selection function
    zbins = np.arange(0., 3.+0.01, 0.01)
    zbinsc = 0.5 * (zbins[1:] + zbins[:-1])
    sel_bins = binned_selection_fn(zs, zbins, selfns[0])

    # Plot selection functions
    P.subplot(111)

    for i in range(selfns.shape[0]):
        P.plot(zs, selfns[i], lw=1.8)

    P.step(zbinsc, sel_bins, where='mid', color='r', lw=1.8)
    P.plot(zbinsc, sel_bins, 'gx')

    #P.plot(zs, dNdz_lsst(zs), 'k-', lw=1.8)

    P.xlim((0., 3.4))
    #P.yscale('log')
    P.tight_layout()
    P.show()
