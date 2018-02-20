#!/usr/bin/env python
#
#
# Python code to compute the "prior" covariance matrix of
# binned number counts for objects selected within a photo-z
# bin.  The photo-z nuisance parameters are marginalized over.
#
# This code is quite slow, but hopefully the brute-force approach
# makes the logic easy to follow.
#
#
from __future__ import print_function, division
import numpy as np
from clustering_z import dNdz_lsst, zbins_lsst_alonso #(nbins=15, sigma_z0=0.03)
from scipy.interpolate import interp1d
import time

np.random.seed(10)

class SpectroZ(object):
    """
    A class to define some variables defining the spectroscopic sample.
    """
    def __init__(self, selection_fn=None, zlo=0.8, zhi=2.5, ngals=1e5, dz=0.1):
        self.zlo   = zlo			# Model range after CHIME.
        self.zhi   = zhi			# Model range after CHIME.
        self.ngals = int(ngals)  	# Total number of galaxies.
        
        # True z bin edges which define our N_i.
        self.edges = np.arange(self.zlo, self.zhi+0.1*dz, dz)
        
        self.z_cdf = None
        if selection_fn is not None:
            self.z_cdf = self.selection_cdf(selection_fn)
    
    def selection_cdf(self, selection_fn, zlow=0., zhigh=4.):
        """
        Cumulative distribution function derived from dN/dz curve. Returns the 
        inverted relation, z(cdf), instead of z(cdf).
        """
        z = np.linspace(zlow, zhigh, 1000)
        sel = selection_fn(z)
        
        # Calculate cdf
        cdf = np.cumsum(sel)
        cdf /= cdf[-1] # Rescale to interval [0, 1]
        
        # Interpolate cdf
        cdf_i = interp1d(cdf, z, kind='linear')
        return cdf_i
        
    
    def zspec(self):
        """
        Returns spectroscopic redshifts, following the true redshift 
        distribution of the survey.
        """
        if self.z_cdf is None:
            raise ValueError("No selection function was provided for SpectroZ")
        
        zs = np.random.uniform(low=self.zlo, high=self.zhi, size=self.ngals)
        return zs # FIXME
        
        # Sample spectroscopic redshifts using the inverted cdf of the 
        # selection function
        u = np.random.uniform(low=0., high=1., size=self.ngals)
        zs = self.z_cdf(u)
        return(zs)


class PhotoZ(object):
    """
    A class to define P(zphoto|zspectro).
    Currently modeled as two Gaussians (core and tail) and a relative amplitude.
    At present only a single offset (for the mean) is variable.
    Also holds information on the photo-z cut which is applied to the
    sample of interest.
    """
    def __init__(self, dzc, dzt=None, sigc=0.03, sigt=0.3, ptail=0.05, 
                 zrange=[1.3, 1.6]):
        self.sigc  = sigc	# Width of core.
        self.sigt  = sigt	# Width of tails.
        self.ptail = ptail	# Probability of an outlier/tail.
        self.zr    = zrange	# Photo-z bin to select galaxies in.
        self.dzc   = dzc
        self.dzt   = dzt if dzt is not None else dzc
    
    def zphot(self, zs):
        """
        Returns photometric redshifts, given z_spectro.
        """
        # We model the core as a Gaussian, offset by dzc and of width sigc.
        zcore= self.dzc+self.sigc*np.random.normal(loc=0, scale=1, size=zs.size)
        
        # The tail, or catastrophic failures, are currently modeled by a
        # Gaussian but better choices would be something offset, or
        # asymmetrical, something with tails (e.g. Lorentzian) or even a
        # uniform distribution.
        ztail= self.dzt+self.sigt*np.random.normal(loc=0, scale=1, size=zs.size)
        
        # Now just produce a mixture of core and tail shifts.  We can also
        # make ptail depend on zs (as a proxy for luminosity for example).
        pint = np.random.binomial(n=1, p=self.ptail, size=zs.size)
        zp   = zs + (1. - pint)*zcore + pint*ztail
        return(zp)
    
    def pz(self, zs):
        """
        Model the photo-z pdf as a function of zs.
        """
        def gaus(x, mu, sigma): 
            return np.exp(-0.5 * (x - mu)**2. / sigma**2.) \
                 / np.sqrt(2.*np.pi) / sigma
        
        return  (1. - self.ptail) * gaus(zs, self.dzc, self.sigc) \
              + self.ptail * gaus(zs, self.dzt, self.sigt)
        
def make_one(S, P):
    """
    Given a spectro set up (S) and photo-z parameters (P), return one
    realization of the number of galaxies in each spectroscopic bin
    (with edges defined in S) when selected on photo-z bin (defined in P).
    """
    # Start with a uniform distribution of spectroscopic galaxies.
    zs   = S.zspec()
    
    # Generate the photo-z for each galaxy.
    zp   = P.zphot(zs)
    
    # Select galaxies in a top-hat bin of photo-z.
    ww   = np.nonzero( (zp > P.zr[0]) & (zp < P.zr[1]) )[0]
    
    # and compute the number of galaxies in each "true/spectroscopic" bin.
    y,x  = np.histogram(zs[ww], bins=S.edges, normed=True)
    return(y)


def make_cov(S, priors, pzbin, Nmc=1000):
    """
    Generate many realizations of the N_i, each with a different set of
    photo-z parameters, and determine the covariance by Monte-Carlo.
    """
    P = PhotoZ(0.0)
    y = make_one(S,P)
    Ns= np.empty( (Nmc, y.size) )
    
    # Get photo-z bin parameters
    zrange = pzbin['zrange']
    
    zs = np.linspace(-1., 1., 200)
    pz_pdf = np.zeros((zs.size, Nmc))
    
    # Perform Nmc Monte Carlo runs
    for i in range(Nmc):
        if i % 500 == 0: print("  %d / %d" % (i, Nmc))
        
        # Draw hyperparameters from hyperpriors
        ptail = np.random.uniform(low=priors['ptail_min'], 
                                  high=priors['ptail_max'])
        dzc = np.random.normal(loc=0., scale=priors['sigma_dzc'])
        dzt = np.random.normal(loc=priors['mean_dzt'], 
                               scale=priors['sigma_dzt'])
        sigc = np.random.normal(loc=priors['mean_sigc'], 
                                scale=priors['sigma_sigc'])
        sigt = np.random.normal(loc=priors['mean_sigt'], 
                                scale=priors['sigma_sigt'])
        
        # Create new PhotoZ object
        P = PhotoZ(dzc=dzc, dzt=dzt, sigc=sigc, sigt=sigt, ptail=ptail, 
                   zrange=zrange)
        
        pz_pdf[:,i] = P.pz(zs)
        
        # Draw a selection function given this set of PhotoZ parameters
        Ns[i,:] = make_one(S, P)
    
    avg = np.mean(Ns,axis=0)
    cov = np.cov(Ns,rowvar=False)
    
    pct = np.arange(101)
    pcts = [np.percentile(Ns, _p, axis=0) for _p in pct]
    
    return( (avg, cov, np.array(pcts)) )
    #

def selection_uniform(z):
    """
    Uniform spectroscopic selection function.
    """
    return 0.*z + 1.


if __name__ == "__main__":
    t0 = time.time()
    
    # Set up MPI
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    myid = comm.Get_rank()
    size = comm.Get_size()
    
    #S = SpectroZ(selection_fn=dNdz_lsst)
    S = SpectroZ(selection_fn=selection_uniform, zlo=0., zhi=3., 
                 ngals=1e5, dz=0.01)
    
    zmin, zmax = zbins_lsst_alonso(nbins=15, sigma_z0=0.03)
    for i in [10,]: #range(zmin.size):
        if i % size != myid: continue
        
        print("Bin %d (%2.2f -- %2.2f) [worker %d]" % (i, zmin[i], zmax[i], myid))
        zc = 0.5 * (zmin[i] + zmax[i])
        
        # Define redshift bin parameters
        pzbin = {'zrange': [zmin[i], zmax[i]]}
        sigma_z = 0.03 * (1. + zc)
        
        # Define hyperprior parameters
        priors = {
            'ptail_min':    0.,
            'ptail_max':    0.1,
            'mean_dzt':     0.,
            'mean_sigc':    sigma_z,
            'mean_sigt':    5.*sigma_z,
            'sigma_dzc':    0.1,
            'sigma_dzt':    0.1,
            'sigma_sigc':   0.1*sigma_z,
            'sigma_sigt':   0.5*sigma_z,
        }
        
        # Estimate covariance by Monte Carlo
        avg, cov, pcts = make_cov(S, priors, pzbin, Nmc=10000)
        
        # Store results
        np.save("output/zlsst_zphot_dz001_avg_%d" % i, avg)
        np.save("output/zlsst_zphot_dz001_cov_%d" % i, cov)
        np.save("output/zlsst_zphot_dz001_pcts_%d" % i, pcts)
        
    print("Worker %d finished in %d sec." % (myid, time.time() - t0))
    comm.barrier()
    
