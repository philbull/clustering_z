#!/usr/bin/env python
"""
Calculate Fisher matrix for photo-z / spectro-z cross-correlation.
"""
import numpy as np
import pylab as P
import pyccl as ccl
import clustering_z as clz
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d
import time

# Set up cosmology and LSST instrumental specs
cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045, h=0.67, A_s=2.1e-9, n_s=0.96)
chi = ccl.comoving_radial_distance(cosmo, a=1.) # Precomputes interpolation fn.

def selection_fid(zn, zbar, sigma_z0=0.03):
    """
    Placeholder fiducial selection function for the photo-z bin. For now, this 
    is just a Gaussian time dN/dz.
    """
    # FIXME FIXME FIXME
    zc = 0.5 * (zn[1:] + zn[:-1])
    sigma_z = sigma_z0 * (1. + zbar)
    
    # Make a fake selection function, just for testing
    pdf = np.exp(-0.5*(zc - zbar)**2. / sigma_z**2.) \
        / np.sqrt(2.*np.pi) / sigma_z
    phi = clz.dNdz_lsst(zc) * pdf
    
    # Normalise
    norm = np.sum(phi)
    return phi / norm


def pk_integral(cosmo, cache_name="pkint_cache"):
    """
    Perform cumulative integral of P(k) and interpolate.
    """
    k = np.logspace(-4., 1., 1000)
    
    # Load cache file if it exists
    try:
        # Load from cache
        pkint = np.load("%s.npy" % cache_name)
    except:
        # Calculate power spectrum and perform cumulative integral
        pk = ccl.linear_matter_power(cosmo, k, a=1.)
        pkint = cumtrapz(pk, k, initial=0.)
        
        # Save to cache file
        np.save(cache_name, pkint)
    
    # Construct interpolation function
    ipkint = interp1d(np.log(k), pkint, kind='linear')
    ifn = lambda k: ipkint(np.log(k))
    return ifn


def limber_k(l, z):
    """
    Return the k bounds using the Limber relation
    """
    a = 1. / (1. + z)
    chi = ccl.comoving_radial_distance(cosmo, a.flatten()).reshape(z.shape)
    return (l + 0.5) / chi


def pkdiff(ell, zs):
    """
    Power spectrum integral for each redshift bin.
    """
    kl = limber_k(ell, zs)
    pkl = pkint(kl)
    return -1.*np.diff(pkl, axis=0)


def Csignal(cosmo, ells, zp, zs, zn):
    """
    Construct signal covariance matrix for photo-z x spectro-z analysis.
    """
    zpc = 0.5 * (zp[1:] + zp[:-1])
    zsc = 0.5 * (zs[1:] + zs[:-1])
    znc = 0.5 * (zn[1:] + zn[:-1])
    Np = zpc.size; Ns = zsc.size; Nn = znc.size
    
    # Bias factors
    bs = clz.bias_HI(zsc) * clz.Tb(zsc)
    bp_zn = np.sqrt(1. + znc) # FIXME: Use proper bias function
    bp_zs = np.sqrt(1. + zsc) # FIXME
    Hzn = (100.*cosmo['h']) * ccl.h_over_h0(cosmo, 1./(1.+znc))
    Hzs = (100.*cosmo['h']) * ccl.h_over_h0(cosmo, 1./(1.+zsc))
    
    # P(k) integral over selection functions, assuming photo-z selection is 
    # broader than spectro-z selection
    ell2d, zs2d = np.meshgrid(ells, zs)
    ell2dn, zn2d = np.meshgrid(ells, zn)
    pkd = pkdiff(ell2d, zs2d).T
    pkdn = pkdiff(ell2dn, zn2d).T
    
    # Calculate selection function cross-terms (for p x s block only) (TESTING)
    # FIXME: Use proper selection_fid function
    cn_fid = [selection_fid(zn, zbar=_zpc, sigma_z0=0.03) for _zpc in zpc]
    rij = np.zeros((Np, Ns))
    for i in range(Np):
        for j in range(Ns):
            # Get the right c_n value for this j bin
            idx = np.where( np.logical_and(zsc[j] >= zn[:-1],
                                           zsc[j] <= zn[1:] ) )
            rij[i,j] = cn_fid[i][idx]
    
    # Build covariance matrix
    cs = np.zeros((ells.size, Np+Ns, Np+Ns))
    
    # photo x photo
    for i in range(Np):
        for j in range(Np):
            y = cn_fid[i] * cn_fid[j] * (bp_zn * Hzn)**2. * pkdn
            cs[:,i,j] = cs[:,j,i] = np.sum(y, axis=1)
    
    # spectro x spectro (diagonal only)
    for j in range(Ns):
        cs[:,Np+j,Np+j] = (Hzs[j] * bs[j])**2. * pkd[:,j]
    
    # spectro x photo
    for i in range(Np):
        cs[:,i,Np:] = cs[:,Np:,i] = Hzs**2. * bp_zs[i] * bs * pkd[:,:]
    
    # Multiply all elements by 2/(2l+1)
    fac = np.atleast_3d( 2. / (2.*ells + 1.) )
    fac = np.swapaxes(fac, 1, 0)
    cs *= fac
    
    # Apply selection functions to cross-terms
    for i in range(Np):
        for j in range(Ns):
            cs[:,i,Np+j] *= rij[i,j]
            cs[:,Np+j,i] *= rij[i,j]
    
    return cs, rij


def corr(c):
    """
    Calculate correlation matrix from covariance matrix.
    """
    corr = np.zeros(c.shape)
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            corr[i,j] = c[i,j] / np.sqrt(c[i,i]*c[j,j])
    return corr

# Start timing
t0 = time.time()

# Define spectroscopic redshift bins
#ells = np.array([10., 100., 1000.])
ells = np.arange(10, 1000, 1)
zp = np.array([1.0, 1.3, 1.6, 1.9]) # Photo-z bins
zs = np.arange(0.8, 2.4, 0.01) # Spectro-z bins
zn = np.arange(zs.min(), zs.max()+0.1, (zs[1]-zs[0])*3) # Photo-z sub-bins

# Integral of P(k)
pkint = pk_integral(cosmo)

# Calculate signal covariance and 
cs, rij = Csignal(cosmo, ells, zp, zs, zn)

print("Run took %2.1f sec." % (time.time() - t0))

# Plot matrix
P.matshow(corr(cs[1]), vmin=-1., vmax=1., cmap='RdBu')
##P.matshow(cs[1])
#P.matshow(rij)
P.colorbar()
P.show()

