#!/usr/bin/python
"""
Plot cross-correlation between photometric survey and IM survey.
"""
import numpy as np
import pylab as P
import pyccl as ccl
import matplotlib.ticker
from multiprocessing import Pool

NTHREADS = 2 # Doesn't do anything yet

C = 29979.2458 # Speed of light, km/s
INF_NOISE = 1e100

# Set up cosmology and LSST instrumental specs
cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045, h=0.67, A_s=2.1e-9, n_s=0.96)

def kperp_fg(z, xi):
    """
    Effective radial scale beyond which correlated foregrounds begin to mask 
    the cosmological signal. (From Eq. 11 of Alonso et al.)
    """
    Hz = 100.*cosmo['h'] * ccl.h_over_h0(cosmo, 1./(1.+z)) # km/s/Mpc
    k_FG = np.pi * Hz / ((1. + z) * C * xi)


def Tb(z):
    """
    Brightness temperature Tb(z), in mK. Uses a simple power-law fit to Mario's 
    updated data (powerlaw M_HI function with alpha=0.6)
    """
    return 5.5919e-02 + 2.3242e-01*z - 2.4136e-02*z**2.
    

def bias_HI(z):
    """
    b_HI(z), obtained using a simple polynomial fit to Mario's data.
    """
    return 6.6655e-01 + 1.7765e-01*z + 5.0223e-02*z**2.


def calculate_block_noise_int(ells, zmin, zmax):
    """
    Approximate noise expression for a radio interferometer, using Eq. 8 of 
    Alonso et al.
    """
    # Interferometer parameters
    d_min = 6.
    d_max = 300.
    sigma_T = 1e-3 # mK rad MHz^1/2
    
    # Frequency scaling
    zc = 0.5 * (zmin + zmax)
    nu = 1420. / (1. + zc)
    lam = (C * 1e3) / (nu * 1e6) # wavelength in m
    
    # Bin widths and angular cutoffs
    dnu = 1420. * (1./(1. + zmin) - 1./(1. + zmax))
    
    # Angular scaling
    _ell, _lam = np.meshgrid(ells, lam)
    f_ell = np.exp(_ell*(_ell+1.) * (1.22 * _lam / d_max)**2. \
          / (8.*np.log(2.)))
    
    # Construct noise covariance
    N_ij = f_ell * sigma_T**2. / dnu[:,None] # FIXME: Is this definitely dnu?
    
    # Apply large-scale cut
    N_ij[np.where(_ell*_lam/(2.*np.pi) <= d_min)] = INF_NOISE
    return N_ij.T
    

def selection_lsst(zmin, zmax, sigma_z0, debug=False):
    """
    Calculate tomographic redshift bin selection function and bias for LSST.
    """
    # Number counts/selection function in this tomographic redshift bin
    z = np.linspace(0., 3., 1000)
    pz_lsst = ccl.PhotoZGaussian(sigma_z0)
    tomo_lsst = ccl.dNdz_tomog(z, 'nc', zmin, zmax, pz_lsst)
    #tomo_lsst = np.exp(-0.5*(z - 0.5)**2. / sigma_z0**2.)
    
    # Clustering bias
    bz_lsst = ccl.bias_clustering(cosmo, 1./(1.+z))

    # Number density tracer object
    n_lsst = ccl.ClTracerNumberCounts(cosmo, 
                                      has_rsd=False, has_magnification=False, 
                                      n=(z, tomo_lsst), bias=(z, bz_lsst))
    if debug:
        return z, tomo_lsst, bz_lsst, n_lsst
    else:
        return n_lsst


def selection_im(zmin, zmax, debug=False, scale=1.):
    """
    Calculate tomographic redshift bin selection function and bias for an 
    IM experiment. Each factor of this tracer will have units of mK.
    """
    # Number counts/selection function in this tomographic redshift bin
    z = np.linspace(zmin*0.9, zmax*1.1, 500) # Pad zmin, zmax slightly
    tomo_im = np.zeros(z.size)
    tomo_im[np.where(np.logical_and(z >= zmin, z < zmax))] = 1.
    
    # Clustering bias and 21cm monopole temperature, in mK
    bz_im = scale * bias_HI(z) * Tb(z)
    
    # Number density tracer object
    n_im = ccl.ClTracerNumberCounts(cosmo, 
                                    has_rsd=False, has_magnification=False, 
                                    n=(z, tomo_im), bias=(z, bz_im))
    if debug:
        return z, tomo_im, bz_im, n_im
    else:
        return n_im


def calculate_block_fg(ells, zc):
    """
    Calculate a correlated foreground block, using the model from Eq. 10 of 
    Alonso et al. (arXiv:1704.01941).
    """
    # Foreground residual parameters
    A_fg = 1. # mK^2
    alpha = -2.7
    beta = -2.4
    xi = 0.01 # Frequency correlation scale
    
    # Pivot scales and frequency scaling
    l_star = 1000.
    nu_star = 130. # MHz
    nu = 1420. / (1. + zc)
    
    # Calculate angle-dep. factor
    f_ell = (ell / l_star)**beta
    
    # Frequency-dependent covariance factor
    _nu, _nuprime = np.meshgrid(nu, nu)
    f_nu = (_nu*_nuprime/nu_star**2.)**beta \
         * np.exp(-0.5 * np.log(_nu/_nuprime)**2. / xi**2.)
    
    # Take product and return
    Fij = f_ell[:,None,None] * f_nu
    return A_fg * Fij
    

def calculate_block_gen(ells, tracer1, tracer2):
    """
    Calculate a general (i.e. dense) block of cross-correlations.
    """
    Ni = len(tracer1)
    Nj = len(tracer2)
    Cij = np.zeros((ell.size, Ni, Nj))
    
    # Construct cross-correlation block
    for i in range(Ni):
        print " ", i
        for j in range(Nj):
            print "   ", j
            Cij[:,i,j] = ccl.angular_cl(cosmo, tracer1[i], tracer2[j], ells)
    return Cij


def calculate_block_diag(ells, tracer1):
    """
    Calculate an assumed-diagonal block of auto-correlations.
    """
    Ni = len(tracer1)
    Cij = np.zeros((ell.size, Ni))
    
    # Construct cross-correlation block
    for i in range(Ni):
        print " ", i
        Cij[:,i] = ccl.angular_cl(cosmo, tracer1[i], tracer1[i], ells)
    return Cij


def expand_diagonal(dmat):
    """
    Expand diagonal elements of matrix into full matrix.
    """
    mat = np.zeros((dmat.shape[0], dmat.shape[1], dmat.shape[1]))
    for i in range(dmat.shape[1]):
        mat[:,i,i] = dmat[:,i]
    return mat


def build_covmat(ells, tracer1, tracer2, zmin_im, zmax_im):
    """
    Build full covariance matrix from individual blocks.
    N.B. tracer2 should be the IM tracer!
    """
    # Number of tracer redshift bins
    N1 = len(tracer1)
    N2 = len(tracer2)
    
    # Define angular scales and redshift bins
    zc = 0.5 * (zmin_im + zmax_im)
    
    # Calculate IM auto block
    print "signal im-im"
    Sij_im_im = calculate_block_diag(ell, tracer2)

    # Calculate LSST auto block
    print "signal photoz-photoz"
    Sij_pz_pz = calculate_block_gen(ell, tracer1, tracer1)
    
    # Calculate cross-tracer signal block
    print "signal photoz-im"
    Sij_pz_im = calculate_block_gen(ell, tracer1, tracer2)
    
    # Calculate IM noise auto block
    Nij_im_im = calculate_block_noise_int(ell, zmin_im, zmax_im)
    
    # Calculate LSST noise auto block
    # TODO
    Nij_pz_pz = 0.
    
    # Calculate IM foreground residual auto block
    print "foreground im-im"
    Fij_im_im = calculate_block_fg(ell, zc)
    
    # Construct total covariance matrix
    Cij = np.zeros((ells.size, N1 + N2, N1 + N2))
    Cij[:,:N1,:N1] = Sij_pz_pz + Nij_pz_pz
    Cij[:,N1:,N1:] = expand_diagonal(Sij_im_im) \
                   + expand_diagonal(Nij_im_im) \
                   + Fij_im_im
    Cij[:,:N1,N1:] = Sij_pz_im
    Cij[:,N1:,:N1] = np.transpose(Sij_pz_im, axes=(0,2,1))
    return Cij


def corrmat(mat):
    """
    Construct correlation matrix
    """
    mat_corr = np.zeros(mat.shape)
    for ii in range(mat.shape[0]):
        for jj in range(mat.shape[0]):
            mat_corr[ii,jj] = mat[ii, jj] / np.sqrt(mat[ii,ii] * mat[jj,jj])
    return mat_corr


# Define angular scales and redshift bins
ell = np.arange(4, 100)
z_min = np.arange(0.2, 0.8, 0.05)
z_max = z_min + (z_min[1] - z_min[0])

# Define lists of tracers
tracer1 = [ selection_lsst(0.4, 0.6, sigma_z0=0.05), 
            selection_lsst(0.6, 0.8, sigma_z0=0.05) ]
tracer2 = [ selection_im(z_min[i], z_max[i]) for i in range(z_min.size) ]

# Build covariance matrix
Cij = build_covmat(ell, tracer1, tracer2, z_min, z_max)

# Plot correlation matrix for a given ell
print corrmat(Cij[80])[0,:]
print corrmat(Cij[80])[-1,:]
P.matshow(corrmat(Cij[80]), cmap='RdBu', vmin=-1., vmax=1.)
P.colorbar()
P.show()

"""
# Calculate x-corr across many spectroscopic redshift bins
ell = np.arange(4, 100)
z_min = np.arange(0.2, 0.8, 0.02)
z_max = z_min + (z_min[1] - z_min[0])
zc = 0.5 * (z_min + z_max)
xcorr = np.zeros((ell.size, z_min.size))

# Calculate angular cross-power in each z_spec bin
for i in range(z_min.size):
    print i
    z_im, tomo_im, bz_im, n_im = selection_im(z_min[i], z_max[i], debug=True)
    cls = ccl.angular_cl(cosmo, n_lsst, n_im, ell)
    xcorr[:,i] = cls * np.sqrt(2.*ell + 1.)

idx = np.where(xcorr == np.max(xcorr))
print "l_max = %d" % ell[idx[0]]
print "z_max = %2.2f" % zc[idx[1]]


z_im, tomo_im, bz_im, n_im = selection_im(z_min[i], z_max[i])
cls = ccl.angular_cl(cosmo, n_lsst, n_im, ell)



# Plot x-corr
P.matshow(xcorr / 1e-6, cmap='bone', aspect='auto', origin='lower',
          extent=(z_min[0], z_max[-1], ell[0], ell[-1]))

# Set axis labels
P.gca().xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.1))
P.tick_params(axis='both', which='major', labelsize=18, size=8.,
              width=1.25, pad=8., labelbottom='on', labeltop='off')

P.ylabel(r'$\ell$', fontsize=20., labelpad=10.)
P.xlabel(r'$z_{\rm spec}$', fontsize=20.)

P.colorbar()
#P.tight_layout()
P.show()
"""
