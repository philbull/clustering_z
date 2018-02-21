#!/usr/bin/env python
"""
Calculate Fisher matrix for photo-z / spectro-z cross-correlation.
"""
import numpy as np
import pylab as P
import pyccl as ccl
import clustering_z as clz
import photoz_selection_fns as selfns
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d
import time
from mpi4py import MPI

C = 2.99792458e5 # km/s


def bias_lsst(z):
    """
    Bias of LSST galaxies, taken from p3 of arXiv:1704.01941.
    """
    return 1. + 0.84*z


#def selection_fid(zn, zbar, sigma_z0=0.03, coeffs=True):
#    """
#    Placeholder fiducial selection function for the photo-z bin. For now, this 
#    is just a Gaussian.
#    """
#    # FIXME FIXME FIXME
#    zc = 0.5 * (zn[1:] + zn[:-1])
#    sigma_z = sigma_z0 * (1. + zbar)
#    
#    # Make a fake selection function, just for testing
#    pdf = np.exp(-0.5*(zc - zbar)**2. / sigma_z**2.) \
#        / np.sqrt(2.*np.pi) / sigma_z
#    phi = clz.dNdz_lsst(zc) * pdf # dN/dz ~ galaxies / rad^2 / dz
#    
#    # Normalisation
#    # Currently assuming model where phi(z) = Sum_n c_n Theta_n(z), where
#    # Theta_n(z) = 1 / (Delta z_n) within bin n, and zero elsewhere. The integral 
#    # \int phi(z) dz = 1 then implies that sum_n c_n = 1.
#    if coeffs:
#        # Return coefficient for each photo-z sub-bin, c_n
#        return phi / np.sum(phi) # Sum of c_n coeffs. should be 1
#    else:
#        # Return total number of galaxies in photo-z bin (per unit area)
#        # N_tot = \int dN/dz p(z) dz ~ Sum(phi * dz)
#        return np.sum(phi) * (zn[1] - zn[0])


def centroids(x):
    """
    Return centroids of bins with bin edges 'x'.
    """
    return 0.5 * (x[1:] + x[:-1])

def xi_to_kpar(cosmo, z, xi):
    """
    Convert frequency correlation parameter, xi, to a k_parallel value.
    """
    c = 299792. # km/s
    
    # From Eq. 11 of arXiv:1704.01941
    Hz = (100.*cosmo['h']) * ccl.h_over_h0(cosmo, 1./(1.+z))
    kpar = np.pi * Hz / (c * xi * (1. + z)) # Mpc^-1
    return kpar


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


def binned_selection_fns(zp, zn, zs=None, pdf=selfns.pdf_2gaussian, 
                         sigma_z0=0.03, cache_name="selfn_cache"):
    """
    Calculate binned selection functions for all photo-z bins. Caches results.
    """
    # Load cache file if it exists
    try:
        # Load from cache
        dat = np.load("%s.npy" % cache_name)
        sel_fns = dat['sel_fns']; zs = dat['zs']
        print("\tLoaded binned selection functions from cache.")
    except:
        print("\tComputing binned selection functions.")
        if zs is None: zs = np.linspace(0., 3.6, 10000) # Spectroscopic sampling
        
        # Calculate selection fns from photo-z pdf
        sel_fns = [ selfns.calc_selection_fn(zs, zp[i], zp[i+1], 
                                             pdf=pdf, sigma_z0=sigma_z0) 
                   for i in range(zp.size-1) ]
        sel_fns = np.array(sel_fns)
        np.savez(cache_name, sel_fns=sel_fns, zs=zs)
    
    # Calculate binned selection function
    sel_bins = [ selfns.binned_selection_fn(zs, zn, sel_fns[i]) 
                 for i in range(zp.size-1) ]
    return np.array(sel_bins)


def limber_k(cosmo, l, z):
    """
    Return the k bounds using the Limber relation
    """
    a = 1. / (1. + z)
    chi = ccl.comoving_radial_distance(cosmo, a.flatten()).reshape(z.shape)
    return (l + 0.5) / chi


def pkdiff(cosmo, pkint, ell, zs):
    """
    Power spectrum integral for each redshift bin.
    """
    kl = limber_k(cosmo, ell, zs)
    pkl = pkint(kl)
    return -1.*np.diff(pkl, axis=0)


def Csignal(cosmo, ells, zp, zs, zn, sigma_z0=0.03):
    """
    Construct signal covariance matrix for photo-z x spectro-z analysis.
    """
    zpc = centroids(zp); zsc = centroids(zs); znc = centroids(zn)
    dzn = zn[1] - zn[0]; dzs = zs[1] - zs[0]
    Np = zpc.size; Ns = zsc.size; Nn = znc.size
    
    # Integral of P(k)
    pkint = pk_integral(cosmo)
    
    # Bias factors
    bs = clz.bias_HI(zsc) * clz.Tb(zsc)
    bp_zn = bias_lsst(znc) # FIXME: Use proper bias function
    bp_zs = bias_lsst(zsc) # FIXME
    Hzn = (100.*cosmo['h']) * ccl.h_over_h0(cosmo, 1./(1.+znc))
    Hzs = (100.*cosmo['h']) * ccl.h_over_h0(cosmo, 1./(1.+zsc))
    Dzn = ccl.growth_factor(cosmo, 1./(1.+znc))
    Dzs = ccl.growth_factor(cosmo, 1./(1.+zsc))
    
    # P(k) integral over selection functions, assuming photo-z selection is 
    # broader than spectro-z selection
    ell2d, zs2d = np.meshgrid(ells, zs)
    ell2dn, zn2d = np.meshgrid(ells, zn)
    pkd = pkdiff(cosmo, pkint, ell2d, zs2d).T
    pkdn = pkdiff(cosmo, pkint, ell2dn, zn2d).T
    
    # Calculate binned selection functions
    sel_fns = binned_selection_fns(zp, zn, sigma_z0=sigma_z0)
    dNdz = clz.dNdz_lsst(znc) * sel_fns # dN/dz ~ galaxies / rad^2 / dz
    cn_fid = selfns.norm_selection_fn(zn, dNdz) # Normalise
    """
    # Calculate selection function cross-terms (for p x s block only)
    # (When zn = zs, we should recover rij = cn_fid)
    rij = np.zeros((Np, Ns))
    for i in range(Np):
        for j in range(Ns):
            # Get the right c_n value for this j bin
            idx = np.where( np.logical_and(zsc[j] >= zn[:-1],
                                           zsc[j] <= zn[1:] ) )
            rij[i,j] = cn_fid[i][idx]
    """
    
    # Build covariance matrix
    cs = np.zeros((ells.size, Np+Ns, Np+Ns))
    
    # photo x photo
    for i in range(Np):
        for j in range(Np):
            y = cn_fid[i] * cn_fid[j] * (Dzn*bp_zn*Hzn/C)**2. * pkdn #/ (dzn)**2.
            cs[:,i,j] = cs[:,j,i] = np.sum(y, axis=1)
    
    # spectro x spectro (diagonal only)
    # FIXME: *** Should remove dzs factors etc? ***
    for j in range(Ns):
        cs[:,Np+j,Np+j] = (Dzs[j] * Hzs[j] * bs[j]/C)**2. * pkd[:,j] #/ (dzs)**2.
    
    # spectro x photo (don't apply selection functions til later)
    for i in range(Np):
        cs[:,i,Np:] = \
        cs[:,Np:,i] = cn_fid[i,:] * (Dzs * Hzs / C)**2. * bp_zs[i] * bs \
                    * pkd[:,:] #/ (dzn * dzs) # FIXME
    
    # Multiply all elements by 2/(2l+1)
    fac = np.atleast_3d( 2. / (2.*ells + 1.) )
    fac = np.swapaxes(fac, 1, 0)
    cs *= fac
    
    # Apply selection functions to cross-terms
    #for i in range(Np):
    #    for j in range(Ns):
    #        cs[:,i,Np+j] *= rij[i,j] / (dzn * dzs)
    #        cs[:,Np+j,i] *= rij[i,j] / (dzn * dzs)
    
    return cs, cn_fid #, dNdz


def Cn_photoz(cosmo, ells, zp, zn, sigma_z0=0.03):
    """
    Construct noise covariance for photo-z survey.
    """
    zpc = centroids(zp); znc = centroids(zn)
    
    # Calculate binned selection functions
    sel_fns = binned_selection_fns(zp, zn, sigma_z0=sigma_z0)
    phi = clz.dNdz_lsst(znc) * sel_fns # dN/dz ~ galaxies / rad^2 / dz
    Nz = np.sum(phi, axis=1) * (zn[1] - zn[0]) # sum over photo-z bin
    
    # Construct covariance matrix with the right shape
    cn = np.zeros((ells.size, zpc.size, zpc.size))
    for i in range(zpc.size): cn[:,i,i] = 1. / Nz[i]
    return cn


def fisher_cn(inst, cosmo, ells, zp, zs, zn, kmax0=0.14, xi=0.1):
    """
    Calculate Fisher matrix for {c_n} parameters (ignore derivative of pz x pz).
    """
    Np = zp.size - 1; Ns = zs.size - 1

    # Calculate signal covariance and photo x spectro coeffs
    cs, cn_fid = Csignal(cosmo, ells, zp, zs, zn)

    # Calculate photo-z noise covariance
    cn_pz = Cn_photoz(cosmo, ells, zp, zn)

    # Calculate IM noise covariance
    expt = clz.setup_expt_info(cosmo, inst, kmax=kmax0, 
                               ignore_photoz_corr=False, sigma_z0=sigma_z0)
    cn_im = clz.calculate_block_noise_im(expt, ells, zs[:-1], zs[1:])
    cn_im = clz.expand_diagonal(cn_im)

    # Calculate IM foregrounds (FIXME)
    zsc = centroids(zs)
    cfg = 1.*clz.calculate_block_fg(cosmo, ells, zsc, xi=xi) # k_par ~ 0.01 Mpc^-1

    # Sum covariances
    cov = cs.copy()
    cov[:,:Np,:Np] += cn_pz
    cov[:,Np:,Np:] += cn_im + cfg

    # Invert total covariance matrix
    cinv = clz.invert_covmat(cov)

    # Calculate derivative of signal cov w.r.t. photo-z sub-bin params
    cs_deriv = np.zeros((Ns,) + cs.shape)
    for j in range(Ns):
        if j % 10 == 0: print("\t%d" % j)
        
        # Build derivative matrix
        dCs = np.zeros(cs.shape)
        dCs[:,:Np,Np+j] = cs[:,:Np,Np+j] / cn_fid[:,j] # dC/dc_n ~ C / c_n
        dCs[:,Np+j,:Np] = dCs[:,:Np,Np+j]
        
        # C^-1 dC/dc_n
        cs_deriv[j] = np.array([ np.dot(cinv[l], dCs[l]) 
                                 for l in range(cs.shape[0]) ])
        
    # Shape: (N_ell, Ns, Np+Ns, Np+Ns)
    cs_deriv = np.swapaxes(cs_deriv, 1, 0)

    # Build Fisher matrix
    F = np.zeros((ells.size, Ns, Ns))
    for l in range(ells.size):
        if l % size != myid: continue
        print("\tell = %d" % ells[l])
        for i in range(Ns):
            if i % 10 == 0: print("\t    i=%d" % i)
            for j in range(i, Ns):
                F[l,i,j] = np.trace( np.dot(cs_deriv[l,i], cs_deriv[l,j]) )
                F[l,j,i] = F[l,i,j]
    
    # Combine all calculated F_ell at the root process
    F_all = np.zeros(F.shape)
    comm.Reduce(F, F_all, op=MPI.SUM)
    return F_all
    

def corr(c):
    """
    Calculate correlation matrix from covariance matrix.
    """
    corr = np.zeros(c.shape)
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            corr[i,j] = c[i,j] / np.sqrt(c[i,i]*c[j,j])
    return corr


#-------------------------------------------------------------------------------
if __name__ == '__main__':
    
    # Set up MPI
    comm = MPI.COMM_WORLD
    myid = comm.Get_rank()
    size = comm.Get_size()

    # Set up cosmology and LSST instrumental specs
    cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045, h=0.67, 
                          A_s=2.1e-9, n_s=0.96)
    chi = ccl.comoving_radial_distance(cosmo, a=1.) # Precomputes interpolation fn.

    # Start timing
    t0 = time.time()

    # Define spectroscopic redshift bins
    ells = np.arange(2, 600, 1)
    
    i = 10 # FIXME
    #zp = np.array([1.2, 1.3])
    zpmin, zpmax = clz.zbins_lsst_alonso(nbins=15, sigma_z0=0.03)
    zp = np.array([zpmin[i], zpmax[i]])

    #zs = np.arange(0., 3.01, 0.01) # Spectro-z bin edges
    zs = np.arange(0.8, 2.41, 0.01) # FIXME
    zn = zs

    if myid == 0:
        print "ell range (z=%2.2f) = %1.1f, %1.1f" \
            % ( zs[0], 
                clz.lmin_for_redshift(cosmo, zs[0], dmin=6.), 
                clz.lmax_for_redshift(cosmo, zs[0], kmax0=0.2) )
        print "ell range (z=%2.2f) = %1.1f, %1.1f" \
            % ( zs[-1], 
                clz.lmin_for_redshift(cosmo, zs[-1], dmin=6.), 
                clz.lmax_for_redshift(cosmo, zs[-1], kmax0=0.2) )

    # Assumed sigma_z0 for LSST
    sigma_z0 = 0.03
    KMAX0 = 0.14 # Mpc^-1

    # Example HIRAX interferometer
    inst_hirax = {
        'name':     "hrx",
        'type':     "interferometer",
        'd_min':    6., # m
        'd_max':    32.*6.*np.sqrt(2), # m
        'Ndish':    32*32,
        'fsky' :    0.4,
        'Tsys' :    50., # in K
        'ttot':     2.8e4, # hrs
        'fsky_overlap': 0.4,
    }

    F = fisher_cn(inst_hirax, cosmo, ells, zp, zs, zn, kmax0=0.2, xi=1000.)
    np.save("Fisher_LSSTxHIRAX_selfn", F)
    Ftot = np.sum(F, axis=0)

    # FIXME
    cov = np.linalg.pinv(Ftot[:50,:50])

    #-------------------------------------------------------------------------------
    print("Run took %2.1f sec." % (time.time() - t0))

    # Plot Fisher matrix
    if myid == 0:
        #P.matshow(np.log10(Ftot), cmap='Blues') #, vmin=-10., vmax=2.)
        #P.matshow(np.log10(cov), cmap='Blues')
        
        #P.matshow(corr(cov), cmap='RdBu', vmin=-1., vmax=1.)
        P.plot(np.sqrt(np.diag(cov)), 'r-', lw=1.8)
        P.yscale('log')
        P.ylim((1e-4, 1e4))
        #P.colorbar()
        #P.plot(np.diag(cov), 'r-', lw=1.8)
        P.show()

