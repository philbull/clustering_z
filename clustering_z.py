#!/usr/bin/env python
"""
Plot cross-correlation between photometric survey and IM survey.
"""
import numpy as np
import pylab as P
import pyccl as ccl
from scipy.integrate import simps
from scipy.special import erf
from scipy.interpolate import interp1d
import matplotlib.ticker
#from multiprocessing import Pool
import time
from mpi4py import MPI

# Set up MPI
comm = MPI.COMM_WORLD
myid = comm.Get_rank()
size = comm.Get_size()

C = 299792.458 # Speed of light, km/s
INF_NOISE = 1e50 #np.inf #1e100
LMAX=2000
perPZ=False
NBINS=15

# Set up cosmology and LSST instrumental specs
cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045, h=0.67, A_s=2.1e-9, n_s=0.96)


inst = {
    # Interferometer parameters
    'name': "hrx",
    'type': "interferometer",
    'd_min':    6., # m
    'd_max':    32*6.*np.sqrt(2), # m
    'Ndish':   32*32,
    'fsky' : 0.4,
    'Tsys' : 50, # in K
    'intgtime' : 2.8e4,
    'fsky_overlap': 0.4
}

inst = {
    # Interferometer parameters
    'name': "GBT",
    'type': "dish",
    'D':    100,
    'Ndish': 7,
    'fsky' : 0.15,
    'Tsys' : 30, # in K
    'intgtime' : 3.2e4,
    'fsky_overlap': 0.15
}



prefix = inst['name']+str(hash(frozenset(inst.items())))



# FIXME
sigma_z0 = 0.03


def calc_sigmaT(inst):
    ## 1000 (mK in  K) /sqrt(1e6 MHz in Hz) = 1
    sigma_T=np.sqrt(inst['Tsys']**2*np.pi*4*inst['fsky']/(inst['intgtime']*3600)/inst['Ndish'])
    inst['sigma_T']=sigma_T
    print "Sigma_T=",sigma_T
    return inst

    
def status(msg):
    """
    Print a status message (handles MPI)
    """
    if myid == 0: print(msg)


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


def dNdz_lsst(z):
    """
    dN/dz for LSST, in galaxies/rad^2.
    """
    # Define parameters for sample
    i_lim = 26. # Limiting i-band magnitude
    z0 = 0.0417*i_lim - 0.744

    Ngal = 46. * 100.31 * (i_lim - 25.) # Normalisation, galaxies/deg^2?
    pz = 1./(2.*z0) * (z / z0)**2. * np.exp(-z/z0) # Redshift distribution, p(z)
    dNdz = Ngal * pz # Number density distribution
    
    # FIXME: Fudge factor to get the right order of mag, c.f. Fig. 2 of Alonso
    dNdz *= 18. # in deg^-2
    return dNdz * (180./np.pi)**2. # in rad^-2


def photoz_pdf(z_ph, z_s, sigma_z0):
    """
    Photometric redshift probability, p(z_ph | z_s).
    """
    sigma_z = sigma_z0 * (1. + z_s)
    return np.exp(- (z_ph - z_s)**2. / (2.*sigma_z**2.)) \
          / (np.sqrt(2.*np.pi) * sigma_z)


def photoz_window(z, z_i, z_f, sigma_z0):
    """
    Integrated photo-z probability for a top-hat bin between z_i and z_f, 
    assuming a Gaussian photo-z pdf. From Eq. 28 of arXiv:1507.03550.
    """
    sigma_z = sigma_z0 * (1. + z)
    w = 0.5 * (  erf((z - z_i) / (np.sqrt(2.)*sigma_z)) \
               - erf((z - z_f) / (np.sqrt(2.)*sigma_z)) )
    return w


def photoz_selection(z, zmin, zmax, dNdz_func, sigma_z0):
    """
    Get selection function in a redshift bin, given some galaxy number density 
    function.
    """
    # Get galaxy number density curve
    dNdz = dNdz_func(z)
    pz = photoz_window(z, zmin, zmax, sigma_z0)
    
    # Get total galaxy number density in bin
    n_tot = simps(dNdz * pz, z)
    
    # Return unnormed selection function and total galaxy number density
    return dNdz * pz, n_tot

def calculate_nonlinear_ell (zc):
    dist=ccl.comoving_angular_distance(cosmo, 1./(1+zc))
    gf=ccl.growth_factor(cosmo,1/(1+zc))
    targetSigma=1.0/gf #SigmaR returns sigma^2 at z=0, but we want it at z=zc
    Rs=np.arange(1,20)
    sigmas2=ccl.sigmaR(cosmo,Rs)
    RNl=interp1d(sigmas2,Rs)(targetSigma)
    kNl=2./RNl  # From RNl to kNl
    ellNl = dist*kNl
    ellNl[ellNl>2000]=2000
    status ("At z="+repr(zc)+" kNl="+repr(kNl)+ " ellNl="+repr(ellNl))
    return ellNl

def calculate_block_noise_int(ells, zmin, zmax):
    """
    Approximate noise expression for a radio interferometer, using Eq. 8 of 
    Alonso et al.
    """
    # Frequency scaling
    zc = 0.5 * (zmin + zmax)
    nu = 1420. / (1. + zc)
    lam = (C * 1e3) / (nu * 1e6) # wavelength in m

    #Calculate ell at which nonlinear stuff matters:
    lmax = calculate_nonlinear_ell (zc)
    
    # Bin widths and angular cutoffs
    dnu = 1420. * (1./(1. + zmin) - 1./(1. + zmax))

    if inst['type']=='interferometer':
    
        # Angular scaling
        _ell, _lam = np.meshgrid(ells, lam)
        f_ell = np.exp(_ell*(_ell+1.) * (1.22 * _lam / inst['d_max'])**2. \
              / (8.*np.log(2.)))

        # Construct noise covariance
        N_ij = f_ell * inst['sigma_T']**2. / dnu[:,None]
        # FIXME: Is this definitely channel bandwidth, rather than total bandwidth?

        # Apply large-scale cut
        N_ij[np.where(_ell*_lam/(2.*np.pi) <= inst['d_min'])] = INF_NOISE

    elif inst['type']=='dish':
        # Angular scaling
        _ell, _lam = np.meshgrid(ells, lam)
        fwhm = 1.22*_lam/inst['D']
        Bell = np.exp(-_ell*(_ell+1) * fwhm**2 / (16*np.log(2)))
        N_ij = inst['sigma_T']**2 / dnu[:,None] / Bell**2 ## Ndish already put in there
    else:
        raise NotImplemented

    for i,clmax in enumerate(lmax):
        N_ij[i,np.where(ells>clmax)] = INF_NOISE
    return N_ij.T


def calculate_block_noise_lsst(ells, nz_lsst):
    """
    Shot noise in each redshift bin, taken by integrating dN/dz over the 
    selection function for the bin.
    """
    # Construct diagonal shot noise covariance for LSST bins
    N_ij = np.zeros((ells.size, len(nz_lsst)))
    for i, nz in enumerate(nz_lsst):
        N_ij[:,i] = np.ones(ells.size) / nz
    return N_ij
    

def selection_lsst(zmin, zmax, sigma_z0, bias_factor=1.):
    """
    Calculate tomographic redshift bin selection function and bias for LSST.
    """
    status("\tselection_lsst")
    # Number counts/selection function in this tomographic redshift bin
    z = np.linspace(0., 3.5, 2000)
    
    nz, ntot = photoz_selection(z, zmin, zmax, dNdz_lsst, sigma_z0)
    tomo_lsst = nz / ntot
    
    # Clustering bias
    bz_lsst = bias_factor * ccl.bias_clustering(cosmo, 1./(1.+z))

    # Number density tracer object
    n_lsst = ccl.ClTracerNumberCounts(cosmo, 
                                      has_rsd=False, has_magnification=False, 
                                      n=(z, nz), bias=(z, bz_lsst))
    return n_lsst, ntot


def selection_im(zmin, zmax, debug=False, bias_factor=1.):
    """
    Calculate tomographic redshift bin selection function and bias for an 
    IM experiment. Each factor of this tracer will have units of mK.
    """
    status("\tselection_im")
    # Number counts/selection function in this tomographic redshift bin
    #z = np.linspace(zmin*0.9, zmax*1.1, 500) # Pad zmin, zmax slightly
    z = np.linspace(zmin*0.8, zmax*1.2, 1000) # Pad zmin, zmax slightly
    tomo_im = np.zeros(z.size)
    tomo_im[np.where(np.logical_and(z >= zmin, z < zmax))] = 1.
    
    # Clustering bias and 21cm monopole temperature, in mK
    bz_im = bias_factor * bias_HI(z) * Tb(z)
    
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
    xi = 100. # Frequency correlation scale
    
    # Pivot scales and frequency scaling
    l_star = 1000.
    nu_star = 130. # MHz
    nu = 1420. / (1. + zc)
    
    # Calculate angle-dep. factor
    f_ell = (ells / l_star)**beta
    
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
    Cij = np.zeros((ells.size, Ni, Nj))
    
    # Construct cross-correlation block
    for i in range(Ni):
        if i % size != myid: continue
        print "  Bin %d / %d (worker %d)" % (i, Ni, myid)
        for j in range(Nj):
            print "    xcorr %d / %d (worker %d)" % (j, Nj, myid)
            Cij[:,i,j] = ccl.angular_cl(cosmo, tracer1[i], tracer2[j], ells)
    
    # Combine all calculated Cij at the root process
    Cij_all = np.zeros(Cij.shape)
    comm.Reduce(Cij, Cij_all, op=MPI.SUM)
    return Cij_all


def calculate_block_diag(ells, tracer1):
    """
    Calculate an assumed-diagonal block of auto-correlations.
    """
    Ni = len(tracer1)
    Cij = np.zeros((ells.size, Ni))
    
    # Construct cross-correlation block
    for i in range(Ni):
        if i % size != myid: continue
        print "  Bin %d / %d (worker %d)" % (i, Ni, myid)
        Cij[:,i] = ccl.angular_cl(cosmo, tracer1[i], tracer1[i], ells)
    
    # Combine all calculated Cij at the root process
    Cij_all = np.zeros(Cij.shape)
    comm.Reduce(Cij, Cij_all, op=MPI.SUM)
    return Cij_all


def expand_diagonal(dmat):
    """
    Expand diagonal elements of matrix into full matrix.
    """
    # Do nothing if the input is not an array
    if isinstance(dmat, float): return dmat
    
    # Unpack into full-sized matrix
    mat = np.zeros((dmat.shape[0], dmat.shape[1], dmat.shape[1]))
    for i in range(dmat.shape[1]):
        mat[:,i,i] = dmat[:,i]
    return mat


def hasharg(args):
    if args==None:
        return ""
    st=""
    for a in args:
        if hasattr(a,"shape"):
            st+=str(a.shape)
        elif type(a) in [int, bool, str, float]:
                st+=str(a)
        elif type(a)==list:
            st+=hasharg(a)
        else:
            st+='*'
    return  "_"+str(hash(st))

def cache_save(blkname, blk, args=None):
    """
    Save a block to a cache file.
    """

    if myid == 0:
        np.save("%s_%s" % (prefix, blkname+hasharg(args)), blk)


def cache_load(blkname, args=None, shape=None):
    """
    Load a cached datafile.
    """
    cache_hit = False
    cache_valid = False
    res = None
    
    # Get expected shape of cached data
    ## why would argument be the same as output?!

    if args is not None:
        shape = None
        ##shape = [len(a) for a in args]
    
    # Try to load cached data
    fname="%s_%s.npy" % (prefix, blkname+hasharg(args))
    status ("  Trying to load  "+fname)
    try:
        res = np.load(fname)
        status("  Loaded from cache.")
        cache_hit = True
        
        if shape is not None:
        # Sanity check on shape of cached data
            assert res.shape[0] == shape[0]
            assert res.shape[1] == shape[1]
            if len(res.shape) > 2: assert res.shape[2] == shape[2]
        cache_valid = True
    except:
        pass
    
    return res, cache_hit, cache_valid


def cache(blkname, func, args, disabled=False):
    """
    Simple caching of computed blocks.
    """
    cache_hit = False
    cache_valid = False
    res = None
    
    # Check if cache exists and load from it if so (only root does I/O)
    # (can be disabled if requested)
    if myid == 0 and not disabled:
        res, cache_hit, cache_valid = cache_load(blkname, args)
    
    # Inform all processes about cache status
    cache_hit = comm.bcast(cache_hit, root=0)
    cache_valid = comm.bcast(cache_valid, root=0)
    if cache_hit and cache_valid:
        return res
    
    # Return None if no function specified (useful for flagging uncached data)
    if func is None: return res
    
    # If no cache exists (or if it's invalid), recompute and save to cache
    if not cache_hit:
        status("  Not cached; recomputing.")
    else:
        if not cache_valid:
            status("  Cache is from different setup. Overwriting.")
    res = func(*args)
    if not disabled:
        cache_save(blkname, res, args)
    return res


def deriv_photoz(ells, tracer1, tracer2, zmin_lsst, zmax_lsst, var, dp=1e-3):
    """
    Calculate derivative of covariance matrix w.r.t. photo-z parameters.
    """
    if var not in ['sigma','delta']:
        print "Vary either sigma_z or delta_z"
        
    N1 = len(tracer1)
    N2 = len(tracer2)
    
    # Build new set of tracers with modified photo-z properties
    tracer1_p = []; tracer1_m = []
    for i in range(zmin_lsst.size):
        if var=='sigma':
            tp, _ = selection_lsst(zmin_lsst[i], zmax_lsst[i], sigma_z0=sigma_z0+dp)
            tm, _ = selection_lsst(zmin_lsst[i], zmax_lsst[i], sigma_z0=sigma_z0-dp)
        elif var=='delta':
            tp, _ = selection_lsst(zmin_lsst[i]+dp, zmax_lsst[i]+dp, sigma_z0=sigma_z0)
            tm, _ = selection_lsst(zmin_lsst[i]-dp, zmax_lsst[i]-dp, sigma_z0=sigma_z0)
        else:
            raise NotImplemented
        tracer1_p.append(tp); tracer1_m.append(tm)
    
    # Calculate photoz derivs
    status("derivs photoz")
    
    # Calculate +/- for each photoz bin; loop over tracers with +/- sigma_z0
    dCij_dvarz0 = []
    shape = (ells.size, N1+N2, N1+N2)
    for i in range(N1):
        
        # Check if this derivative exists in the cache
        cache_hit = False; cache_valid = False
        if myid == 0:
            blkname = "deriv_"+var+"z0_%d" % i
            res, cache_hit, cache_valid = cache_load(blkname, shape=shape)
            
            # Append result to list if cache exists; otherwise, run through the 
            # whole calculation below
            if cache_hit and cache_valid:
                dCij_dvarz0.append(res)
        
        # Inform all processes about cache status
        cache_hit = comm.bcast(cache_hit, root=0)
        cache_valid = comm.bcast(cache_valid, root=0)
        if cache_hit and cache_valid:
            continue
        
        # Cache not found; set up to do the whole calculation for this deriv.
        dC_p = np.zeros((ells.size, N1+N2, N1+N2))
        dC_m = np.zeros((ells.size, N1+N2, N1+N2))
        
        # Get angular Cl for this bin (w. modified sigma_z0) with other pz bins
        status("  deriv photoz-photoz %d / %d" % (i, N1))
        comm.barrier()
        
        for j in range(N1):
            if j % size != myid: continue
            print "    Bin %d / %d (worker %d)" % (j, N1, myid)
            
            # Calculate finite difference deriv.
            trp = tracer1[j] if i != j else tracer1_p[j]
            trm = tracer1[j] if i != j else tracer1_m[j]
            dC_p[:,i,j] = dC_p[:,j,i] \
                = ccl.angular_cl(cosmo, tracer1_p[i], trp, ells)
            dC_m[:,i,j] = dC_m[:,j,i] \
                = ccl.angular_cl(cosmo, tracer1_m[i], trm, ells)
        
        # FIXME: Derivative of pz-pz noise term, along the diagonal?
        # TODO: Should this be in there?
        
        # Get angular Cl for this bin crossed with IM bins
        status("  deriv photoz-im %d / %d" % (i, N1))
        for j in range(N2):
            if j % size != myid: continue
            print "    Bin %d / %d (worker %d)" % (j, N2, myid)
            
            # Calculate finite difference deriv.
            try:
                dC_p[:,i,N1+j] = dC_p[:,N1+j,i] \
                    = ccl.angular_cl(cosmo, tracer1_p[i], tracer2[j], ells)
                dC_m[:,i,N1+j] = dC_m[:,N1+j,i] \
                    = ccl.angular_cl(cosmo, tracer1_m[i], tracer2[j], ells)
            except:
                # If CCL fails, it's probably because the integrator is trying 
                # to integrate two bins with no overlap, and it's just hitting 
                # numerical noise. So, just set the terms to zero for this bin.
                print("WARNING: CCL failed for tracer1[%d], tracer2[%d]" % (i,j))
                dC_p[:,i,N1+j] = dC_p[:,N1+j,i] = 0.
                dC_m[:,i,N1+j] = dC_m[:,N1+j,i] = 0.
            
        # Combine all calculated dC_p,m at the root process
        dC_p_all = np.zeros(dC_p.shape)
        dC_m_all = np.zeros(dC_m.shape)
        comm.Reduce(dC_p, dC_p_all, op=MPI.SUM)
        comm.Reduce(dC_m, dC_m_all, op=MPI.SUM)
        
        # Calculate finite difference derivatives for each block and insert 
        # into full matrix for derivative of covmat
        if myid == 0:
            res = (dC_p_all - dC_m_all) / (2.*dp)
            dCij_dvarz0.append(res)
            cache_save(blkname, res)
        
    return dCij_dvarz0


def deriv_bias(ells, tracer1, tracer2, z_lsst, z_im, dp=0.02):
    """
    Calculate derivative of covariance matrix w.r.t. photo-z parameters.
    """
    N1 = len(tracer1)
    N2 = len(tracer2)
    zmin_lsst, zmax_lsst = z_lsst
    zmin_im, zmax_im = z_im
    
    # Build new set of tracers with modified bias properties
    tracer1_p = []; tracer1_m = []
    for i in range(zmin_lsst.size):
        tp, _ = selection_lsst(zmin_lsst[i], zmax_lsst[i], 
                               sigma_z0=sigma_z0, bias_factor=1.+dp)
        tm, _ = selection_lsst(zmin_lsst[i], zmax_lsst[i], 
                               sigma_z0=sigma_z0, bias_factor=1.-dp)
        tracer1_p.append(tp); tracer1_m.append(tm)
    
    tracer2_p = []; tracer2_m = []
    for i in range(zmin_im.size):
        tp = selection_im(zmin_im[i], zmax_im[i], bias_factor=1.+dp)
        tm = selection_im(zmin_im[i], zmax_im[i], bias_factor=1.-dp)
        tracer2_p.append(tp); tracer2_m.append(tm)
    
    # Define dummy arguments for build_covmat() that have the correct length
    dummy1, dummy2, dummy3 = tracer1, zmin_im, zmax_im
    
    # Calculate tracer1 bias deriv
    status("deriv bias1")
    Cij_bias1p = cache('deriv_bias1p', 
                       build_covmat, 
                       (ells, tracer1_p, tracer2, dummy1, dummy2, dummy3, 
                       ['Sij_im_im', 'Nij_im_im', 'Nij_pz_pz', 'Fij_im_im'],
                       True))
    Cij_bias1m = cache('deriv_bias1m', 
                       build_covmat, 
                       (ells, tracer1_m, tracer2, dummy1, dummy2, dummy3, 
                       ['Sij_im_im', 'Nij_im_im', 'Nij_pz_pz', 'Fij_im_im'],
                       True))
    
    # Calculate tracer2 bias deriv
    status("deriv bias2")
    # nz_lsst, zmin_im, zmax_im
    Cij_bias2p = cache('deriv_bias2p', 
                       build_covmat, 
                       (ells, tracer1, tracer2_p, dummy1, dummy2, dummy3,  
                       ['Sij_im_im', 'Nij_im_im', 'Nij_pz_pz', 'Fij_im_im'],
                       True))
    Cij_bias2m = cache('deriv_bias2m', 
                       build_covmat, 
                       (ells, tracer1, tracer2_m, dummy1, dummy2, dummy3, 
                       ['Sij_pz_pz', 'Nij_im_im', 'Nij_pz_pz', 'Fij_im_im'],
                       True))
    
    # Calculate finite difference derivatives
    deriv1 = None; deriv2 = None
    if myid == 0:
        deriv1 = (Cij_bias1p - Cij_bias1m) / (2.*dp)
        deriv2 = (Cij_bias2p - Cij_bias2m) / (2.*dp)
    return deriv1, deriv2


def invert_covmat(cov):
    """
    Invert a covariance matrix, ell by ell. Assumes covmat has the following 
    shape: (N_ells, N_zbins, N_zbins).
    """
    icov = np.zeros(cov.shape)
    for i in range(cov.shape[0]):
        icov[i,:,:] = np.linalg.inv(cov[i,:,:])
    return icov


def build_covmat(ells, tracer1, tracer2, nz_lsst, zmin_im, zmax_im, 
                 exclude=[], nocache=False):
    """
    Build full covariance matrix from individual blocks.
    N.B. tracer2 should be the IM tracer!
    """
    # Number of tracer redshift bins
    N1 = len(tracer1)
    N2 = len(tracer2)
    
    # Define angular scales and redshift bins
    if zmin_im is not None and zmax_im is not None:
        zc = 0.5 * (zmin_im + zmax_im)
    
    # Set all blocks to zero by default
    Sij_im_im = 0.
    Sij_pz_pz = 0.
    Sij_pz_im = 0.
    Nij_im_im = 0.
    Nij_pz_pz = 0.
    Fij_im_im = 0.
    
    # FIXME: This has to ignore the cached files when being called from the deriv function
    
    # Calculate IM auto block
    if 'Sij_im_im' not in exclude:
        status("signal im-im")
        Sij_im_im = cache('Sij_im_im', 
                          calculate_block_diag, 
                          (ells, tracer2), disabled=nocache)

    # Calculate LSST auto block
    if 'Sij_pz_pz' not in exclude:
        status("signal photoz-photoz")
        Sij_pz_pz = cache('Sij_pz_pz', 
                          calculate_block_gen, 
                          (ells, tracer1, tracer1), disabled=nocache)
    
    # Calculate cross-tracer signal block
    if 'Sij_pz_im' not in exclude:
        status("signal photoz-im")
        Sij_pz_im = cache('Sij_pz_im', 
                          calculate_block_gen, 
                          (ells, tracer1, tracer2), disabled=nocache)
    
    # Calculate IM noise auto block
    if 'Nij_im_im' not in exclude:
        status("noise im-im")
        Nij_im_im = cache('Nij_im_im', 
                          calculate_block_noise_int, 
                          (ells, zmin_im, zmax_im), disabled=nocache)
    
    # Calculate LSST noise auto block
    if 'Nij_pz_pz' not in exclude:
        status("noise photoz-photoz")
        Nij_pz_pz = cache('Nij_pz_pz', 
                          calculate_block_noise_lsst, 
                          (ells, nz_lsst), disabled=nocache)
    
    # Calculate IM foreground residual auto block
    #if 'Fij_im_im' not in exclude:
        #status("foreground im-im")
        #Fij_im_im = calculate_block_fg(ells, zc) # FIXME
    
    # Piece together blocks on root worker only
    if myid != 0: return None
    
    # Construct total covariance matrix
    status("Combining covariance blocks...")
    Cij = np.zeros((ells.size, N1 + N2, N1 + N2))
    Cij[:,:N1,:N1] = Sij_pz_pz + expand_diagonal(Nij_pz_pz)
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


def fisher(ells, z_lsst, z_im):
    """
    Calculate Fisher matrix.
    """
    # Get redshift bin arrays
    zmin_lsst, zmax_lsst = z_lsst
    zmin_im, zmax_im = z_im
    
    # Initialise tracers
    status("Initialising tracers...")
    lsst_sel = [ selection_lsst(zmin_lsst[i], zmax_lsst[i], sigma_z0=sigma_z0) 
                 for i in range(zmin_lsst.size) ]
    tracer1, nz_lsst = zip(*lsst_sel)
    tracer2 = [ selection_im(zmin_im[i], zmax_im[i]) 
                for i in range(zmin_im.size) ]
    
    # Build covariance matrix and invert
    cov = build_covmat(ells, tracer1, tracer2, nz_lsst, zmin_im, zmax_im)
    status("All processes finished covmat.")
    
    # Get Fisher derivatives
    derivs_pz_sigma = deriv_photoz(ells, tracer1, tracer2, 
                                   zmin_lsst, zmax_lsst, 'sigma',
                                   dp=1e-3)
    derivs_pz_delta = deriv_photoz(ells, tracer1, tracer2, 
                                   zmin_lsst, zmax_lsst, 'delta',
                                   dp=1e-3)

    derivs_bias = deriv_bias(ells, tracer1, tracer2, 
                             (zmin_lsst, zmax_lsst), 
                             (zmin_im, zmax_im), 
                             dp=0.02)
    

    if myid!=0:
        comm.barrier()
        return None
    
    if perPZ:
        zcl = (zmin_lsst+zmax_lsst)/2
            # Calculate Fisher matrix
        
        Nzlsst=NBINS
        NzIm=len(zmin_im)            
        Nparam=Nzlsst*4 # sigma,bias, bl,bi
        Fij_ell=np.zeros((ells.size, Nparam, Nparam))
        for bi,zc in enumerate(zcl):
            status ("Doing perPZ %i/%i"%(bi,Nzlsst))
            ## Let's see which bins do we want from IM
            sigma_z = sigma_z0 * (1. + zc)
            indx=np.where((zmax_im>zc-3*sigma_z) & (zmin_im<zc+3*sigma_z))[0]
            #print zmin_im
            #Eprint zmax_im
            print zc-3*sigma_z, zc+3*sigma_z
            print indx,'indx'
            curcov=np.array(cov)
            
            for k in range(Nzlsst):
                if k==bi:
                    continue
                curcov[:,k,:]=0
                curcov[:,:,k]=0
                curcov[:,k,k]=INF_NOISE
            for k in range(NzIm):
                if k in indx:
                    continue
                curcov[:,Nzlsst+k,:]=0
                curcov[:,:,Nzlsst+k,]=0
                curcov[:,Nzlsst+k,Nzlsst+k]=INF_NOISE
                           
                Cinv = invert_covmat(curcov)

            # Accounting for number of parameters
            derivs_all=[derivs_pz_sigma[bi],derivs_pz_delta[bi]]+list(derivs_bias)

            for l in range(len(ells)):
              for i in range(4):
                  y_i = np.dot(Cinv[l], derivs_all[i][l])
                  for j in range(4):
                      y_j = np.dot(Cinv[l], derivs_all[j][l])
                      Fij_ell[l,i*Nzlsst+bi,j*Nzlsst+bi] = inst['fsky_overlap']*(ells[l] + 0.5) * np.trace(np.dot(y_i, y_j))
                  
    
    else:
        Cinv = invert_covmat(cov)

        # Accounting for number of parameters
        derivs_all=list(derivs_pz_sigma)+list(derivs_pz_delta)+list(derivs_bias)
        Nparam = len(derivs_all)

        # Calculate Fisher matrix
        Fij_ell = np.zeros((ells.size, Nparam, Nparam))
        for l in range(len(ells)):
          for i in range(Nparam):
            y_i = np.dot(Cinv[l], derivs_all[i][l])
            for j in range(Nparam):
              y_j = np.dot(Cinv[l], derivs_all[j][l])
              Fij_ell[l,i,j] = inst['fsky_overlap']*(ells[l] + 0.5) * np.trace(np.dot(y_i, y_j))
    
    comm.barrier()
    return Fij_ell
    

def zbins_lsst_alonso(nbins=15):
    """
    Get redshift bins edges as defined in Alonso et al.
    """
    Deltaz = lambda zmin: 3.*sigma_z0 * (1. + zmin) / (1 - 0.5*3.*sigma_z0)
    zmin = [0.,]; zmax = []
    for i in range(nbins-1):
        dz = Deltaz(zmin[i])
        zmin.append(zmin[i] + dz)
        zmax.append(zmin[i] + dz)
    zmax.append(zmin[-1] + dz)
    return np.array(zmin), np.array(zmax)


def zbins_im_growing(zmin, zmax, dz0=0.05):
    """
    Get redshift bins edges for an IM experiment that aren't too narrow at high 
    redshift (causes integration errors in CCL otherwise).
    """
    z_edges = [zmin,]
    z = zmin
    while z < zmax:
        dz = dz0 * (1. + z) / (1. + zmin)
        z += dz
        z_edges.append(z)
    zmin = np.array(z_edges)[:-1]
    zmax = np.array(z_edges)[1:]
    return zmin, zmax



if __name__ == '__main__':
    ## caclulate sigma_T
    inst=calc_sigmaT(inst)
    # Define angular scales and redshift bins
    ells = np.arange(5, LMAX+1)
    zmin_lsst, zmax_lsst = zbins_lsst_alonso(nbins=NBINS)
    zmin_im, zmax_im = zbins_im_growing(0.2, 2.5, dz0=0.04)

    # Build Fisher matrix
    status("Calculating Fisher matrix...")
    t0 = time.time()
    Fij_ell = fisher(ells, (zmin_lsst, zmax_lsst), (zmin_im, zmax_im))
    status("Run finished in %1.1f min." % ((time.time() - t0)/60.))
    
    if myid == 0:
        # Sum over ell modes; save unsummed matrix to file
        Fij = np.sum(Fij_ell, axis=0)
        np.save("%s_Fij" % prefix, Fij_ell)

        C=np.linalg.inv(Fij)
        P.matshow(C)
        P.colorbar()
        P.show()

        errs=np.sqrt(C.diagonal())
        P.plot((zmin_lsst+zmax_lsst)/2.0,errs[:NBINS],label='sigmaz')
        P.plot((zmin_lsst+zmax_lsst)/2.0,errs[NBINS:2*NBINS],label='deltaz')
        P.legend()
        P.semilogy()
        P.show()
        
    comm.barrier()
    exit()
    # Plotting
    P.subplot(111)
    P.plot(ell, Cij[:,0,7], lw=1.8, 
           label="z_lsst = %2.2f, z_im = %2.2f" % (zmin_lsst[0], zmin_im[0]))
    P.plot(ell, Cij[:,0,8], lw=1.8, 
           label="z_lsst = %2.2f, z_im = %2.2f" % (zmin_lsst[0], zmin_im[1]))
    P.plot(ell, Cij[:,0,9], lw=1.8, 
           label="z_lsst = %2.2f, z_im = %2.2f" % (zmin_lsst[0], zmin_im[2]))
    P.plot(ell, Cij[:,0,10], lw=1.8, 
           label="z_lsst = %2.2f, z_im = %2.2f" % (zmin_lsst[0], zmin_im[3]))
    P.plot(ell, Cij[:,0,11], lw=1.8, 
           label="z_lsst = %2.2f, z_im = %2.2f" % (zmin_lsst[0], zmin_im[4]))

    P.plot(ell, Cij[:,7,7], 'k-', lw=1.8)

    P.yscale('log')
    #P.ylim(())

    P.legend(loc='lower right', frameon=False)
    P.tight_layout()
    P.show()
    exit()

    # Plot correlation matrix for a given ell
    i_ell = 15
    print corrmat(Cij[i_ell])[0,:]
    print corrmat(Cij[i_ell])[-1,:]
    P.matshow(corrmat(Cij[:,0,7]), cmap='RdBu', vmin=-1., vmax=1.)
    P.axhline(len(tracer1)-0.5, color='k', ls='dashed', lw=1.8)
    P.axvline(len(tracer1)-0.5, color='k', ls='dashed', lw=1.8)
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
