#!/usr/bin/env python
"""
Calculate Fisher matrix for LSST lensing + clustering.
"""
import numpy as np
import pylab as P
import pyccl as ccl
from scipy.integrate import simps
import copy, time
import photoz_selection_fns as selfns
from clustering_z import zbins_lsst_alonso
from fisher_limber import bias_lsst, corr


def trim_zeros(z, fz, thres=1e-40):
    """
    Trim leading and trailing zeros from a function of redshift, fz. This is 
    necessary to work around a bug in the CCL ClTracer() initialisation.
    """
    jmin = np.argmax(fz > thres)
    jmax = -np.argmax(fz[::-1] > thres) # Relies on counting indices backwards
    
    # Handle special case where jmax=0
    if jmax == 0:
        return z[jmin:], fz[jmin:]
    else:
        return z[jmin:jmax], fz[jmin:jmax]


def lsst_dndz(z, kind='nc'):
    """
    Return true number density distribution of sources.
    kind : 'nc', 'wl_cons', 'wl_fid', 'wl_opt'
    """
    fac = 110. # FIXME: Fudge factor to ~reproduce Fig. 2 of 1704.01941!
    fac *= (180.*60./np.pi)**2. # arcmin^-2 -> rad^-2
    
    # Return dN/dz for different types
    if kind == 'nc':
        # Special case for number counts
        z0 = 0.3; x = z / z0
        return fac * (0.5 / z0) * x**2. * np.exp(-x)
    elif kind == 'wl_cons':
        alpha = 1.28; beta = 0.97; z0 = 0.41
    elif kind == 'wl_fid':
        alpha = 1.24; beta = 1.01; z0 = 0.51
    elif kind == 'wl_opt':
        alpha = 1.23; beta = 1.05; z0 = 0.59
    else:
        raise KeyError("Unknown dNdz type '%s'." % kind)
  
    # Return lensing dN/dz
    x = z / z0
    return fac * z**alpha * np.exp(-x**beta)


def fiducial_selection_fns(z, sigma_z0=0.03):
    """
    Calculate fiducial selection functions and number densities.
    """
    # Define photo-z bins
    zmin, zmax = zbins_lsst_alonso(nbins=15, sigma_z0=sigma_z0)
    
    # Calculate selection functions
    sel = [ selfns.calc_selection_fn(z, zmin[i], zmax[i], sigma_z0=sigma_z0) 
            for i in range(zmin.size) ]
    sel = np.array(sel)
    
    # Calculate photo-z selection functions and dN/dz for each tracer
    dNdz_wl = lsst_dndz(z, kind='wl_fid')
    dNdz_nc = lsst_dndz(z, kind='nc')
    sel_wl = np.array( [sel[i] * dNdz_wl for i in range(sel.shape[0])] )
    sel_nc = np.array( [sel[i] * dNdz_nc for i in range(sel.shape[0])] )
    
    return sel_wl, sel_nc


def Csignal(cosmo, ells, z, sel_wl, sel_nc):
    """
    Calculate the lensing + number count signal covariance matrix.
    """
    # Instantiate lensing and number count tracer objects
    # (Needs to trim surrounding zeros to work around a CCL bug)
    tracers = []
    for i in range(sel_wl.shape[0]):
        zz, sel = trim_zeros(z, sel_wl[i])
        tr = ccl.ClTracerLensing(cosmo, False, n=(zz, sel))
        tracers += [tr,]
    
    for i in range(sel_nc.shape[0]):
        zz, sel = trim_zeros(z, sel_nc[i])
        bz = bias_lsst(zz)
        tr = ccl.ClTracerNumberCounts(cosmo, False, False, 
                                      n=(zz, sel), bias=(zz, bz))
        tracers += [tr,]
    
    # Calculate cross-correlations
    Ntracers = len(tracers)
    cs = np.zeros((ells.size, Ntracers, Ntracers))
    for i in range(Ntracers):
        if i % 10 == 0: print("  Row %d / %d" % (i, Ntracers))
        for j in range(i, Ntracers):
            cs[:,i,j] = cs[:,j,i] \
                      = ccl.angular_cl(cosmo, tracers[i], tracers[j], ells)
    return cs


def Cnoise(cosmo, z, sel_wl, sel_nc):
    """
    Calculate the diagonal shot noise term.
    """
    # FIXME: Are there cross-terms between bins?
    Nwl = sel_wl.shape[0]
    Nnc = sel_nc.shape[0]
    
    # Calculate number density of galaxies in each bin by integrating sel. fn.
    cn = np.zeros((Nwl+Nnc, Nwl+Nnc))
    for i in range(Nwl): cn[i,i] = 1. / simps(sel_wl[i], z)
    for i in range(Nnc): cn[Nwl+i,Nwl+i] = 1. / simps(sel_nc[i], z)
    return cn


def derivs(ells, pname, dp, z, sel_wl, sel_nc):
    """
    Calculate derivatives of signal covariance with respect to a cosmological 
    parameter.
    """
    print("  deriv w.r.t '%s'" % pname)
    # Set parameters, using fiducial parameter set as a base
    params_p = copy.deepcopy(params0)
    params_m = copy.deepcopy(params0)
    params_p[pname] += dp
    params_m[pname] -= dp
    
    # Construct Cosmology() objects for each parameter set
    cosmo_p = ccl.Cosmology(**params_p)
    cosmo_m = ccl.Cosmology(**params_m)
    
    # Calculate signal covariance for both parameter sets
    cs_p = Csignal(cosmo_p, ells, z, sel_wl, sel_nc)
    cs_m = Csignal(cosmo_m, ells, z, sel_wl, sel_nc)
    
    # Calculate finite difference derivative and return
    return (cs_p - cs_m) / (2. * dp)


def fisher(ells, params0):
    """
    Calculate Fisher matrix for LSST lensing + number counts.
    """
    # Define redshift range
    z = np.linspace(0., 3.6, 1000) # Redshift sampling for selection fns. etc.
    
    # Define fiducial cosmological parameter set
    cosmo0 = ccl.Cosmology(**params0)

    # Calculate selection functions for lensing and number count tracers
    print(">>> Calculating selection functions")
    sel_wl, sel_nc = fiducial_selection_fns(z, sigma_z0=0.03)

    # Calculate signal covariance
    print(">>> Calculating signal covariance")
    cs = Csignal(cosmo0, ells, z, sel_wl, sel_nc)

    # Calculate shot noise term
    print(">>> Calculating noise covariance")
    cn = Cnoise(cosmo0, z, sel_wl, sel_nc)
    
    # Calculate inverse covariance, C^-1 = (C_S + C_N)^-1
    ctot = cs + cn
    cinv = np.zeros(ctot.shape)
    for l in range(ctot.shape[0]):
        cinv[l] = np.linalg.inv(ctot[l])

    # Calculate inverse covariance-weighted derivatives, C^-1 dC_S/dp
    pnames = ['Omega_c', 'Omega_b', 'h', 'A_s', 'n_s', 'w0', 'wa']
    dps = [0.005, 0.001, 0.005, 0.05e-9, 0.005, 0.01, 0.01]
    Nparams = len(pnames)
    cs_derivs = []
    
    for i in range(Nparams):
        print(">>> Derivative for '%s'" % pnames[i])
        # Calculate derivative of signal covariance w.r.t. parameter
        dCdp = derivs(ells, pnames[i], dps[i], z, sel_wl, sel_nc)
        
        # Multiply with inverse covariance
        deriv = np.zeros(dCdp.shape)
        for l in range(ells.size): deriv[l] = np.dot(cinv[l], dCdp[l])
        cs_derivs.append(deriv)
    
    # Combine together into Fisher matrix
    print(">>> Combining into Fisher matrix")
    F_l = np.zeros((ells.size, Nparams, Nparams))
    for l in range(ells.size):
        print("  ell = %d" % ells[l])
        for i in range(Nparams):
            for j in range(i, Nparams):
                F_l[l,i,j] = np.trace( np.dot(cs_derivs[i][l], cs_derivs[j][l]) )
                F_l[l,j,i] = F_l[l,i,j]
    
    # Save raw by-ell Fisher matrix
    fname = "Fisher_LSST_wl_nc"
    print(">>> Saving Fisher matrix to %s" % fname)
    np.save(fname, F_l)
    
    # Sum Fisher matrix over ell
    _ell = np.swapaxes(np.atleast_3d(ells), 0, 1)
    F = np.sum(F_l * 0.5 * (2.*_ell + 1.), axis=0)
    return F


if __name__ == '__main__':
    
    # Start timing
    t0 = time.time()
    
    # Define ell and redshift ranges
    ells = np.arange(2, 600, 1) # FIXME: Set to a sensible value
    
    # Define fiducial cosmological parameter set
    params0 = {
        'Omega_c':  0.27, 
        'Omega_b':  0.045, 
        'h':        0.67, 
        'A_s':      2.1e-9, 
        'n_s':      0.96,
        'w0':       -1.,
        'wa':       0.,
    }
    
    # Calculate Fisher matrix
    F = fisher(ells, params0)
    
    # Finish timing
    print("Finished in %1.1f min." % ((time.time() - t0)/60.))
    
    P.matshow(corr(F), cmap='RdBu', vmin=-1., vmax=1.)
    P.colorbar()
    P.show()
    
    cov = np.linalg.inv(F)
    print(np.sqrt(np.diag(cov)))
