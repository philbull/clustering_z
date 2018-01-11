#!/usr/bin/env python
"""
Run forecasts for optical-IM clustering redshift photo-z calibration.
"""
import numpy as np
import pylab as P
import clustering_z as clz
import pyccl as ccl
import time

# Set up cosmology and LSST instrumental specs
cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045, h=0.67, A_s=2.1e-9, n_s=0.96)

# Assumed sigma_z0 for LSST
sigma_z0 = 0.03
KMAX0 = 0.2 # Mpc^-1

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

# GBT single-dish, 7-beam receiver
inst_gbt = {
    'name':     "GBT",
    'type':     "dish",
    'D':        100.,
    'Ndish':    7,
    'fsky':     0.15,
    'Tsys':     30., # in K
    'ttot':     3.2e4, # hrs
    'fsky_overlap': 0.15
}

# Setup experimental settings
expt = clz.setup_expt_info(cosmo, inst_hirax, kmax=0.2, ignore_photoz_corr=False, sigma_z0=sigma_z0)

# Define angular scales and redshift bins
ells = np.arange(5, 501)
zmin_lsst, zmax_lsst = clz.zbins_lsst_alonso(nbins=15, sigma_z0=sigma_z0)
zmin_im, zmax_im = clz.zbins_im_growing(0.2, 2.5, dz0=0.04)

# Build Fisher matrix
clz.status("Calculating Fisher matrix...")
t0 = time.time()
Fij_ell, dbg = clz.fisher(expt, ells, (zmin_lsst, zmax_lsst), (zmin_im, zmax_im), debug=True)
clz.status("Run finished in %1.1f min." % ((time.time() - t0)/60.))

if clz.myid == 0:
    # Get debug info
    cov = dbg['cov']
    derivs_pz_sigma = dbg['derivs_pz_sigma']
    derivs_pz_delta = dbg['derivs_pz_delta']

    print cov.shape


clz.comm.barrier()
exit()

if clz.myid == 0:
    # Sum over ell modes; save unsummed matrix to file
    Fij = np.sum(Fij_ell, axis=0)
    np.save("%s_Fisher_ij" % expt['prefix'], Fij_ell)

    C = np.linalg.inv(Fij)
    P.matshow(C)
    P.colorbar()
    P.show()

    nbins=zmin_lsst.size
    errs=np.sqrt(C.diagonal())
    P.plot((zmin_lsst+zmax_lsst)/2.0,errs[:nbins],label='sigmaz')
    P.plot((zmin_lsst+zmax_lsst)/2.0,errs[nbins:2*nbins],label='deltaz')
    P.legend()
    P.semilogy()
    P.show()
    
clz.comm.barrier()


