#!/usr/bin/python
"""
Make realisations of Gaussians that are consistent with the prior covariances.
"""
import numpy as np
import pylab as P
from clustering_z import zbins_lsst_alonso

np.random.seed(10)

# Load covariance matrix
i = 10
cov = np.load("output/zlsst_zphot_dz001_cov_%d.npy" % i)
avg = np.load("output/zlsst_zphot_dz001_avg_%d.npy" % i)
pcts = np.load("output/zlsst_zphot_dz001_pcts_%d.npy" % i)
dz = 0.01 # Redshift bin width

# Covariance matrix redshift bins
zedges = np.arange(0., 3.+0.1*dz, dz)
zc = 0.5 * (zedges[:-1] + zedges[1:])

# Get photo-z bins
zmin, zmax = zbins_lsst_alonso(nbins=15, sigma_z0=0.03)

# Ridge-adjusted covariance matrix
cov_r = cov + 1e-6 * np.eye(cov.shape[0])

# Realise Gaussian
g = np.random.multivariate_normal(avg, cov_r, size=10000).T


pp = np.arange(101)

#P.plot(pp, pcts[:,150], 'k-')
#P.plot(pp, pcts[:,151], 'r-')
#P.plot(pp, pcts[:,152], 'b-')

#P.plot( np.diff(pp)/np.diff(pcts[:,150]) )

cdf = pp
x = pcts[:,150]

px = np.diff(cdf) / np.diff(x)
xc = 0.5 * (x[1:] + x[:-1])

P.plot(xc, px, 'b.' )
P.show()
exit()


# Plot results
P.subplot(111)

for j in [10, 20, 30, 40]:
    P.fill_between(zc, pcts[j], pcts[100-j], color='r', alpha=0.1)

P.plot(zc, pcts[50], 'r-', lw=1.8) # Median

P.axvline(zmin[i], color='k', lw=1.8, ls='dashed', alpha=0.7)
P.axvline(zmax[i], color='k', lw=1.8, ls='dashed', alpha=0.7)
P.xlim((0.7, 2.4))

P.gca().tick_params(axis='both', which='major', labelsize=16, size=8., 
                    width=1.5, pad=5.)
P.gca().tick_params(axis='both', which='minor', labelsize=16, size=5., 
                    width=1.5, pad=5.)

P.xlabel("$z_s$", fontsize=18)
P.ylabel("$\phi_p(z_s)$", fontsize=18)

#P.yscale('log')
P.gcf().set_size_inches((6., 4.5))
P.tight_layout()

#P.savefig("pub_selfn_prior.pdf")
P.show()

