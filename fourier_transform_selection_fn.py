#!/usr/bin/env python
"""
Calculate Fourier transform of a given photo-z function.
"""
import numpy as np
import pyccl as ccl
import pylab as P
from scipy.integrate import trapz

def sel_fn(z, zc, dz, sigz):
    """
    Example selection function.
    """
    return np.exp(-0.5 * (z - zc - dz)**2. / sigz**2.) / np.sqrt(2.*np.pi) / sigz
    

# Define cosmology
cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045, h=0.67, A_s=2.1e-9, n_s=0.96)

# Calculate comoving radial distance for a given redshift sampling
chi = np.linspace(0., 12000., 2**17)
print chi.size
z = 1./ccl.scale_factor_of_chi(cosmo, chi) - 1.
kpar = np.fft.fftfreq(z.size, d=chi[1]-chi[0]) * 2.*np.pi

# FT of selection function
phi = sel_fn(z, zc=1.0, dz=0.0, sigz=0.003*2.)
phi2 = sel_fn(z, zc=1.0, dz=0.3, sigz=0.03*2.)

phi_k = np.fft.fftshift( np.abs( np.fft.fft(phi) ) )
phi2_k = np.fft.fftshift( np.abs( np.fft.fft(phi2) ) )
kpar = np.fft.fftshift(kpar)


# Do FT manually
_kpar = np.logspace(-3., 1., 200)
y = np.abs( [trapz(phi * np.exp(-1.j * _k * chi), chi) for _k in _kpar] )


P.subplot(111)
#P.plot(kpar, 0.9*phi_k + 0.1*phi2_k, 'k-', lw=1.8)
P.plot(kpar, phi_k, 'b-', lw=1.8)
P.plot(_kpar, y, 'r-', lw=1.8)
#P.plot(kpar, 0.1*phi2_k, 'r-', lw=1.8)
P.yscale('log')
P.xscale('log')
P.show()
