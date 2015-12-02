import sys
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from discrete1d_model import discrete1d_model
from lnprob import lnprob
import emcee
import os
import time
from astropy.io import fits
sys.path.append('/home/sandrews/mypy/')
from deproject_vis import deproject_vis


# define the "true" SB distribution
nbins = 2000
b = 0.001 + 0.001*np.arange(nbins)
a = np.roll(b, 1)
a[0] = 0.1/140.
r = 0.5*(a+b)

flux = 0.12
sig = 0.5
SB = (r/sig)**-0.75 * np.exp(-0.5*(r/sig)**2.5)
int_SB = np.trapz(2.*np.pi*SB*r, r)
SB *= flux/int_SB


# define a "binned" version of the SB distribution
nbbins = 40
bb = np.linspace(0.03, 1.1, num=nbbins)
ba = np.roll(bb, 1)
ba[0] = 0.1/140.
br = 0.5*(ba+bb)
bSB = (br/sig)**-0.75 * np.exp(-0.5*(br/sig)**2.5)
bSB *= flux/int_SB
stepSB = np.zeros_like(r)
for i in np.arange(nbbins): stepSB[(r>ba[i]) & (r<=bb[i])] = bSB[i]
bins = 0.1/140., bb


# define a hybrid "binned" version of the SB distribution
nbbins = 24
b1 = 0.02 + 0.035*np.arange(10)
b2 = np.logspace(np.log10(b1[-1]), np.log10(1.1), num=15)
bb = np.concatenate([b1[:-1], b2])
ba = np.roll(bb, 1)
ba[0] = 0.1/140.
br = 0.5*(ba+bb)
bSB = (br/sig)**-0.75 * np.exp(-0.5*(br/sig)**2.5)
bSB *= flux/int_SB
stepSB = np.zeros_like(r)
for i in np.arange(nbbins): stepSB[(r>ba[i]) & (r<=bb[i])] = bSB[i]
bins = 0.1/140., bb



# use the synthesized image to get an initial "guess" on SB distribution
hdu = fits.open('../DATA/testA.image.fits')
dimage = np.squeeze(hdu[0].data)
h = hdu[0].header
RA  = h['CDELT1']*(np.arange(h['NAXIS1'])-(h['CRPIX1']-1))
DEC = h['CDELT2']*(np.arange(h['NAXIS2'])-(h['CRPIX2']-1))
RAo, DECo = np.meshgrid(RA, DEC)
imrad = 3600.*np.sqrt(RAo**2 + DECo**2)
gSB = np.zeros_like(br)
for i in range(len(br)):
    gSB[i] = np.mean(dimage[(imrad > ba[i]) & (imrad <= bb[i])])
omega_beam = np.pi*(3600.**2)*h['BMAJ']*h['BMIN']/(4.*np.log(2.))
gSB /= omega_beam
gstepSB = np.zeros_like(r)
for i in np.arange(nbbins): gstepSB[(r>ba[i]) & (r<=bb[i])] = gSB[i]


# use the "guess" to generate an initial ball of guesses (enforce monotonicity,
# and account for convolution smearing at unresolved scales)
ndim, nwalkers, nthreads = nbbins, 100, 8
p0 = np.zeros((nwalkers, ndim))
for i in range(nwalkers):
    ixs = br < 0.5*0.08
    sbtrial_o = gSB[~ixs] * (1.+np.random.uniform(-0.2, 0.2, np.sum(~ixs)))
    sbtrial_i = gSB[ixs] * (1.+np.random.uniform(0, 2, np.sum(ixs)))
    if (np.sum(ixs) > 1):
        m = np.log(sbtrial_i[0]/sbtrial_o[0])/np.log((br[ixs])[0]/(br[~ixs])[0])
        sbtrial_i[0:] = (sbtrial_i[0]/(br[ixs])[0]**m) * (br[ixs])[0:]**m 
    p0[i][:] = np.minimum.accumulate(np.concatenate([sbtrial_i, sbtrial_o]))


# plot the SB distributions together
plt.axis([0.01, 3, 1e-4, 2])
plt.loglog(imrad, dimage/omega_beam, '.y', alpha=0.01)
plt.loglog(r, SB, 'k', r, stepSB, 'r', r, gstepSB, 'g')
for i in range(nwalkers):
    plt.loglog(br, p0[i][:], 'b', alpha=0.05)
#plt.savefig('SB.png')
plt.show()
plt.clf()


# save the guesses
np.savez('p0', p0=p0)
