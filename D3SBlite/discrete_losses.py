import sys
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from discrete1d_model import discrete1d_model
from lnprob import lnprob
import emcee
import os
import time
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



# load the "true" visibilities and convert to a binned 1-D profile
data = np.load('../testbed/testA.vis.npz')
tt = deproject_vis([data['u'], data['v'], data['nf_Vis'], data['Wgt']],  
                   bins= np.linspace(10., 3000., num=150.))
trho, tvis, tsig = tt
# and with the noise
nt = deproject_vis([data['u'], data['v'], data['Vis'], data['Wgt']], 
                   bins= np.linspace(10., 3000., num=150.))
nrho, nvis, nsig = nt



# plot SB profile + posteriors
plt.axis([0.001, 3, 1e-4, 2])
plt.loglog(r, SB, 'k', r, stepSB, 'r')
plt.savefig('losses_SB.pdf')
plt.clf()


# calculate the "binned" and "guess" visibilities 
bvis = discrete1d_model(bSB, trho, bins)

# plot the visibility profiles together
plt.axis([0, 2700, -0.015, 0.13])
plt.plot([0, 2700], [0, 0], '--k', alpha=0.5)
plt.errorbar(1e-3*nrho, nvis.real, yerr=nsig.real, ecolor='k', fmt='.k', \
             alpha=0.1)
plt.plot(1e-3*trho, tvis.real, 'k', 1e-3*trho, bvis, 'r')
plt.savefig('losses_VIS.pdf')
plt.clf()


