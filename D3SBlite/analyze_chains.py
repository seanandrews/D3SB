import numpy as np
import os
import time
import sys
import matplotlib.pyplot as plt
from discrete1d_model import discrete1d_model
sys.path.append('/home/sandrews/mypy/')
from deproject_vis import deproject_vis


# define the "true" SB distribution
nbins = 2000
b = 0.001 + 0.001*np.arange(nbins)
a = np.roll(b, 1)
a[0] = 0.1/140.
r = 0.5*(a+b)

flux = 0.12
sig = 0.6
SB = (r/sig)**-0.7 * np.exp(-(r/sig)**2.5)
int_SB = np.sum(np.pi*SB*(b**2-a**2))
SB *= flux/int_SB


# define a "binned" version of the SB distribution
nbbins = 24
b1 = 0.015 + 0.037*np.arange(10)
b2 = np.logspace(np.log10(b1[-1]), np.log10(1.1), num=15)
bb = np.concatenate([b1[:-1], b2])
ba = np.roll(bb, 1)
ba[0] = 0.1/140.
br = 0.5*(ba+bb)
bSB = (br/sig)**-0.7 * np.exp(-(br/sig)**2.5)
bSB *= flux/int_SB
stepSB = np.zeros_like(r)
for i in np.arange(nbbins): stepSB[(r>ba[i]) & (r<=bb[i])] = bSB[i]
bins = 0.1/140., bb



iSB = np.zeros_like(bb)
for i in range(len(bb)):
    inbin  = np.where((r>ba[i]) & (r<=bb[i]))
    fluxin = np.sum(np.pi*SB[inbin]*(b[inbin]**2-a[inbin]**2))
    omega  = np.pi*(bb[i]**2-ba[i]**2)
    iSB[i] = fluxin/omega
iSB *= flux/np.sum(np.pi*iSB*(bb**2-ba**2))




ndim, nwalkers, nthreads = nbbins, 100, 8

chain = np.load('chain.npy')[:,3000:,:]
trial  = np.arange(np.shape(chain)[1])

# plot chain progress for SB values
fig = plt.figure(1)
for idim in np.arange(ndim):
    for iw in np.arange(nwalkers):
        plt.subplot(6,4,idim+1)
        plt.plot(1e-3*trial, chain[iw, :, idim], 'b')
        plt.plot(1e-3*trial, iSB[idim]*np.ones_like(trial), 'r')
fig.savefig('chain_sbs.png')
fig.clf()



# derive best-fit and percentiles
fchain = chain.reshape(-1, ndim)
post = np.percentile(fchain, [16, 50, 84], axis=0)

# plot SB profile + posteriors
plt.axis([0.01, 3, 1e-4, 2])
plt.loglog(r, SB, 'k') #, r, stepSB, 'r')
plt.errorbar(br, post[1][:],xerr=[br-ba, bb-br],
             yerr=[post[1][:]-post[0][:], post[2][:]-post[1][:]], 
             fmt='.r')
plt.savefig('SBpost.png')
plt.clf()



# load the "true" visibilities and convert to a binned 1-D profile
data = np.load('../DATA/fullA.vis.npz')
nt = deproject_vis([data['u'], data['v'], data['Vis'], data['Wgt']],
                   bins=np.linspace(10., 3000., num=150.), incl=50., PA=70.,
                   offx=-0.3, offy=-0.2)
nrho, nvis, nsig = nt



# calculate the "binned" and "guess" visibilities 
bvis = discrete1d_model(bSB, nrho, bins)
pvis = discrete1d_model(post[1][:], nrho, bins)

print(post[1][:])


# plot the visibility profiles together
plt.axis([0, 2700, -0.015, 0.13])
plt.plot([0, 2700], [0, 0], '--k', alpha=0.5)
plt.errorbar(1e-3*nrho, nvis.real, yerr=nsig.real, ecolor='k', fmt='.k', \
             alpha=0.1)
plt.plot(1e-3*nrho, bvis.real, 'b', 1e-3*nrho, pvis.real, 'r')
plt.show()
plt.clf()

