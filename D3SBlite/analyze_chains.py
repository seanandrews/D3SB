import numpy as np
import os
import time
import sys
import matplotlib.pyplot as plt

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


ndim, nwalkers, nthreads = nbbins, 100, 8

chain = np.load('chain.npy')
fchain = chain.reshape(-1, ndim)
trial  = np.arange(np.shape(chain)[1])

# plot chain progress for SB values
fig = plt.figure(1)
for idim in np.arange(ndim):
    for iw in np.arange(nwalkers):
        plt.subplot(5,8,idim+1)
        plt.plot(trial, chain[iw, :, idim], 'b')
        plt.plot(trial, bSB[idim]*np.ones_like(trial), 'r')
fig.savefig('chain_sbs.png')
fig.clf()
