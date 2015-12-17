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
sig = 0.6
SB = (r/sig)**-0.7 * np.exp(-(r/sig)**2.5)
int_SB = np.trapz(2.*np.pi*SB*r, r)
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



# load the "true" visibilities and convert to a binned 1-D profile
data = np.load('../DATA/fullA.vis.npz')
dprj_vis = deproject_vis([data['u'], data['v'], data['Vis'], data['Wgt']], 
                   bins=np.linspace(10., 3000., num=150.), incl=50., PA=70.,
                   offx=-0.3, offy=-0.2)
drho, dvis, dsig = dprj_vis



# - SAMPLE POSTERIOR

# load initial guesses
p0 = (np.load('p0.npz'))['p0']

# create a file to store progress information
os.system('rm notes.dat')
f = open("notes.dat", "w")
f.close()

# initialize sampler
ndim, nwalkers, nthreads = nbbins, 100, 8
data = drho, dvis, dsig
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, threads=nthreads, \
                                args=[data, bins])

# emcee sampler; track time
iter = 100
tic0 = time.time()
sampler.run_mcmc(p0, iter)
toc = time.time()
print(toc-tic0)

# save the results in a binary file
np.save('chain', sampler.chain)

# add a note
f = open("notes.dat", "a")
f.write("{0:f}   {1:f}   {2:f}\n".format((toc-tic0)/3600., (toc-tic0)/3600., \
                                         iter))
f.close()


for i in range(199):
    tic = time.time()
    sampler.run_mcmc(sampler.chain[:, -1, :], iter)
    toc = time.time()
    np.save('chain', sampler.chain)
    f = open("notes.dat", "a")
    f.write("{0:f}   {1:f}   {2:f}\n".format((toc-tic0)/3600., \
                                             (toc-tic)/3600., \
                                             (2+i)*iter))
    f.close()

