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
nbbins = 50
bb = np.linspace(0.03, 1.5, num=nbbins)
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
Ruv = np.linspace(10., 3000., num=1000.)
tt = deproject_vis([data['u'], data['v'], data['nf_Vis'], data['Wgt']],  
                   bins= np.linspace(10., 3000., num=1000.))
trho, tvis, tsig = tt
# and with the noise
nt = deproject_vis([data['u'], data['v'], data['Vis'], data['Wgt']], 
                   bins= np.linspace(10., 3000., num=1000.))
nrho, nvis, nsig = nt



# calculate the "binned" and "guess" visibilities 
bvis = discrete1d_model(bSB, trho, bins)
gvis = discrete1d_model(gSB, trho, bins)



# plot the visibility profiles together
plt.axis([0, 2700, -0.015, 0.13])
plt.plot([0, 2700], [0, 0], '--k', alpha=0.5)
plt.errorbar(1e-3*nrho, nvis.real, yerr=nsig.real, ecolor='k', fmt='.k', \
             alpha=0.1)
plt.plot(1e-3*trho, tvis.real, 'k', 1e-3*trho, bvis, 'r', 1e-3*trho, gvis, 'g')
plt.savefig('VIS.pdf')
plt.clf()




# - SAMPLE POSTERIOR

# create a file to store progress information
os.system('rm notes.dat')
f = open("notes.dat", "w")
f.close()

# initialize sampler
data = nrho, nvis.real, nsig.real
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


for i in range(29):
    tic = time.time()
    sampler.run_mcmc(sampler.chain[:, -1, :], iter)
    toc = time.time()
    np.save('chain', sampler.chain)
    f = open("notes.dat", "a")
    f.write("{0:f}   {1:f}   {2:f}\n".format((toc-tic0)/3600., \
                                             (toc-tic)/3600., \
                                             (2+i)*iter))
    f.close()

