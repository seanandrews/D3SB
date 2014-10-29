import numpy as np
from get_vis import get_vis
from lnprob import lnprob
from astropy.io import ascii
from astropy.table import Table
import emcee
import time

# get data
data = get_vis('model.vis.fits')

# choose radial bins
nbins = 20
ba = np.linspace(0.01, 0.09, num=3)
bb = np.logspace(-1, 1, num=nbins-3)
b = np.concatenate([ba, bb])
rin = 7.142857e-4
bins = rin, b

# initial guess for weights
sig = 0.15
nrm = 0.1/(2.*np.pi*sig**2)
a = np.roll(b, 1)
a[0] = rin
cb = 0.5*(a+b)
w0 = nrm*np.exp(-0.5*(cb/sig)**2)
sig = 0.8
nrm = 0.05/(2.*np.pi*sig**2)
w1 = nrm*np.exp(-0.5*(cb/sig)**2)
w0 = w0+w1
w0 = np.zeros_like(cb)+0.04

# set up walkers
ndim, nwalkers, nthreads = nbins, 200, 14
p0 = np.array(np.log10(w0))
p0 = [p0 + np.random.randn(ndim)*1.5 for i in range(nwalkers)]

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, threads=nthreads, args=[data, bins])

tic = time.time()
sampler.run_mcmc(p0, 500)
toc = time.time()
print(toc-tic)

trace = sampler.chain.reshape(-1, ndim)
output = Table(trace)
ascii.write(output, 'trace_thu.txt')
