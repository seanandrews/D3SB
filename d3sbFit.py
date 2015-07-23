import numpy as np
from getVis import getVis
from lnprob import lnprob
from astropy.io import ascii
from astropy.table import Table
from scipy.optimize import minimize
from opt_func import opt_func
import emcee
import time
import sys

# get data
data = getVis('DATA/model.vis.fits')

# choose radial bins
nbins = 15
b = np.linspace(0.1, 3., num=nbins)
a = np.roll(b, 1)
rin = 0.01/140.
a[0] = rin
cb = 0.5*(a+b)
bins = rin, b

# initial guess for weights (based on Gaussian fit to fake visibilities)
sig = 0.3
nrm = 0.1/(2.*np.pi*sig**2)
wg = nrm*np.exp(-0.5*(cb/sig)**2)
wg[wg <= 1e-4] = 1e-4

# optimization
#opt = minimize(opt_func, np.log10(wg), args=(data, bins))
#w0 = 10.**(opt.x)
#print(w0)
#np.save('opt_guess',w0)
#sys.exit()
w0 = np.load('opt_guess.npy')

# set up walkers
ndim, nwalkers, nthreads = nbins, 80, 8
theta0 = np.log10(w0)
p0 = [theta0 + np.random.uniform(-1, 1, ndim) for i in range(nwalkers)]

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, threads=nthreads, args=[data, bins])

# run emcee
tic = time.time()
sampler.run_mcmc(p0, 100)
toc = time.time()
print((toc-tic)/60.)

# save the results in a binary file
np.save('saved_results',sampler.chain)
