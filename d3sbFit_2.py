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
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pdb
from synthimage import synthguess

basename = 'mod_gamma1'
hiresvis = basename + '.vis.fits'
synthimg = basename + '_1mm.fits'

#Still need to automate this part, based on createmodel !!
#True values
dpc = 140. #***
rout = 100./dpc #***

gam = 1.0 #***
Ftot = 0.2 #***

# get data
data = getVis('DATA/'+hiresvis)

#corrupt/change data
u, v, dreal, dimag, dwgt = data
mu = 0
sigma = 1./np.sqrt(dwgt)
# dwgt = dwgt * 100. 
##replace existing data with changed data
# data = u,v,dreal, dimag, dwgt


#Model surface brightness
rvals = np.logspace(-2.1, .7, num=200) #Radial range chosen here. Denser than the actual input range.
wtrueall = ((2-gam)*Ftot/(2*np.pi*rout**2)) * (rvals/rout)**(-gam)*np.exp(-(rvals/rout)**(2.-gam)) #SB model

    
# choose radial bins
binsizes = np.arange(10,11,5)
for nbins in binsizes:
    b = np.linspace(0.1, 3., num=nbins)
    a = np.roll(b, 1)
    rin = 0.01/dpc #***
    a[0] = rin
    cb = 0.5*(a+b)
    dbins = rin, b

    #model values at bin locations
    wtrue = ((2-gam)*Ftot/(2*np.pi*rout**2)) * (cb/rout)**(-gam)*np.exp(-(cb/rout)**(2.-gam))

    #mean values at bin locations
    wg = synthguess(a, b, nbins, synthimg)

    
    ## # optimization
    ## print "Entering minimization"
    ## opt = minimize(opt_func, np.log10(wg), args=(data, dbins))
    ## w0 = 10.**(opt.x)
    ## print(w0)
    ## print opt.message  
    ## opt2 = minimize(opt_func, np.log10(wg), args=(data, dbins), method='Nelder-Mead', options={'maxfev': 10000})
    ## w02 = 10.**(opt2.x)
    ## print(w02)
    ## print opt2
    ## print "Left minimization"
    ## filename = 'opt_'+basename+'_'+str(nbins)
    ## np.savez(filename, cb=cb, wg=wg, w02=w02, w0=w0, wtrue=wtrue)
    #pdb.set_trace()


#Continue from pre-optimized values (for loop over different bin ssizes removed) !!
infile = np.load('opt_'+basename+'_'+str(nbins)+'.npz')
w0 = infile['w02'] #If you want to run with w02, change the infile reference ***

# set up walkers
ndim, nwalkers, nthreads = nbins, 4*nbins, 6 #***
theta0 = np.log10(w0)
print theta0

p0 = np.zeros((nwalkers, nbins))
#[np.zeros_like(w0) for j in range(nwalkers)]

for walker in range(nwalkers):
    radii = range(nbins) #HARDCODED FOR NOW !!
    for rs in radii:
        if rs > 6:
            sizecorr = 1.
        else:
            sizecorr = 1.
        rand = np.random.uniform(-np.log10(.2*w0[rs]), np.log10(.2*w0[rs]))
        rand2 = np.random.uniform(-np.log10(.2*w0[rs]*sizecorr), np.log10(.2*w0[rs]*sizecorr))
        if rs > 6:
            print 'Rand', rand, 'Rand2', rand2
        p0[walker][rs] = theta0[rs] + rand

    plt.plot(p0[walker][:], '*')
plt.plot(theta0, 'ko', markersize=12, alpha=0.5)
plt.plot(np.log10(infile['wtrue']), 'k-o')
plt.show(block=False)
pdb.set_trace()
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, threads=nthreads, args=[data, dbins])

# run emcee
tic = time.time()
sampler.run_mcmc(p0, 1500)
toc = time.time()
print((toc-tic)/60.)

# save the results in a binary file
np.save('saved_results_20percentdiff10_'+basename,sampler.chain)
