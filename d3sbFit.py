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
dpc = 140.
rout = 100./dpc
#gam = 0.
gam = 1.0
Ftot = 0.2

# get data
data = getVis('DATA/'+hiresvis)

#corrupt data
u, v, dreal, dimag, dwgt = data
mu = 0
## dwgt = dwgt * 100. 
## data = u,v,dreal, dimag, dwgt
sigma = 1./np.sqrt(dwgt)

#realcorr = [np.random.normal(mu, i) for i in sigma]
#imagcorr = [np.random.normal(mu, i) for i in sigma]
#drealcorr = (dreal + realcorr)
#dimagcorr = (dimag + imagcorr)

## plt.figure(0)
## plt.subplot(1,2,1)
## plt.plot(dreal, dimag, 'ko')
## plt.title('Original')pdb.set_trace()
## plt.subplot(1,2,2)
## plt.plot(drealcorr, dimagcorr, 'ko')
## plt.title('Corrupted')
## plt.show(block='False')
#pdb.set_trace()

#replace existing data with corrupted data
#data = u,v,drealcorr, dimagcorr, dwgt


#Model surface brightness
rvals = np.logspace(-2.1, .7, num=200) #Radial range chosen here. Not the actual input range.
wtrueall = ((2-gam)*Ftot/(2*np.pi*rout**2)) * (rvals/rout)**(-gam)*np.exp(-(rvals/rout)**(2.-gam))

    
# choose radial bins
binsizes = np.arange(10,11,5)
for nbins in binsizes:
    b = np.linspace(0.1, 3., num=nbins)
    a = np.roll(b, 1)
    rin = 0.01/140.
    a[0] = rin
    cb = 0.5*(a+b)
    dbins = rin, b

    wtrue = ((2-gam)*Ftot/(2*np.pi*rout**2)) * (cb/rout)**(-gam)*np.exp(-(cb/rout)**(2.-gam))
    wg = synthguess(a, b, nbins, synthimg)



    
    # initial guess for weights (based on Gaussian fit to fake visibilities)
    sig = 0.3
    nrm = 0.1/(2.*np.pi*sig**2) #This is the same as 
    ##wg2 = nrm*np.exp(-0.5*(cb/sig)**2)
    ##    wg[wg <= 1e-4] = 1e-4

    # optimization
    print "Entering minimization"
    opt = minimize(opt_func, np.log10(wg), args=(data, dbins))
    w0 = 10.**(opt.x)
    print(w0)
    print opt.message

    
    opt2 = minimize(opt_func, np.log10(wg), args=(data, dbins), method='Nelder-Mead', options={'maxfev': 10000})
    w02 = 10.**(opt2.x)
    print(w02)
    print opt2

    ## opt3 = minimize(opt_func, np.log10(wg), args=(data, dbins), method='Powell')
    ## w03 = 10.**(opt3.x)
    ## print(w03)
    ## print opt3.message
    print "Left minimization"
    ####    filename = 'opt_bin_corr'+str(nbins)
    filename = 'opt_'+basename+'_'+str(nbins)

    ## np.savez(filename, cb=cb, w0=w0, wtrue=wtrue)
    np.savez(filename, cb=cb, wg=wg, w02=w02, w0=w0, wtrue=wtrue)
    ####np.save('opt_guess',w0)
    ####sys.exit()
#w0 = np.load('opt_guess.npy')
#pdb.set_trace()

#Plot different weights
plt.figure(1)
#ax = plt.gca()
#plt.plot(rvals, wtrueall, 'k')
colors = cm.rainbow(np.linspace(0,1,len(binsizes)))
for i, (nbins, c) in enumerate(zip(sorted(binsizes), colors)):
    #    infile = np.load('opt_bin'+str(nbins)+'.npz')
    #    plt.plot(infile['cb'], infile['wg'], 'go', infile['cb'], infile['w0'], 'ro')

    #    plotindex = (i//3)*6 + (i%3 + 1) #Integer division
    #    plt.subplot(3,2,i+1)#plotindex)
    #    plt.plot(infile['cb'], infile['wg'], 'gs')#, color=c)
    #    plt.plot(infile['cb'], infile['w0'], 'ro')#, color=c)

    infilecorr = np.load('opt_'+basename+'_'+str(nbins)+'.npz')
    plt.subplot(3,1,1)
    plt.plot(rvals, wtrueall, 'k')

    plt.plot(infilecorr['cb'], infilecorr['wtrue'], 'ko', markersize = 8, label='True')
    plt.plot(infilecorr['cb'], infilecorr['wg'], 'bs', markersize=12, label='Guess (Mean img)')
    plt.plot(infilecorr['cb'], infilecorr['w0'], 'ro', label='BFGS')
    plt.plot(infilecorr['cb'], infilecorr['w02'], 'co', label='Simplex')
    #    plt.plot(infilecorr['cb'], infilecorr['w03'], 'go', label='Anneal')

    #    plt.plot(infilecorr['cb'], infilecorr['wg2'], 'bs', alpha=0.2, markersize=12)#, color=c)
    ## plt.show()
    ## pdb.set_trace()


    plt.xlim(0.013,4.1)
    plt.ylim(1e-9, 5)
    ax = plt.gca()
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.title(str(nbins)+' bins')
    plt.ylabel('Jy/arcsec^2')
    plt.xlabel('Angle [arcsec]')


    # Shrink current axis's height by 10% on the bottom
    ax.legend(loc='lower left', bbox_to_anchor=(0.05, 0.1),numpoints=1, fontsize = 10)

    ## if ((i==0) or (i==3)):
    ##     plt.xlabel('Angle ["]')
    ##     plt.ylabel(r'Intensity [Jy/arcsec$^2$]')
    ##plt.show()
    ##    pdb.set_trace()
        
    plotindex2 = (i//3)*6 + (i%3 + 4) #Integer division
    #    plt.subplot(4,3,plotindex2)
    plt.subplot(3,1,3)
    plt.hist(np.true_divide(np.subtract(infilecorr['w0'],infilecorr['wtrue']), infilecorr['wtrue']), histtype='stepfilled', color='r')
    plt.hist(np.true_divide(np.subtract(infilecorr['wg'],infilecorr['wtrue']), infilecorr['wtrue']),histtype='step', color='b',linewidth=3)
    #    plt.hist(np.true_divide(np.subtract(infilecorr['w03'],infilecorr['wtrue']), infilecorr['wtrue']), histtype='stepfilled', color='g', alpha=0.8)
    plt.hist(np.true_divide(np.subtract(infilecorr['w02'],infilecorr['wtrue']), infilecorr['wtrue']), histtype='stepfilled', color='c', alpha=0.6)

    plt.xlim(-1.2, 1.2)
    plt.ylim(0, 2.1)
    plt.xlabel('Residual (Value-True)/True')
    plt.ylabel('Number')
    #    box = ax.get_position()
    #ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    ## plt.hist(np.subtract(infilecorr['w0'],infilecorr['wtrue']), bins=np.logspace(0.1,3,20), histtype='stepfilled', color='c', alpha=0.6)

    plt.subplot(3,1,2)
    plt.plot(infilecorr['cb'],np.true_divide(np.subtract(infilecorr['wg'],infilecorr['wtrue']), infilecorr['wtrue']), 'bs', markersize=12)
    plt.plot(infilecorr['cb'],np.true_divide(np.subtract(infilecorr['w0'],infilecorr['wtrue']), infilecorr['wtrue']), 'ro')
    plt.plot(infilecorr['cb'],np.true_divide(np.subtract(infilecorr['w02'],infilecorr['wtrue']), infilecorr['wtrue']), 'co')
    #    plt.plot(infilecorr['cb'],np.true_divide(np.subtract(infilecorr['w03'],infilecorr['wtrue']), infilecorr['wtrue']), 'go')
    plt.plot([0.01, 5], [0, 0], '--k')
    plt.xlim(0.013,4.1)
    plt.ylim(-1.2, 1.2)
    ax = plt.gca()
    ax.set_xscale('log')
    plt.xlabel('Angle [arcsec]')
    plt.ylabel('Residual (Value-True)/True')
        
    #    print np.amax(np.true_divide(np.subtract(infilecorr['w03'],infilecorr['wtrue']), infilecorr['wtrue']))
    #    print np.amin(np.true_divide(np.subtract(infilecorr['w03'],infilecorr['wtrue']), infilecorr['wtrue']))
    
    #    plt.hist(np.subtract(infile['wg'],infile['wtrue']), bins=20, histtype='stepfilled', color='g',alpha=0.5)
    ## if ((i==0) or (i==3)):
    ##     plt.xlabel('Residual (W0-Wtrue)')
    ##     plt.ylabel('Frequency')
    #    plt.xlim(-0.4,0.45)
    #ax.set_yscale('log')
    #plt.tight_layout(pad=0.5)
plt.show(block=False)
pdb.set_trace()
plt.savefig(basename+'_resid.png', bbox_inches='tight')
# set up walkers
ndim, nwalkers, nthreads = nbins, 60, 6
theta0 = np.log10(w0)
p0 = [theta0 + np.random.uniform(-1, 1, ndim) for i in range(nwalkers)]

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, threads=nthreads, args=[data, bins])

# run emcee
tic = time.time()
sampler.run_mcmc(p0, 200)
toc = time.time()
print((toc-tic)/60.)

# save the results in a binary file
np.save('saved_results',sampler.chain)
