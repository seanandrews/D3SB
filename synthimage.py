import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import sys
import pdb

def synthguess(a, b, nbins, filename):

    # -- SYNTHESIZED IMAGE
    # read in image, grab pixel dimensions in arcseconds
    pimage, hdr = fits.getdata('DATA/'+filename, header=True)
    pimage = np.squeeze(pimage)
    nx  = hdr['NAXIS1']
    ny  = hdr['NAXIS2']
    xps = np.absolute(hdr['CDELT1']*3600.) #angular width of the pixels in the RA and DEC dimensions
    yps = np.absolute(hdr['CDELT2']*3600.)

    bmaj =  hdr['BMAJ']*3600
    bmin =  hdr['BMIN']*3600
    
    # construct the radial grid for the image (generic for projected viewing geometry)
    incl = 0.
    PA   = 0.
    xp_,yp_ = np.meshgrid(xps*(np.arange(nx)-(nx/2.-0.5)),yps*(np.arange(ny)-(ny/2.-0.5)))
    ang = np.radians(270.-PA)
    xp  = np.cos(ang)*xp_-np.sin(ang)*yp_
    yp  = np.sin(ang)*xp_+np.cos(ang)*yp_
    rim = np.sqrt(xp*xp+1./np.cos(np.radians(incl))**2*yp*yp)
    
    # scale surface brightnesses to Jy/arcsec^2 units

    ##    pimage /= xps*yps     #Originally used pixel size
    pimage /= np.pi * bmaj * bmin/ (4. * np.log(2))
    
    # flatten into a profile
    SBpim = pimage.flatten()
    rpim  = rim.flatten()
    
    
    # -- DISCRETE DECOMPOSITION from PERFECT IMAGE
    # define the annular bins (as in d3sbFit.py)
    #nbins = 15
    ## b = np.linspace(0.1, 3., num=nbins)
    ## a = np.roll(b, 1)
    ## dpc=140.
    ## rin = 0.01/dpc
    ## a[0] = rin
    cb = 0.5*(a+b)
    ## bins = rin, b
    
    # calculate the average surface brightness in each bin
    SBdscp = np.zeros_like(cb)
    for i in range(nbins):
        SBdscp[i] = np.mean(SBpim[((rpim>a[i]) & (rpim<b[i]))])

    # plot things
    ## plt.figure(2)
    ## plt.plot(rpim, SBpim, '.r', cb, SBdscp, 'og')
    ## plt.xscale('log')
    ## plt.yscale('log')
    ## plt.show(block='False')

    return SBdscp
