import numpy as np
from astropy.io import fits
import os

def dummy_image(nx=512, ny=512, cellsize=0.01):

    # define a two-dimensional grid for the image
    xp, yp = np.meshgrid(cellsize*(np.arange(nx)-0.5*nx+0.5), \
                         cellsize*(np.arange(ny)-0.5*ny+0.5))
    r = np.sqrt(xp**2 + yp**2)

    # assign an arbitrary radial brightness profile to the image
    # (you will fix this later to be any brightness profile you want)
    Ixy = np.exp(-0.5*(r/0.3)**2.)

    # make a FITS header
    hdr = fits.Header()
    hdr.set('BITPIX', -32)
    hdr.set('CDELT1', -cellsize/3600.)
    hdr.set('CRPIX1', nx/2+0.5)
    hdr.set('CRVAL1', 165.)
    hdr.set('CTYPE1', 'RA---SIN')
    hdr.set('CDELT2', cellsize/3600.)
    hdr.set('CRPIX2', ny/2+0.5)
    hdr.set('CRVAL2', -35.)
    hdr.set('CTYPE2', 'DEC--SIN')
    hdr.set('CDELT4', 1)
    hdr.set('CRPIX4', 1)
    hdr.set('CRVAL4', 340e9)
    hdr.set('CTYPE4', 'FREQ    ')
    hdr.set('CDELT3', 1)
    hdr.set('CRPIX3', 1)
    hdr.set('CRVAL3', 1)
    hdr.set('CTYPE3', 'STOKES  ')
    hdr.set('EPOCH' , 2000)

    # embed image into FITS file (with name 'dummy.fits')
    hdu = fits.PrimaryHDU(np.reshape(np.float32(Ixy), (1,1,ny,nx)), header=hdr)
    hdulist = fits.HDUList([hdu])
    os.system('rm -rf '+'dummy.fits')
    hdulist.writeto('dummy.fits', clobber=True)
    hdulist.close()

    return 0
