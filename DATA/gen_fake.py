import numpy as np
from astropy.io import fits
import os

def gen_fake(p):

  # - constants
  AU = 1.496e13                 # - definition of an AU [cm]
  pc = 3.09e18                  # - definition of a parsec [cm]

  # - parameter values
  offs = [0.0,0.0]
  incl = 0.                            # - disk inclination (deg from face-on)
  PA   = 0.                             # - PA of major axis (deg E of N)
  dpc  = 140.                           # - distance (pc)
  rin   = 0.1*AU
  ra_pc  = 68.1263333333	#(RA[0]+RA[1]/60.+RA[2]/3600.)*360./24.
  dec_pc = 17.5279638889	#(DEC[0]+DEC[1]/60.+DEC[2]/3600.)

  # - set free parameters
  rout  = p[0]*AU
  ftot  = p[1]
  gam   = p[2]

  # - radial grid
  nx = 1001                 # - preferrably an odd number
  ny = 1001                 # - best if this is square
  rwid = 300.*AU            # - 1/2 width of image
  xps = 2.*rwid/(nx-1)
  yps = 2.*rwid/(ny-1)
  xp_,yp_ = np.meshgrid(xps*(np.arange(nx)-(nx/2.-0.5)),yps*(np.arange(ny)-(ny/2.-0.5)))
  ang = (270.-PA)*np.pi/180.
  xp = np.cos(ang)*xp_-np.sin(ang)*yp_
  yp = np.sin(ang)*xp_+np.cos(ang)*yp_
  r = np.sqrt(xp*xp+1./np.cos(incl*np.pi/180.)**2*yp*yp)

  # - create image
  image = (r/rout)**(-gam)*np.exp(-(r/rout)**(2.-gam))
  image[r<rin] = 0.
  image *= ftot/np.sum(image)

  # - construct header
  hdr = fits.Header()
  hdr.set('BITPIX',-32)
  hdr.set('CDELT1',-1.*xps/(AU*dpc)/3600.)
  hdr.set('CRPIX1',nx/2.+0.5)
  hdr.set('CRVAL1',ra_pc+offs[0]/np.cos(dec_pc*np.pi/180)/3600.)
  hdr.set('CTYPE1',  'RA---SIN')
  hdr.set('CDELT2',yps/(AU*dpc)/3600.)
  hdr.set('CRPIX2',ny/2.+0.5)
  hdr.set('CRVAL2',dec_pc+offs[1]/3600.)
  hdr.set('CTYPE2','DEC--SIN')
  hdr.set('CTYPE4','FREQ    ')
  hdr.set('CRPIX4',1)
  hdr.set('CDELT4',1.96799993515E+09)
  hdr.set('CRVAL4',2.24053329468E+11)
  hdr.set('CTYPE3','STOKES  ')
  hdr.set('CRVAL3',1.)
  hdr.set('CDELT3',1.)
  hdr.set('CRPIX3',1.)
  hdr.set('EPOCH' ,2000)

  # - write image
  hdu = fits.PrimaryHDU(np.reshape(np.float32(image),(1,1,ny,nx)),header=hdr)
  hdulist = fits.HDUList([hdu])
  hdulist.writeto('model.fits',clobber=True)
  hdulist.close()

  # calculate visibilities
  os.system('./pvis.csh')
