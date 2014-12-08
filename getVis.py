import numpy as np
from astropy.io import fits

def getVis(datafile):

   # Tool for extracting visibility data from a .FITS file.
   
    dvis  = fits.open(datafile)
    dreal = np.squeeze(dvis[0].data['Data'])[:,0]
    dimag = np.squeeze(dvis[0].data['Data'])[:,1]
    dwgt  = np.squeeze(dvis[0].data['Data'])[:,2]   
    u = 2.24053329468e11*1e-3*dvis[0].data.par(0)[:]
    v = 2.24053329468e11*1e-3*dvis[0].data.par(1)[:]
    dvis.close()

    data = u, v, dreal, dimag, dwgt

    return data
