import numpy as np
import scipy.special as sc

def discrete1d_model(theta, rho, bins):

    # re-orient arrays
    rin, b = bins
    rbin = np.concatenate([np.array([rin]), b])
    wbin = np.append(np.concatenate([np.array([0.0]), theta]), 0.)
    ww = wbin-np.roll(wbin, -1)
    intensity = np.delete(ww, b.size+1)

    jarg = np.outer(2.*np.pi*rbin, np.pi*rho/(180.*3600.))
    jinc = sc.j1(jarg)/jarg

    vis = np.dot(2.*np.pi*rbin**2*intensity, jinc)

    return vis
