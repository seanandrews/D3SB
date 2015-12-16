import numpy as np
import scipy.special as sc

def discreteModel(theta, uvsamples, bins):

    # retrieve inputs
    incl, PA, offset, w = theta
    rin, b = bins
    u, v = uvsamples			# in **lambda** units

    # convert angles to radians
    inclr = np.radians(incl)
    PAr = 0.5*np.pi-np.radians(PA)
    offr = offset * np.pi / (180.*3600.)

    # coordinate change to deal with projection, rotation, and shifts
    #uprime = ((u-offr[0])*np.cos(PAr) + (v-offr[1])*np.sin(PAr)) * np.cos(inclr)
    #vprime = (-(u-offr[0])*np.sin(PAr) + (v-offr[1])*np.cos(PAr))
    uprime = (u*np.cos(PAr) + v*np.sin(PAr)) 
    vprime = (-u*np.sin(PAr) + v*np.cos(PAr)) * np.cos(inclr)
    rho = np.sqrt(uprime**2 + vprime**2) * np.pi / (180.*3600.)

    # re-orient arrays
    rbin = np.concatenate([np.array([rin]), b])
    wbin = np.append(np.concatenate([np.array([0.0]), w]), 0.)
    ww = wbin-np.roll(wbin, -1)
    intensity = np.delete(ww, b.size+1)

    # compute the visibilities
    jarg = np.outer(2.*np.pi*rbin, rho)
    jinc = sc.j1(jarg)/jarg
    vis = np.dot(2.*np.pi*rbin**2*intensity, jinc)

    # impart a phase center shift
    shift = np.exp(-2.*np.pi*1.0j*((u*-offr[0]) + (v*-offr[1])))
    model_vis = vis*shift

    return model_vis
