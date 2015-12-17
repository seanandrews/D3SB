import numpy as np
from discrete1d_model import discrete1d_model

def lnprob(p, data, bins):

    # unpack data
    rho, dvis, dsig = data


    # PRIORS

    # enforce positive surface brightnesses
    if (np.any(p) < 0.):
        return -np.inf

    # compute penalty for oscillations
    dcoeff = np.diff(p)
    prior = 1.*np.sum(dcoeff[1:]*dcoeff[:-1] < 0) * 2.*len(dvis) / len(p)



    # generate model visibilities
    mvis = discrete1d_model(p, rho, bins)


    # compute a chi2 value (proportional to log-likelihood)
    chi2 = np.sum(((dvis.real-mvis.real)/dsig.real)**2 + 
                  ((dvis.imag-mvis.imag)/dsig.imag)**2)

    # return a log-posterior value
    return -0.5*(chi2 + prior)

