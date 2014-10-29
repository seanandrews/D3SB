import numpy as np
from d3sb_model import d3sb_model

def lnprob(theta, data, bins):

    u, v, dreal, dimag, dwgt = data
    uvsamples = u, v

    mreal = d3sb_model(theta, uvsamples, bins)
    mimag = np.zeros_like(u)

    chi2 = np.sum(dwgt*(dreal-mreal)**2) + np.sum(dwgt*(dimag-mimag)**2)
    lnp = -0.5*chi2

    return lnp
