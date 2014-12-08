from lnprob import lnprob

def opt_func(theta, data, bins):

    chisq = -2.*lnprob(theta, data, bins)

    return chisq
