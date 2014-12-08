import numpy as np
from get_vis import get_vis
from opt_func import opt_func
from scipy.optimize import minimize
import time

# get data
data = get_vis('model_faceon.vis.fits')

# choose radial bins
nbins = 10
b = np.linspace(0.1, 2., num=nbins)
a = np.roll(b, 1)
rin = 0.01/140.
a[0] = rin
cb = 0.5*(a+b)
bins = rin, b

# initial guess for weights
w0 = np.ones_like(cb)
sig = 0.15
nrm = 0.1/(2.*np.pi*sig**2)
w0 = nrm*np.exp(-0.5*(cb/sig)**2)
w0[w0 <= 1e-4] = 1e-4


tic = time.time()
res = minimize(opt_func, np.log10(w0), args=(data, bins))
toc = time.time()
print((toc-tic)/60.)

print(res.x)

np.save('opt_results',res)
