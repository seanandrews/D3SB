import numpy as np
import sys
import os
from discreteModel import discreteModel


# the output filename
filename = 'testA'


# define a high resolution set of bins for an "idealized" model
nbins = 4000
b = 0.001 + 0.001*np.arange(nbins)
a = np.roll(b, 1)
rin = 0.1/140.
a[0] = rin
cb = 0.5*(a+b)
bins = rin, b

# the brightness profile of the model
flux = 0.12
sig = 0.5
incl = PA = offx = offy = 0.0
SB = (cb/sig)**-0.75 * np.exp(-0.5*(cb/sig)**2.5)
int_SB = np.trapz(2.*np.pi*SB*cb, cb)		# a check on the total flux
SB *= flux/int_SB
itheta = incl, PA, np.array([offx, offy]), SB


# FT and sample it for each configuration
ldata = np.load('../DATA/base/dummy.c1.sim.alma.cycle3.3.vis.npz')
lu = ldata['u']
lv = ldata['v']
lvis = ldata['Vis']
lrms = 0.035e-3
lo_vis = discreteModel(itheta, [lu, lv], bins)

hdata = np.load('../DATA/base/dummy.c2.sim.alma.cycle3.6.vis.npz')
hu = hdata['u']
hv = hdata['v']
hvis = hdata['Vis']
hrms = 0.025e-3
hi_vis = discreteModel(itheta, [hu, hv], bins)

combo_vis_nonoise = np.concatenate([lo_vis, hi_vis])



# add (white) noise to the model visibilities; record noise in weights
lnoise = np.random.normal(loc=0., scale=lrms*np.sqrt(len(lu)), size=len(lu))
lo_vis += lnoise + 1j*lnoise
lo_wgt = np.ones_like(lu)/(lrms*np.sqrt(len(lu)))**2

hnoise = np.random.normal(loc=0., scale=hrms*np.sqrt(len(hu)), size=len(hu))
hi_vis += hnoise + 1j*hnoise
hi_wgt = np.ones_like(hu)/(hrms*np.sqrt(len(hu)))**2



# concatenate and output
combo_u = np.concatenate([lu, hu])
combo_v = np.concatenate([lv, hv])
combo_vis = np.concatenate([lo_vis, hi_vis])
combo_wgt = np.concatenate([lo_wgt, hi_wgt])
os.system('rm -rf '+filename+'.vis.npz')
np.savez(filename+'.vis', u=combo_u, v=combo_v, Vis=combo_vis, nf_Vis=combo_vis_nonoise, Wgt=combo_wgt)
