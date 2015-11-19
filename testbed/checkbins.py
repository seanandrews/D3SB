import numpy as np
import scipy as sc
import os
import sys
import matplotlib.pyplot as plt
from matplotlib import gridspec
from discreteModel import discreteModel


filename = 'testA'


# define the "true" SB distribution
nbins = 4000
b = 0.001 + 0.001*np.arange(nbins)
a = np.roll(b, 1)
rin = 0.1/140.
a[0] = rin
r = 0.5*(a+b)

flux = 0.12
sig = 0.5
incl = PA = offx = offy = 0.0
SB = (r/sig)**-0.75 * np.exp(-0.5*(r/sig)**2.5)
int_SB = np.trapz(2.*np.pi*SB*r, r)  
SB *= flux/int_SB


# define the "binned" SB distribution for the linear case
nlbins = 38
lb = 0.04 + 0.04*np.arange(nlbins)
la = np.roll(lb, 1)
rin = 0.1/140.
la[0] = rin
lcb = 0.5*(la+lb)
lbins = rin, lb
lSB = (lcb/sig)**-0.75 * np.exp(-0.5*(lcb/sig)**2.5)
lSB *= flux/int_SB
lstep_SB = np.zeros_like(r)
for i in np.arange(nlbins): lstep_SB[(r > la[i]) & (r <= lb[i])] = lSB[i]


# define the "binned" SB distribution for a 1/r-distributed case
nvbins = 38
vb = 1./np.linspace(1./0.04, 1./5.5, num=nvbins)
va = np.roll(vb, 1)
va[0] = rin
vcb = 0.5*(va+vb)
vbins = rin, vb
vSB = (vcb/sig)**-0.75 * np.exp(-0.5*(vcb/sig)**2.5)
vSB *= flux/int_SB
vstep_SB = np.zeros_like(r)
for i in np.arange(nvbins): vstep_SB[(r > va[i]) & (r <= vb[i])] = vSB[i]


# define the "binned" SB distribution for an equal integrated-S/N case
#rrun = r[r>0.04]
#SBrun = SB[r>0.04]
#noise = 0.025e-3	# Jy/beam
#omegab = np.pi*0.07*0.07/(4.*np.log(2.))	# beam area in arcsec**2
#thresh = 50.

#rcut = 0.04
#xb = np.array([])
#while (rcut < np.amax(rrun)):
#    rrun = b[b>rcut]
#    SBrun = SB[b>rcut]
#    fcum = sc.integrate.cumtrapz(SBrun*2.*np.pi*rrun, rrun, initial=0.)
#    scum = sc.integrate.cumtrapz(2.*np.pi*rrun, rrun, initial=0.)
#    snr = fcum/(noise*scum/omegab)
#    snr[0] = snr[1]
#    print(snr[0:100])
#    print(rrun[0:100])
#    sys.exit()
#    xb = np.append(xb, rcut)
#xa = np.roll(xb, 1)
#xa[0] = rin
#xcb = 0.5*(xa+xb)
#xbins = rin, xb
#nxbins = len(xb)
#xSB = (xcb/sig)**-0.75 * np.exp(-0.5*(xcb/sig)**2.5)
#xSB *= flux/int_SB
#xstep_SB = np.zeros_like(r)
#for i in np.arange(nxbins): xstep_SB[(r > xa[i]) & (r <= xb[i])] = xSB[i]
#print(nxbins)


# do a hybrid linear-log SB distribution
nxbins = 40
print(lb)
xb1 = 0.025 + 0.025*np.arange(nxbins-20)
xb2 = np.logspace(np.log10(xb1[-1]), np.log10(2.0), num=21)
print(xb1)
xb = np.concatenate([xb1[:-1], xb2])
xa = np.roll(xb, 1)
rin = 0.1/140.
xa[0] = rin
xcb = 0.5*(xa+xb)
xbins = rin, xb
xSB = (xcb/sig)**-0.75 * np.exp(-0.5*(xcb/sig)**2.5)
xSB *= flux/int_SB
xstep_SB = np.zeros_like(r)
for i in np.arange(nxbins): xstep_SB[(r > xa[i]) & (r <= xb[i])] = xSB[i]



# plot these both
plt.axis([0.01, 10, 1e-6, 10])
plt.loglog(r, SB, 'k', r, lstep_SB, 'r', r, vstep_SB, 'g', r, xstep_SB, 'b')
plt.savefig('SB.pdf')
plt.clf()




# load the "data" (w/ and w/o noise)
data = np.load(filename+'.vis.npz')
u = data['u']
v = data['v']
rho = np.sqrt(u**2 + v**2)
ivis = data['nf_Vis']
vis = data['Vis']
wgt = data['Wgt']


# binned models for the data
ltheta = incl, PA, np.array([offx, offy]), lSB
lvis = discreteModel(ltheta, [u, v], lbins)

vtheta = incl, PA, np.array([offx, offy]), vSB
vvis = discreteModel(vtheta, [u, v], vbins)

xtheta = incl, PA, np.array([offx, offy]), xSB
xvis = discreteModel(xtheta, [u, v], xbins)




# plot them and the residuals (real only, since symmetric for now)
plt.figure(figsize=(7, 10))
plt.clf()
gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1])

# visibilities
ax0 = plt.subplot(gs[0])
ax0.axis([0, 3000, -0.005, 0.12])
ax0.plot([0, 3000], [0, 0], '--k', alpha=0.5)
ax0.plot(1e-3*rho, ivis.real, '.k', alpha=0.1)
ax0.plot(1e-3*rho, lvis.real, '.r', alpha=0.1)
ax0.plot(1e-3*rho, vvis.real, '.g', alpha=0.1)
ax0.plot(1e-3*rho, xvis.real, '.b', alpha=0.1)


# residuals
ax1 = plt.subplot(gs[1])
lresid = lvis.real-ivis.real
vresid = vvis.real-ivis.real
xresid = xvis.real-ivis.real
max_resid = np.amax([np.amax(np.abs(lresid)), np.amax(np.abs(vresid)), np.amax(np.abs(xresid))])
ax1.axis([0, 3000, -2.*max_resid, 2.*max_resid])
ax1.plot([0, 3000], [0, 0], '--k', alpha=0.5)
ax1.plot(1e-3*rho, lresid, '.r', alpha=0.1)
ax1.plot(1e-3*rho, vresid, '.g', alpha=0.1)
ax1.plot(1e-3*rho, xresid, '.b', alpha=0.1)



# cumulative chi**2
ix_srt = np.argsort(rho)
lresid = np.sqrt(wgt)*(vis.real-lvis.real)
lcum_chi2 = np.cumsum(lresid[ix_srt]**2)
vresid = np.sqrt(wgt)*(vis.real-vvis.real)
vcum_chi2 = np.cumsum(vresid[ix_srt]**2)
xresid = np.sqrt(wgt)*(vis.real-xvis.real)
xcum_chi2 = np.cumsum(xresid[ix_srt]**2)
vcum_chi2 /= lcum_chi2[-1]
xcum_chi2 /= lcum_chi2[-1]
lcum_chi2 /= lcum_chi2[-1]
ax2 = plt.subplot(gs[2])
ax2.axis([0, 3000, 0, 1.0])
ax2.plot([0, 3000], [0, 0], '--k', alpha=0.5)
ax2.plot(1e-3*rho[ix_srt], lcum_chi2, '.r')
ax2.plot(1e-3*rho[ix_srt], vcum_chi2, '.g')
ax2.plot(1e-3*rho[ix_srt], xcum_chi2, '.b')
plt.show()
plt.clf()

