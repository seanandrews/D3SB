# CASA script to generate synthetic measurement set(s); change this to give you
# whatever you would like.

import numpy as np

# filename
simname = 'dummy'
config1 = 'alma.cycle3.3'
config2 = 'alma.cycle3.6'

# simulate observations in short spacings setup
default('simobserve')
simobserve(project=simname+'.c1.sim', skymodel=simname+'.fits', \
           incenter='340GHz',inwidth='8GHz',integration='30s', \
           refdate='2015/05/01',antennalist=config1+'.cfg', totaltime='1800s', \
           overwrite=T, graphics='none', verbose=False)

# simulate observations with long baselines
default('simobserve')
simobserve(project=simname+'.c2.sim', skymodel=simname+'.fits', \
           incenter='340GHz',inwidth='8GHz',integration='20s', \
           refdate='2015/05/02',antennalist=config2+'.cfg', totaltime='5400s', \
           overwrite=T, graphics='none', verbose=False)


# Export the individual MSs to be more accessible in Python
dir1 = simname+'.c1.sim/'
vis1 = simname+'.c1.sim.'+config1
dir2 = simname+'.c2.sim/'
vis2 = simname+'.c2.sim.'+config2
dirs = [dir1, dir2]
viss = [vis1, vis2]

for ix in np.arange(len(dirs)):

    # Extract data from MS table
    tb.open(dirs[ix]+viss[ix]+'.ms')
    data   = tb.getcol("DATA")
    flag   = tb.getcol("FLAG")
    uvw    = tb.getcol("UVW")
    weight = tb.getcol("WEIGHT")
    spwid  = tb.getcol("DATA_DESC_ID")
    tb.close()

    # Get the frequency information
    tb.open(dirs[ix]+viss[ix]+'.ms/SPECTRAL_WINDOW')
    freq = tb.getcol("CHAN_FREQ")
    tb.close()

    # Get rid of any flagged columns 
    flagged   = np.all(flag, axis=(0, 1))
    unflagged = np.squeeze(np.where(flagged == False))
    data   = data[:,:,unflagged]
    weight = weight[:,unflagged]
    uvw    = uvw[:,unflagged]
    spwid  = spwid[:,unflagged]

    # Break out the u, v spatial frequencies (in **lambda** units)
    freqlist = freq[0]
    findices = lambda ispw: freqlist[ispw]
    freqs = findices(spwid)
    u = uvw[0,:]*freqs/2.9979e8
    v = uvw[1,:]*freqs/2.9979e8

    # Assign uniform spectral-dependence to the weights
    sp_wgt = np.zeros_like(data.real)
    for i in range(len(freq)): sp_wgt[:,i,:] = weight

    # (weighted) average the polarizations
    Re  = np.sum(data.real*sp_wgt, axis=0) / np.sum(sp_wgt, axis=0)
    Im  = np.sum(data.imag*sp_wgt, axis=0) / np.sum(sp_wgt, axis=0)
    Vis = np.squeeze(Re + 1j*Im)
    Wgt = np.squeeze(np.sum(sp_wgt, axis=0))

    # output to numpy file
    os.system('rm -rf '+viss[ix]+'.vis.npz')
    np.savez(viss[ix]+'.vis', u=u, v=v, Vis=Vis, Wgt=Wgt)


# make a concatenated MS to stuff in simulation for CASA imaging
cfiles = [dir1+vis1+'.ms', dir2+vis2+'.ms']
os.system('rm -rf dummy.ms')
concat(vis=cfiles, concatvis=simname+'.ms')

# clean up garbage
os.system('rm -rf *.log *.last')
