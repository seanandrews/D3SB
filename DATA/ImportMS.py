import numpy as np

# original .ms file name
oms_path = 'base'
oms_name = 'dummy'
mdl_name = 'test'

# copy the data file into a model 
os.system('rm -rf '+mdl_name+'.ms')
os.system('cp -r '+oms_path+'/'+oms_name+'.ms '+mdl_name+'.ms')

# load the data
tb.open(mdl_name+'.ms')
data = tb.getcol("DATA")
weight = tb.getcol("WEIGHT")
flag = tb.getcol("FLAG")
tb.close()

# Note the flagged columns
flagged = np.all(flag, axis=(0, 1))
unflagged = np.squeeze(np.where(flagged == False))

# load the model file (presume this is just an array of complex numbers, with 
# the appropriate sorting/ordering in original .ms file; also assume that the 
# polarizations have been averaged, and that the model is unpolarized)
mvis = (np.load(mdl_name+'.vis.npz'))['Vis']
mwgt = (np.load(mdl_name+'.vis.npz'))['Wgt']

# replace the original data with the model
data[:,:,unflagged] = mvis
weight[:,unflagged] = mwgt

# now re-pack those back into the .ms
tb.open(mdl_name+'.ms', nomodify=False)
tb.putcol("DATA", data)
tb.putcol("WEIGHT", weight)
tb.flush()
tb.close()
