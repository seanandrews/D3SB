import numpy as np

# original .ms file name
oms_path = 'base'
oms_name = 'dummy'
mdl_name = 'testA'

# copy the data file into a model 
os.system('rm -rf '+mdl_name+'.ms')
os.system('cp -r '+oms_path+'/'+oms_name+'.ms '+mdl_name+'.ms')

# load the data
tb.open(mdl_name+'.ms')
data = tb.getcol("DATA")
tb.close()

# load the model file (presume this is just an array of complex numbers, with 
# the appropriate sorting/ordering in original .ms file; also assume that the 
# polarizations have been averaged, and that the model is unpolarized)
mvis = (np.load(mdl_name+'.vis.npz'))['Vis']

# replace the original data with the model
data[0,:,:] = mvis
data[1,:,:] = mvis

# now re-pack those back into the .ms
tb.open(mdl_name+'.ms', nomodify=False)
tb.putcol("DATA", data)
tb.flush()
tb.close()
