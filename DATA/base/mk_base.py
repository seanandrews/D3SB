import numpy as np
import os
import sys
from dummy_image import dummy_image

# make a dummy image that can be used to generate (u,v) tracks
cellsize = 0.005
nx = ny = 1024
foo = dummy_image(cellsize=cellsize, nx=nx, ny=ny)

# FT and sample the dummy image onto synthetic (u,v) tracks
# (user edits 'mk_MS.py' to set this up)
os.system('casapy --nologger -c mk_synthMS.py')
