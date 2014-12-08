#! /bin/csh -f
set data=GGTau.1mm
set name=model

fits in=$name.fits out=$name.im op=xyin

rm -rf $name.vis
uvmodel vis=$data.vis model=$name.im options=replace out=$name.vis

fits in=$name.vis out=$name.vis.fits op=uvout

rm -rf $name.im
#rm -rf $name.vis
