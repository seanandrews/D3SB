# NAME INPUT VARIABLES
set file=model.vis
set dir=.
set name=model

goto start
start:

# FOURIER INVERSION TO CREATE A DIRTY MAP
rm -r $name.map $name.beam
invert vis=$file map=$name.map beam=$name.beam line=channel,1,1 \
       imsize=1024 cell=0.025 robust=2.0 options=systemp,mfs 

# CLEAN DECONVOLUTION AND SYNTHESIZED BEAM RESTORATION
rm -r $name.clean
clean  map=$name.map beam=$name.beam out=$name.clean \
       region='arcsec,box(3,-3,-3,3)' niters=5500 cutoff=0.0015

rm -r $name.cm
restor model=$name.clean beam=$name.beam map=$name.map out=$name.cm fwhm=0.55

cgcurs_start:
cgdisp in=$name.cm type=con \
       slev=a,0.00075  \
       levs1=3,6,9,12,15,18,21,24,27,30,33,36,39,42,45,48,51,54,57,60 \
       labtyp=arcsec options=mirror beamtyp="b,l" \
       region='arcsec,box(3,-3,-3,3)' device=/xs

fits in=$name.cm op=xyout out=$dir/$name.fits

goto end

cgcurs in=$name.cm options=stat \
       type=con slev=a,0.00275 \
       levs=-3,3,4,5,6,7,8,9,10,11,12,13,14,15 \
       labtyp=arcsec region='arcsec,box(15,-15,-15,15)' \
       device=/xs
goto end

end:
