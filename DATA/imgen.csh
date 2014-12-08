#!/bin/csh -f
# Use imgen to create simple models 

if ($1 == "") then
  echo "Syntax: imgen.csh <filename>"
  exit
endif

goto start
start:

set ra  = 4.5
set dec = 25
set harange = -8.0,8.0,0.0166666666666666666667
set ellim   = 18
set time    = 14NOV01:10:30:00.0
set freq    = 225
set cyc_on  = 0.4
set cyc_off = 0.1
set bw = 2

# optically thin, gaussian disk
rm -r $1.model
imgen object=gaussian spar=1.,1.,-1.,1.25,2.5,60 \
      options=totflux \
      imsize=2048 cell=0.005 \
      radec=$ra,$dec \
      out=$1.model 
puthd in=$1.model/naxis value=4
puthd in=$1.model/naxis3 value=1
puthd in=$1.model/naxis4 value=1
puthd in=$1.model/ctype3 value=FREQ
puthd in=$1.model/crval3 value=$freq
puthd in=$1.model/ctype4 value=STOKES
puthd in=$1.model/crval4 value=1
puthd in=$1.model/cdelt4 value=1
puthd in=$1.model/epoch value=2000

cgdisp device=/xs labtyp=arcsec in=$1.model region='arcsec,box(5,-5,-5,5)'

rm -rf sma.uv
uvgen ant=CE.ant baseunit=-3.33564 freq=$freq corr=1,1,0,$bw \
      radec=$ra,$dec lat=19.83 harange=$harange ellim=$ellim time=$time \
      jyperk=130 source=$MIRCAT/no.source out=sma.uv cycle=$cyc_on,$cyc_off

uvplt vis=sma.uv axis=uc,vc device=/xs options=nobase,equal

rm -r $1.vis $1.vis.fits
uvmodel vis=sma.uv model=$1.model options=replace out=$1.vis
fits in=$1.vis out=$1.vis.fits op=uvout

uvplt vis=$1.vis axis=uvdist,real device=/xs options=nobase

goto end

end:
