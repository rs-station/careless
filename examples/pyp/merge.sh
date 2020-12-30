out=merge
mtz_0=off.mtz
mtz_1=2ms.mtz
mkdir $out

careless poly \
  --separate-files \
  --sequential-layers=20 \
  --iterations=10000 \
  --learning-rate=0.001 \
  --wavelength-key='Wavelength' \
  "X,Y,Wavelength,BATCH,dHKL,Hobs,Kobs,Lobs" \
  $mtz_0 \
  $mtz_1 \
  $out/pyp


