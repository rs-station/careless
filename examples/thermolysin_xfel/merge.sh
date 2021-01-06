mkdir merge 
careless mono \
  --iterations=10000 \
  --image-scale-key=BATCH \
  --dmin=1.8 \
  --anomalous \
  --learning-rate=0.001 \
  "dHKL,BATCH,xobs,yobs" \
  unmerged.mtz \
  merge/thermolysin

mkdir merge_eo
careless mono \
  --iterations=10000 \
  --image-scale-key=BATCH \
  --dmin=1.8 \
  --anomalous \
  --learning-rate=0.001 \
  "dHKL,BATCH,xobs,yobs,ewald_offset" \
  unmerged.mtz \
  merge_eo/thermolysin

