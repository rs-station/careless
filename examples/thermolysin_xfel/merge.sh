mkdir merge

careless mono \
  --iterations=10000 \
  --anomalous \
  --learning-rate=0.001 \
  "dHKL,BATCH,xobs,yobs" \
  unmerged.mtz \
  merge/thermolysin

mkdir merge_eov

careless mono \
  --iterations=10000 \
  --anomalous \
  --learning-rate=0.001 \
  "dHKL,BATCH,xobs,yobs,cartesian_delta_x,cartesian_delta_y,cartesian_delta_z" \
  unmerged.mtz \
  merge_eov/thermolysin


