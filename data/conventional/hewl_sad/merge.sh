mkdir merge

careless mono \
    --anomalous \
    --embed \
    --merge-files=True \
    --iterations=100000 \
    "image_id,XDET,YDET,BG,SIGBG,LP,QE,FRACTIONCALC" \
    integrated_pass1.mtz \
    integrated_pass2.mtz \
    merge/hewl
