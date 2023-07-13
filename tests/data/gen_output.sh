mkdir output
careless poly \
    --disable-progress-bar \
    --iterations=10 \
    --merge-half-datasets \
    --half-dataset-repeats=3 \
    --test-fraction=0.1 \
    --disable-gpu \
    --anomalous \
    --wavelength-key="Wavelength" \
    "dHKL,Hobs,Kobs,Lobs,Wavelength" \
    pyp_off.mtz \
    pyp_2ms.mtz \
    output/pyp

