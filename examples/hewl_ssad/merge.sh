#########################################################
# Example script that merges the lysozyme data set with
# 1) A normally distributed error model and 
# 2) A T-distributed error model

mkdir -p merge/normal

careless mono \
    --sequential-layers=20 \
    --anomalous \
    --iterations=10000 \
    "BATCH,dHKL,Hobs,Kobs,Lobs,XDET,YDET,BG,SIGBG,LP,QE,FRACTIONCALC" \
    unmerged.mtz \
    merge/normal/hewl


mkdir -p merge/studentt

careless mono \
    --sequential-layers=20 \
    --studentt-likelihood-dof=16 \
    --anomalous \
    --iterations=10000 \
    "BATCH,dHKL,Hobs,Kobs,Lobs,XDET,YDET,BG,SIGBG,LP,QE,FRACTIONCALC" \
    unmerged.mtz \
    merge/studentt/hewl

