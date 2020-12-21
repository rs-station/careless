#!/bin/bash
#
#
#SBATCH -p seas_gpu_requeue # partition (queue)
#SBATCH --mem 32G # memory pool for all cores
#SBATCH -t 0-4:00 # time (D-HH:MM)
#SBATCH --gres=gpu:1
#SBATCH --constraint=v100


out=test
basename=hewl

mkdir -p $out
cp $0 $out
cat $0 

careless mono \
    --sequential-layers=20 \
    --wilson-prior-b=20 \
    --studentt-likelihood-dof=16 \
    --anomalous \
    --iterations=10000 \
    --learning-rate=0.001 \
    "BATCH,dHKL,Hobs,Kobs,Lobs,XDET,YDET,BG,SIGBG,LP,QE,FRACTIONCALC" \
    integrated_pass1.mtz \
    $out/$basename 

