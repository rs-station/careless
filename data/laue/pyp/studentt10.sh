#!/bin/bash
#
#
#SBATCH -p seas_gpu_requeue # partition (queue)
#SBATCH --mem 12G # memory pool for all cores
#SBATCH -t 0-0:45 # time (D-HH:MM)
#SBATCH --gres=gpu:1
#SBATCH --constraint=v100

OUT=studentt10
mkdir $OUT
careless poly \
  --separate-files \
  --iterations=30000 \
  --learning-rate=0.01 \
  --studentt-likelihood-dof=10. \
  --wavelength-key='Wavelength' \
  "X,Y,Wavelength,BATCH,dHKL,file_id" \
  off_varEll.mtz \
  2ms_varEll.mtz \
  $OUT/pyp

