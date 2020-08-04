#!/bin/bash
#
#
#SBATCH -p seas_gpu_requeue # partition (queue)
#SBATCH --mem 12G # memory pool for all cores
#SBATCH -t 0-0:45 # time (D-HH:MM)
#SBATCH --gres=gpu:1
#SBATCH --constraint=v100

mkdir merge
careless poly \
  --merge-files=False \
  --iterations=30000 \
  --learning-rate=0.001 \
  --wavelength-key='Wavelength' \
  "X,Y,Wavelength,Hobs,Kobs,Lobs,image_id" \
  off_varEll.mtz \
  2ms_varEll.mtz \
  merge/pyp

