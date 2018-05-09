#!/bin/bash

#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --mem=50GB
#SBATCH --job-name=lda
#SBATCH --mail-type=END
#SBATCH --mail-user=tosik@nyu.edu
#SBATCH --output=lda.log.txt

source activate fnc

python scripts/train_lda_model.py competition_test.yml
