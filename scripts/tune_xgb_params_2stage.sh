#!/bin/bash

#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --mem=50GB
#SBATCH --job-name=fnc
#SBATCH --mail-type=END
#SBATCH --mail-user=tosik@nyu.edu
#SBATCH --output=xgb2.log.txt

source activate fnc

python scripts/tune_xgb_params_2stage.py competition_test.yml

