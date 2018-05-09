#!/bin/bash

#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --mem=50GB
#SBATCH --job-name=fnc
#SBATCH --mail-type=END
#SBATCH --mail-user=tosik@nyu.edu
#SBATCH --output=xgb1.log.txt

source activate fnc

python scripts/tune_xgb_params_1stage.py competition_test.yml

