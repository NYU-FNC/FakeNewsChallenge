#!/bin/bash

#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --mem=20GB
#SBATCH --job-name=fnc
#SBATCH --mail-type=END
#SBATCH --mail-user=tosik@nyu.edu
#SBATCH --output=fnc.log.txt

source activate fnc

python scripts/run_2stage.py competition_test.yml
python scripts/scorer.py fnc-1/competition_test_stances.csv predictions_2stage.csv
