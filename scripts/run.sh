#!/bin/bash

#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --mem=15GB
#SBATCH --job-name=fnc
#SBATCH --mail-type=END
#SBATCH --mail-user=tosik@nyu.edu
#SBATCH --output=fnc.log.txt

source activate fnc

python scripts/run.py competition_test.yml
