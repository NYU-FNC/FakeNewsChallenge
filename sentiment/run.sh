#!/bin/bash

#SBATCH --cpus-per-task=4
#SBATCH --time=144:00:00
#SBATCH --mem=40GB
#SBATCH --job-name=sentiment
#SBATCH --mail-type=END
#SBATCH --mail-user=tosik@nyu.edu
#SBATCH --output=sentiment.log.txt

export CLASSPATH=stanford-corenlp-full-2018-02-27/*:.

javac SentimentAnnotator.java && java SentimentAnnotator

