# Fake News Challenge (FNC-1)

This [GitHub repository](https://github.com/NYU-FNC/FakeNewsChallenge) contains all of the code and data for our final NLP class project. Our [initial project proposal](https://github.com/NYU-FNC/FakeNewsChallenge/blob/master/PROPOSAL.md) is available as well.

## Data

The original dataset and scorer script are located in the `fnc-1/` directory. All additional data files can be found in the `data/` directory, including the following:

- Generated features files for train and test: `feats.train.all.csv` and `feats.competition_test.all.csv`
- Binary file containing 300-dimensional GloVe embeddings: `GoogleNews-vectors-negative300.bin`
- Serialized LDA topic model and dictionary: `lda.model.pkl` and `lda.dct.pkl`
- "The New York Times Newswire Service" portion of the [English Gigaword](https://catalog.ldc.upenn.edu/ldc2003t05) corpus: `nyt.txt`
- Subdirectory for sentiment analysis: `sentiment/`
- Serialized count and TF-IDF vectorizer models: `vectorizer.count.pkl` and `vectorizer.tfidf.pkl`

## Scripts

All Python and bash scripts are included in the `scripts/` directory. To run, create a `conda` environment and install the requirements:

```
[FakeNewsChallenge]$ conda create -n fnc-nyu python=3.6
[FakeNewsChallenge]$ source activate fnc-nyu
(fnc-nyu) [FakeNewsChallenge]$ pip install -r requirements.txt
```

### Python scripts

Once your environment is set up, you can run the following scripts:

- `feature_builder.py`: feature engineering module
- `run_1stage.py`: train and run 1-stage classifier
- `run_2stage.py`: train and run 2-stage classifier
- `scorer.py`: original evaluation script published by the FNC-1 organizers
- `train_lda_model.py`: train LDA topic model on the NYT portion of Gigaword
- `tune_xgb_params_1stage.py`: tune hyperparameters for 1-stage classifier
- `tune_xgb_params_2stage.py`: tune hyperparameters for 2-stage classifier
- `utils.py`: utilty functions (configuration file import and text preprocessing)

Note that most of the scripts require that you pass a **configuration file** as argument:

```
(fnc-nyu) [FakeNewsChallenge]$ python run_1stage.py competition_test.yml
```

Most importantly, `competition_test.yml` contains the list of features that will be included or generated on each run. 
### Bash scripts

Alternatively, use `sbatch` and the bash scripts below to run extensive Python jobs on the [high performance computing (HPC)](https://wikis.nyu.edu/display/NYUHPC/High+Performance+Computing+at+NYU) cluster:

- `run_1stage.sh`
- `run_2stage.sh`
- `train_lda_model.sh`
- `tune_xgb_params_1stage.sh`
- `tune_xgb_params_2stage.sh`

## Predictions and results

There are two output files containing the predictions on the competition test set:

- `predictions.1stage.csv`: contains the 1-stage classifier predictions
- `predictions.2stage.csv`: contains the 2-stage classifier predictions 

### 1-stage classifier

The top score after feature selection and hyperparameter tuning for the 1-stage classifier was 9128.5, or **78.35%**.

```
(fnc) [mt3685@c38-15 FakeNewsChallenge]$ python scripts/scorer.py fnc-1/competition_test_stances.csv predictions.1stage.csv
CONFUSION MATRIX:
-------------------------------------------------------------
|           |   agree   | disagree  |  discuss  | unrelated |
-------------------------------------------------------------
|   agree   |    144    |     4     |   1607    |    148    |
-------------------------------------------------------------
| disagree  |    12     |     1     |    522    |    162    |
-------------------------------------------------------------
|  discuss  |    190    |     2     |   3874    |    398    |
-------------------------------------------------------------
| unrelated |     2     |     0     |    246    |   18101   |
-------------------------------------------------------------
ACCURACY: 0.870

MAX  - the best possible score (100% accuracy)
NULL - score as if all predicted stances were unrelated
TEST - score based on the provided predictions

||    MAX    ||    NULL   ||    TEST   ||
|| 11651.25  ||  4587.25  ||  9128.5   ||
```

### 2-stage classifier

The top score after feature selection and hyperparameter tuning for the 2-stage classifier was 9161.5, or **78.63%**.

```
(fnc) [mt3685@c38-15 FakeNewsChallenge]$ python scripts/scorer.py fnc-1/competition_test_stances.csv predictions.2stage.csv
CONFUSION MATRIX:
-------------------------------------------------------------
|           |   agree   | disagree  |  discuss  | unrelated |
-------------------------------------------------------------
|   agree   |    27     |     0     |   1733    |    143    |
-------------------------------------------------------------
| disagree  |     9     |     0     |    533    |    155    |
-------------------------------------------------------------
|  discuss  |    45     |     0     |   4060    |    359    |
-------------------------------------------------------------
| unrelated |     5     |     0     |    366    |   17978   |
-------------------------------------------------------------
ACCURACY: 0.868

MAX  - the best possible score (100% accuracy)
NULL - score as if all predicted stances were unrelated
TEST - score based on the provided predictions

||    MAX    ||    NULL   ||    TEST   ||
|| 11651.25  ||  4587.25  ||  9161.5   ||

(fnc) [mt3685@c38-15 FakeNewsChallenge]$
```
