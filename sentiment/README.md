# Sentiment scores

We use [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/) to obtain sentiment scores for each stance and article body in the FNC-1 dataset.

## Download

```
(fnc) [sentiment]$ wget http://nlp.stanford.edu/software/stanford-corenlp-full-2018-02-27.zip
(fnc) [sentiment]$ unzip stanford-corenlp-full-2018-02-27.zip
```

## Run

First, use the Python script to convert the original .csv data files to one-document-per-line .txt files:

```
(fnc) [sentiment]$ python prep_sentiment.py
```

Next, use the Java program in `SentimentAnnotator.java` to obtain a sentiment score for each document in the data .txt files:

```
(fnc) [sentiment]$ export CLASSPATH=stanford-corenlp-full-2018-02-27/*:.
(fnc) [sentiment]$ javac SentimentAnnotator.java && java SentimentAnnotator
```

Alternatively, use `sbatch` to run the bash script on [HPC](https://wikis.nyu.edu/display/NYUHPC/High+Performance+Computing+at+NYU):

```
(fnc) [sentiment]$ sbatch run.sh
```

If there are multiple sentences in a given document, the output sentiment score will be the average sentiment score of all the sentences.
