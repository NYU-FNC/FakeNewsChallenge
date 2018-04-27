#!/usr/bin/env python3

"""
1. Download the CoreNLP models, unzip, place the folder in the root directory of the project
    http://nlp.stanford.edu/software/stanford-corenlp-full-2018-02-27.zip

2. Update the directory path in the .yml configuration file if necessary
    "corenlp_dir: ./stanford-corenlp-full-2018-02-27"

3. Run this script as root (e.g. using sudo), as detailed below
    https://github.com/Lynten/stanford-corenlp/issues/26
"""

import argparse
import yaml
import json

import pandas as pd

from os import listdir, path
from stanfordcorenlp import StanfordCoreNLP


def main():
    """
    Annotate sentiment scores using CoreNLP
    """
    parser = argparse.ArgumentParser(description="Sentiment")
    parser.add_argument(
        "config_file",
        metavar="config_file",
        help=".yml configuration file")
    args = parser.parse_args()

    # Load .yml file
    with open(args.config_file, "r") as config_file:
        global config
        config = yaml.load(config_file)

    # Initialize CoreNLP
    nlp = StanfordCoreNLP(config["corenlp_dir"], memory="8g")
    props = {
        "annotators": "sentiment",
        "pipelineLanguage": "en",
        "outputFormat": "json",
    }

    for file in listdir(config["data_dir"]):

        # Skip data files that are not CSV files
        if not file.endswith(".csv"):
            continue

        print("Annotating file \"{0}\"...".format(file))

        # Read dataframe
        df = pd.read_csv(path.join(config["data_dir"], file))

        # Get text column based on data input type
        for col in ("Headline", "articleBody"):
            if col in list(df):
                text_col = col

        # Skip in case text column does not exist
        if not text_col:
            print("No text column found, skipping file {0}".format(file))
            continue

        # Add new column for sentiment score
        df["sentiment_score"] = 0.0

        for idx, row in df.iterrows():

            # Annotate text
            r = nlp.annotate(
                row[text_col],
                properties=props,
            )

            avg_score = 0.0
            cnt = 0

            # Load CoreNLP result as JSON
            try:
                r_json = json.loads(r)
            except Exception:
                continue

            for sent in r_json:
                """
                sentimentValue":"1"
                sentiment":"Negative"
                """
                sentiment_score = r_json[sent][0]["sentimentValue"]
                avg_score += float(sentiment_score)
                cnt += 1

            # Compute average score
            avg_score = avg_score / cnt

            # Set sentiment score
            row["sentiment_score"] = avg_score

            if idx % 1000 == 0:
                print(idx)

        # Write output file
        df.to_csv(path.join(config["data_dir"], file[:-3] + "sentiment" + ".csv"))

    # This is important, otherwise CoreNLP will keep consuming memory
    nlp.close()


if __name__ == '__main__':
    main()
