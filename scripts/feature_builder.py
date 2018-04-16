#!/usr/bin/env python3

import argparse
import pickle
import sys
import yaml

import pandas as pd
import xgboost as xgb

from scipy.spatial.distance import hamming
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


def build_features(split):
    """
    Build features for train|test split
    """
    if split not in ("train", "test"):
        print("Error: {0} is not a valid split, use train|test".format(split))
        sys.exit(1)

    # Load data sets
    df_stances = pd.read_csv(config["{0}_stances".format(split)])
    df_bodies = pd.read_csv(config["{0}_bodies".format(split)])

    # Merge stances and bodies
    df = pd.merge(df_stances, df_bodies, on="Body ID", how="left")

    # Train or load vectorizer
    if split == "train":
        # Initialize binary count vectorizer, filter stop words
        vectorizer = CountVectorizer(
            binary=True,
            stop_words="english",
        )
        # Fit count vectorizer to all stance and body text
        vectorizer.fit(df["Headline"].tolist() + df["articleBody"].tolist())
        # Write vectorizer object to serialized file
        pickle.dump(vectorizer, open(config["vectorizer"], "wb"))
    else:
        # Load vectorizer object from serialized file
        vectorizer = pickle.load(open(config["vectorizer"], "rb"))

    # Transform data
    data = []
    for idx, row in df.iterrows():

        stance = vectorizer.transform([row["Headline"]])
        body = vectorizer.transform([row["articleBody"]])

        assert(stance.shape == body.shape)

        # Get Hamming distance between stance and body vectors
        dist = hamming(stance.toarray(), body.toarray())
        data.append((dist, row["Stance"]))

    # Write training data to feature file
    df = pd.DataFrame(data, columns=["hamming_distance", "label"])
    df.to_csv(config["{0}_feats".format(split)], index=False)

    return df


def train_and_test(train_df, test_df, features):
    """
    Train and test XGBoost classifier
    """
    # Classifier input data
    X_train = train_df.as_matrix(columns=features)
    y_train = le.transform(train_df["label"].values)
    X_test = test_df.as_matrix(columns=features)
    y_test = le.transform(test_df["label"].values)

    assert(len(X_train) == len(y_train))
    assert(len(X_test) == len(y_test))

    # Train XGBoost classifier
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # Parameters
    param = {
        "objective": "multi:softprob",
        "num_class": 4,
        "silent": 1,
    }
    num_round = 20  # Set number of training iterations

    # Train model, dump feature map, and save model to file
    bst = xgb.train(param, dtrain, num_round)
    bst.dump_model(config["model_dump"])
    # bst.save_model("0001.model")

    # Get prediction probabilities per class
    pred = bst.predict(dtest)

    # Get accuracy score on test
    y_pred = pred.argmax(axis=1)
    print(accuracy_score(y_test, y_pred))

    # # Decode prediction probabilities to string labels
    # y_pred_labels = le.classes_[y_pred]
    # print(y_pred_labels)


def main():
    """
    main()
    """
    parser = argparse.ArgumentParser(description="Feature builder")
    parser.add_argument(
        "config_file",
        metavar="config_file",
        help=".yml configuration file")
    args = parser.parse_args()

    # Load .yml file
    with open(args.config_file, "r") as config_file:
        global config
        config = yaml.load(config_file)

    # Build features
    # NB: Comment this out if feature files don't need to be regenerated
    for split in ("train", "test"):
        build_features(split)

    # Load feature files
    train_df = pd.read_csv(config["{0}_feats".format("train")])
    test_df = pd.read_csv(config["{0}_feats".format("test")])

    # Encode string labels
    global le
    le = LabelEncoder()
    le.fit(["unrelated", "agree", "disagree", "discuss"])

    # List of features to use
    features = [
        "hamming_distance",
    ]

    # Train and test classifier
    train_and_test(train_df, test_df, features)


if __name__ == "__main__":
    main()
