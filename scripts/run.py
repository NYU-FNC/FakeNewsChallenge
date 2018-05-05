#!/usr/bin/env python3

import argparse
import yaml

import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.preprocessing import LabelEncoder

from feature_builder import build_features
from sklearn.metrics import accuracy_score


def init_config():
    """
    Initialize config file
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_file",
        metavar="config_file",
        help=".yml configuration file")
    args = parser.parse_args()

    # Load .yml file
    with open(args.config_file, "r") as config_file:
        global config
        config = yaml.load(config_file)
    return config


def train(stage):
    """
    Training
    """
    # Load feature file and replace labels
    train_df = pd.read_csv(config["train_feats"])

    if stage == 1:
        train_df["label"] = \
            train_df["label"].replace(["agree", "disagree", "discuss"], "related")
    if stage == 2:
        train_df = train_df[train_df.label != "unrelated"]

    # Split features and label
    X_train, y_train = np.split(train_df, [-1], axis=1)

    # Get labels
    if stage == 1:
        labels = ["unrelated", "related"]
    if stage == 2:
        labels = ["agree", "disagree", "discuss"]

    # Encode labels
    le = LabelEncoder()
    le.fit(labels)
    y_train = le.transform(y_train.as_matrix())

    # Build DMatrix
    dtrain = xgb.DMatrix(X_train.as_matrix(), label=y_train)

    # Parameters
    param = {
        "objective": "multi:softprob",
        "num_class": 4,
    }
    num_round = 20

    # Train model, dump feature map, and save model to file
    model = xgb.train(param, dtrain, num_round, verbose_eval=config["verbose"])

    return model


def main():

    # Load config
    config = init_config()

    # Build features
    for split in ("train", "test"):
        build_features(split, config)

    # Train models
    model_1 = train(stage=1)
    model_2 = train(stage=2)

    """
    Stage 1
    """
    test_df = pd.read_csv(config["test_feats"])

    # Save original labels
    original_labels = test_df.iloc[:, -1:]

    # Transform test data
    test_df["label"] = test_df["label"].replace(["agree", "disagree", "discuss"], "related")
    X_test, y_test = np.split(test_df, [-1], axis=1)

    # Encode labels
    le = LabelEncoder()
    le.fit(["unrelated", "related"])

    # Transform and convert to DMatrix
    y_test = le.transform(y_test.as_matrix())
    dtest = xgb.DMatrix(X_test.as_matrix(), label=y_test)

    # Get predictions
    pred_1 = model_1.predict(dtest).argmax(axis=1)
    print(accuracy_score(y_test, pred_1))

    """
    Stage 2
    """
    test_df = pd.read_csv(config["test_feats"])
    X_test, y_test = np.split(test_df, [-1], axis=1)

    # Add stage 1 labels
    X_test["label_1"] = pred_1

    # ?
    X_test = X_test[X_test.label_1 != 0]
    X_test = X_test.drop(["label_1"], axis=1)

    # Encode labels
    le = LabelEncoder()
    le.fit(["agree", "disagree", "discuss"])
    print(le.classes_)

    # Convert to DMatrix
    dtest = xgb.DMatrix(X_test.as_matrix())

    # Get predictions
    pred_2 = model_2.predict(dtest).argmax(axis=1)

    """
    Combine stage 1 and 2
    """
    pred = []
    i = 0
    for idx, y in enumerate(pred_1):
        if y == 0:
            pred.append(pred_2[i])
            i += 1
        else:
            pred.append(3)

    label_map = {
        0: "agree",
        1: "disagree",
        2: "discuss",
        3: "unrelated",
    }

    labels = []
    for x in pred:
        labels.append(label_map[x])

    # Load unlabeled test set
    df = pd.read_csv(config["test_unlabeled"])

    # Write output file
    df["Stance"] = labels
    df.to_csv("predictions.csv", index=False)


if __name__ == '__main__':
    main()
