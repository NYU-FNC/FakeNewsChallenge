#!/usr/bin/env python3

import argparse
import yaml

import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.preprocessing import LabelEncoder

from score import report_score
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


def phase_one_training():
    """
    Training, phase 1
    """
    # Load feature file and replace labels
    train_df = pd.read_csv(config["{0}_feats_phase_one".format("train")])
    train_df['label'] = train_df['label'].replace(["agree", "disagree", "discuss"], "related")

    # Classifier input data
    X_train, y_train = np.split(train_df, [-1], axis=1)

    # Encode string labels
    le = LabelEncoder()
    le.fit(["unrelated", "related"])
    y_train = le.transform(y_train.as_matrix())

    # Get training DMatrix
    dtrain = xgb.DMatrix(X_train.as_matrix(), label=y_train)

    # Parameters
    param = {
        "objective": "multi:softprob",
        "num_class": 4,
    }
    num_round = 20  # Set number of training iterations

    # Train model, dump feature map, and save model to file
    model_ru = xgb.train(param, dtrain, num_round, verbose_eval=config["verbose"])
    model_ru.dump_model(config["model_dump"])

    return model_ru


def phase_two_training():
    """
    Training, phase 2
    """
    # Load feature file and replace labels
    train_df = pd.read_csv(config["{0}_feats_phase_two".format("train")])
    train_df = train_df[train_df.label != "unrelated"]

    # Classifier input data
    X_train, y_train = np.split(train_df, [-1], axis=1)

    # # Encode string labels
    le = LabelEncoder()
    le.fit(["agree", "disagree", "discuss"])
    y_train = le.transform(y_train.as_matrix())

    # Get training DMatrix
    dtrain = xgb.DMatrix(X_train.as_matrix(), label=y_train)

    # Parameters
    param = {
        "objective": "multi:softprob",
        "num_class": 4,
    }
    num_round = 20

    # Train model, dump feature map, and save model to file
    model_add = xgb.train(param, dtrain, num_round, verbose_eval=config["verbose"])
    model_add.dump_model(config["model_dump"])
    return model_add


def main():
    """
    main()
    """
    # Init config file
    config = init_config()

    # Build features and train model, phase 1
    for split in ("train", "test"):
        build_features(split, "phase_one", config)
    model_ru = phase_one_training()

    # Build features and train model, phase 2
    for split in ("train", "test"):
        build_features(split, "phase_two", config)
    model_add = phase_two_training()

    """
    Test, phase 1
    """
    test_df = pd.read_csv(config["{0}_feats_phase_one".format("test")])

    # Save original labels
    original_labels = test_df.iloc[:, -1:]

    # Transform test labels
    test_df['label'] = test_df['label'].replace(["agree", "disagree", "discuss"], "related")

    # Split test features and labels
    X_test, y_test = np.split(test_df, [-1], axis=1)

    # Encode string labels
    le = LabelEncoder()
    le.fit(["unrelated", "related"])

    # Transform test data
    y_test = le.transform(y_test.as_matrix())
    dtest = xgb.DMatrix(X_test.as_matrix(), label=y_test)

    # Get prediction probabilities per class
    pred_ru = model_ru.predict(dtest).argmax(axis=1)

    print(accuracy_score(y_test, pred_ru))

    """
    Test, phase 2
    """
    test_df = pd.read_csv(config["{0}_feats_phase_two".format("test")])

    X_test, y_test = np.split(test_df, [-1], axis=1)

    X_test['label_ru'] = pred_ru

    X_test = X_test[X_test.label_ru != 0]
    X_test = X_test.drop(['label_ru'], axis=1)

    le = LabelEncoder()
    le.fit(["agree", "disagree", "discuss"])

    dtest = xgb.DMatrix(X_test.as_matrix())
    pred = model_add.predict(dtest)

    y_pred = pred.argmax(axis=1)

    final = []
    i = 0
    for idx, y in enumerate(pred_ru):
        if y == 0:
            final.append(y_pred[i])
            i += 1
        else:
            final.append(3)

    le = LabelEncoder()
    le.fit(["agree", "disagree", "discuss", "unrelated"])
    original_labels = le.transform(original_labels.as_matrix())

    report_score(original_labels, final)


if __name__ == '__main__':
    main()
