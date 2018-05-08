#!/usr/bin/env python3

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample

from feature_builder import build_features
from utils import load_config


def select_feats(df):
    """
    Select features based on configuration
    """
    cols = list(df)
    for col in cols:
        if col not in config["feats"] and col != "label":
            df = df.drop(columns=col)
    return df


def resample_train(df):
    """
    Resample training data (stage 2 only)
    """
    # Separate classes
    df_agree = df[df.label == "agree"]
    df_disagree = df[df.label == "disagree"]
    df_discuss = df[df.label == "discuss"]
    # df_unrelated = df[df.label == "unrelated"]

    # Upsample "disagree"
    n = round(len(df_agree) / 2)
    df_disagree_up = resample(df_disagree, replace=True, n_samples=n)

    # Combine original and resampled data
    df_up = pd.concat([
        df_agree,
        df_disagree_up,
        df_discuss,
        # df_unrelated,
    ])
    print(df_up.label.value_counts())
    return df_up


def train(stage):
    """
    Training XGBoost classifier
    """
    # Load and select training features
    train_df = pd.read_csv(config["train_feats"])
    train_df = select_feats(train_df)

    if stage == 1:

        # Replace labels
        train_df["label"] = \
            train_df["label"].replace(["agree", "disagree", "discuss"], "related")
        # Fit labels
        labels = ["unrelated", "related"]
        le.fit(labels)

        # Hyperparameter settings
        params = {
            "learning_rate": 0.019,
            "n_estimators": 1000,
            "max_depth": 5,
            "min_child_weight": 1,
            "gamma": 0.3,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": 1,
            "objective": "multi:softprob",
            "num_class": len(labels),
        }

    if stage == 2:

        # Resample training data
        #train_df = resample_train(train_df)

        # Replace labels
        train_df = train_df[train_df.label != "unrelated"]
        # Fit labels
        labels = ["agree", "disagree", "discuss"]
        le.fit(labels)

        # Hyperparameter settings
        params = {
            "learning_rate": 0.019,
            "n_estimators": 1000,
            "max_depth": 7,
            "min_child_weight": 1,
            "gamma": 0.2,
            "subsample": 0.9,
            "colsample_bytree": 0.7,
            "scale_pos_weight": 1,
            "objective": "multi:softprob",
            "num_class": len(labels),
        }

    # Split features and labels
    X_train, y_train = np.split(train_df, [-1], axis=1)
    # Encode labels
    y_train = le.transform(y_train.as_matrix())
    # Build DMatrix
    dtrain = xgb.DMatrix(X_train.as_matrix(), label=y_train)

    # Train model
    model = xgb.train(params, dtrain, 20)
    return model


def main(rebuild_features=False):
    """
    Run 2-stage classifier
    """
    # Load config
    global config
    config = load_config()

    # Encode labels
    global le
    le = LabelEncoder()

    # Build features
    if rebuild_features:
        for split in ("train", "test"):
            build_features(split, config)

    # Train models
    model_1 = train(stage=1)
    model_2 = train(stage=2)

    """
    Stage 1
    """
    # Load and select testing features
    test_df = pd.read_csv(config["test_feats"])
    test_df = select_feats(test_df)

    # Replace labels
    test_df["label"] = test_df["label"].replace(["agree", "disagree", "discuss"], "related")
    # Split features and labels
    X_test, y_test = np.split(test_df, [-1], axis=1)
    # Fit labels
    le.fit(["unrelated", "related"])
    # Encode labels
    y_test = le.transform(y_test.as_matrix())
    # Build DMatrix
    dtest = xgb.DMatrix(X_test.as_matrix(), label=y_test)

    # Get predictions
    pred_1 = model_1.predict(dtest).argmax(axis=1)
    print(accuracy_score(y_test, pred_1))

    """
    Stage 2
    """
    # Load and select testing features
    test_df = pd.read_csv(config["test_feats"])
    test_df = select_feats(test_df)

    # Split features and labels
    X_test, y_test = np.split(test_df, [-1], axis=1)
    # Add stage 1 labels
    X_test["label_1"] = pred_1

    # Drop "unrelated" data points
    X_test = X_test[X_test.label_1 != 0]
    X_test = X_test.drop(columns="label_1")

    # Fit labels
    le.fit(["agree", "disagree", "discuss"])

    # Build DMatrix
    dtest = xgb.DMatrix(X_test.as_matrix())

    # Get predictions
    pred_2 = model_2.predict(dtest).argmax(axis=1)

    """
    Combine predictions
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

    # Decode labels
    pred_labels = []
    for x in pred:
        pred_labels.append(label_map[x])

    # Load unlabeled test set
    df = pd.read_csv(config["test_unlabeled"])

    # Write output file
    assert(len(df) == len(pred_labels))
    df["Stance"] = pred_labels
    df.to_csv("predictions.2stage.csv", index=False)


if __name__ == "__main__":
    main()
