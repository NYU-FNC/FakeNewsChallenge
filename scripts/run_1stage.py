#!/usr/bin/env python3

import numpy as np
import pandas as pd
import xgboost as xgb

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
    Resample training data
    """
    # Separate majority and minority classes
    df_agree = df[df.label == "agree"]
    df_disagree = df[df.label == "disagree"]
    df_discuss = df[df.label == "discuss"]
    df_unrelated = df[df.label == "unrelated"]

    # Upsample "disagree"
    n = round(len(df_agree) / 2)
    df_disagree_up = resample(df_disagree, replace=True, n_samples=n)

    # Combine original and resampled data
    df_up = pd.concat([
        df_agree,
        df_disagree_up,
        df_discuss,
        df_unrelated,
    ])
    print(df_up.label.value_counts())
    return df_up


def train():
    """
    Train XGBoost classifier
    """
    # Load and select training features
    train_df = pd.read_csv(config["train_feats"])
    train_df = select_feats(train_df)

    # Resample training data
    train_df = resample_train(train_df)

    # Split features and labels
    X_train, y_train = np.split(train_df, [-1], axis=1)
    # Encode labels
    y_train = le.transform(y_train.as_matrix())
    # Build DMatrix
    dtrain = xgb.DMatrix(X_train.as_matrix(), label=y_train)

    # Hyperparameter settings
    params = {
        "learning_rate": 0.02,
        "n_estimators": 800,
        "max_depth": 7,
        "min_child_weight": 1,
        "gamma": 0.0,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "scale_pos_weight": 1,
        "objective": "multi:softprob",
        "num_class": 4,
    }

    # Train model
    model = xgb.train(params, dtrain, 20)
    return model


def main(rebuild_features=False):
    """
    Run 1-stage classifier
    """
    global config
    config = load_config()

    global le
    le = LabelEncoder()
    # Fit labels
    le.fit(["agree", "disagree", "discuss", "unrelated"])

    # Build features
    if rebuild_features:
        for split in ("train", "test"):
            build_features(split, config)

    # Train model
    model = train()

    # Load and select testing features
    test_df = pd.read_csv(config["test_feats"])
    test_df = select_feats(test_df)

    # Split features and labels
    X_test, y_test = np.split(test_df, [-1], axis=1)
    # Encode labels
    y_test = le.transform(y_test.as_matrix())
    # Build DMatrix
    dtest = xgb.DMatrix(X_test.as_matrix(), label=y_test)

    # Get predictions
    pred = model.predict(dtest).argmax(axis=1)
    # Decode labels
    pred_labels = le.inverse_transform(pred)

    # Load unlabeled test set
    df = pd.read_csv(config["test_unlabeled"])

    # Write output file
    assert(len(df) == len(pred_labels))
    df["Stance"] = pred_labels
    df.to_csv("predictions.1stage.csv", index=False)


if __name__ == "__main__":
    main()
