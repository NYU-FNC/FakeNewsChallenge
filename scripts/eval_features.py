#!/usr/bin/env python3

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.preprocessing import LabelEncoder

from utils import load_config

labels = ["agree", "disagree", "discuss", "unrelated"]

# Encode labels
le = LabelEncoder()
le.fit(labels)

feats = [
    # "cosine_count",
    "cosine_tdidf",
    # "dep_object_overlap",
    # "dep_subject_overlap",
    "doc_similarity",
    "doc_similarity_intro",
    # "hamming_distance",
    "len_stance",
    "len_body",
    "ngram_overlap",
    "ngram_overlap_intro",
    # "KL_pk_qk",
    # "KL_qk_pk",
    "refute",
    "refute_intro",
    # "sentiment_body",
    # "sentiment_stance",
    "word_overlap",
    "word_overlap_intro",
    "wmdistance",
]


def drop_feats(df):
    """
    Drop features
    """
    cols = list(df)
    for col in cols:
        if col not in feats and col != "label":
            df = df.drop(columns=col)
    return df


def train():
    """
    Training
    """
    # Load feature file and replace labels
    train_df = pd.read_csv(config["train_feats"])
    train_df = drop_feats(train_df)

    # Split features and label
    X_train, y_train = np.split(train_df, [-1], axis=1)
    y_train = le.transform(y_train.as_matrix())

    # Build DMatrix
    dtrain = xgb.DMatrix(X_train.as_matrix(), label=y_train, silent=True)

    params = {
        "learning_rate": 0.015,
        "n_estimators": 200,
        "max_depth": 9,
        "min_child_weight": 1,
        "gamma": 0.0,
        "subsample": 0.6,
        "colsample_bytree": 0.8,
        "scale_pos_weight": 1,
        "objective": "multi:softprob",
        "num_class": 4,
    }

    # Train model, dump feature map, and save model to file
    model = xgb.train(params, dtrain, 20, verbose_eval=0)

    return model


def main(buildfeats=False):
    """
    Evaluate feature sets for 1-stage classifier
    """
    # Load config
    global config
    config = load_config()

    # Train model
    model = train()

    test_df = pd.read_csv(config["test_feats"])
    test_df = drop_feats(test_df)

    X_test, y_test = np.split(test_df, [-1], axis=1)

    # Transform and convert to DMatrix
    y_test = le.transform(y_test.as_matrix())
    dtest = xgb.DMatrix(X_test.as_matrix(), label=y_test, silent=True)

    # Get predictions
    pred = model.predict(dtest).argmax(axis=1)
    pred_labels = le.inverse_transform(pred)

    # Load unlabeled test set
    df = pd.read_csv(config["test_unlabeled"])

    # Write output file
    assert(len(df) == len(pred_labels))
    df["Stance"] = pred_labels
    df.to_csv("predictions_eval.{0}.csv".format("-".join([f[:5] for f in feats])), index=False)


if __name__ == '__main__':
    main()
