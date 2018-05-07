#!/usr/bin/env python3

import numpy as np
import pandas as pd
import warnings
import xgboost as xgb

from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample

from utils import load_config


def warn(*args, **kwargs):
    pass


warnings.warn = warn
warnings.simplefilter("ignore", DeprecationWarning)


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


def grid_search_params(stage):
    """
    Training
    """
    print("--- STAGE: {0} ---\n".format(stage))
    best = {}

    # Load feature file and replace labels
    train_df = pd.read_csv(config["train_feats"])

    if stage == 1:
        train_df["label"] = \
            train_df["label"].replace(["agree", "disagree", "discuss"], "related")
        le.fit(["unrelated", "related"])
        num_class = 2

    if stage == 2:
        train_df = train_df[train_df.label != "unrelated"]
        le.fit(["agree", "disagree", "discuss"])
        num_class = 3

    # Split features and labels
    X_train, y_train = np.split(train_df, [-1], axis=1)
    # Encode labels
    y_train = le.fit_transform(y_train)

    """
    Following https://www.kaggle.com/saxinou/imbalanced-data-xgboost-tunning
    """
    print("Initialize...")
    model = xgb.XGBClassifier(
        learning_rate=0.1,
        n_estimators=100,
        max_depth=5,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=1,
        objective="multi:softprob",
        num_class=num_class,
    )
    model.fit(X_train.as_matrix(), y_train)
    auc = accuracy_score(y_train, model.predict(X_train.as_matrix()))
    print("Model accuracy:", auc, "\n")

    # Test 1
    print("--- PARAMETER GRID 1 ---\n")
    param_grid_1 = {
        "max_depth": range(3, 10, 2),
        "min_child_weight": range(1, 6, 2),
        "n_estimators": [100, 200, 500, 800],
    }
    grid_search_1 = GridSearchCV(model, param_grid=param_grid_1, verbose=1, n_jobs=-1)
    grid_search_1.fit(X_train.as_matrix(), y_train)
    print(grid_search_1.best_params_, "\n")

    # Add to best parameter map
    for key in grid_search_1.best_params_:
        best[key] = grid_search_1.best_params_[key]

    # Re-initialize with new parameters
    print("Re-initialize...")
    model = xgb.XGBClassifier(
        learning_rate=0.1,
        n_estimators=grid_search_1.best_params_["n_estimators"],
        max_depth=grid_search_1.best_params_["max_depth"],
        min_child_weight=grid_search_1.best_params_["min_child_weight"],
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=1,
        objective="multi:softprob",
        num_class=num_class,
    )
    model.fit(X_train.as_matrix(), y_train)
    auc = accuracy_score(y_train, model.predict(X_train.as_matrix()))
    print("Model accuracy:", auc, "\n")

    # Test 2
    print("--- PARAMETER GRID 2 ---\n")
    param_grid_2 = {
        "gamma": [i / 10.0 for i in range(0, 5)]
    }
    grid_search_2 = GridSearchCV(model, param_grid=param_grid_2, verbose=1, n_jobs=-1)
    grid_search_2.fit(X_train.as_matrix(), y_train)
    print(grid_search_2.best_params_, "\n")

    # Add to best parameter map
    for key in grid_search_2.best_params_:
        best[key] = grid_search_2.best_params_[key]

    # Re-initialize with new parameters
    print("Re-initialize...")
    model = xgb.XGBClassifier(
        learning_rate=0.1,
        n_estimators=grid_search_1.best_params_["n_estimators"],
        max_depth=grid_search_1.best_params_["max_depth"],
        min_child_weight=grid_search_1.best_params_["min_child_weight"],
        gamma=grid_search_2.best_params_["gamma"],
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=1,
        objective="multi:softprob",
        num_class=num_class,
    )
    model.fit(X_train.as_matrix(), y_train)
    auc = accuracy_score(y_train, model.predict(X_train.as_matrix()))
    print("Model accuracy:", auc, "\n")

    # Test 3
    print("--- PARAMETER GRID 3 ---\n")
    param_grid_3 = {
        "subsample": [i / 10.0 for i in range(6, 10)],
        "colsample_bytree": [i / 10.0 for i in range(6, 10)]
    }
    grid_search_3 = GridSearchCV(model, param_grid=param_grid_3, verbose=1, n_jobs=-1)
    grid_search_3.fit(X_train.as_matrix(), y_train)
    print(grid_search_3.best_params_, "\n")

    # Add to best parameter map
    for key in grid_search_3.best_params_:
        best[key] = grid_search_3.best_params_[key]

    # Re-initialize with new parameters
    print("Re-initialize...")
    model = xgb.XGBClassifier(
        learning_rate=0.1,
        n_estimators=grid_search_1.best_params_["n_estimators"],
        max_depth=grid_search_1.best_params_["max_depth"],
        min_child_weight=grid_search_1.best_params_["min_child_weight"],
        gamma=grid_search_2.best_params_["gamma"],
        subsample=grid_search_3.best_params_["subsample"],
        colsample_bytree=grid_search_3.best_params_["colsample_bytree"],
        scale_pos_weight=1,
        objective="multi:softprob",
        num_class=num_class,
    )
    model.fit(X_train.as_matrix(), y_train)
    auc = accuracy_score(y_train, model.predict(X_train.as_matrix()))
    print("Model accuracy:", auc, "\n")

    # Test 4
    print("--- PARAMETER GRID 4 ---\n")
    param_grid_4 = {
        "learning_rate": [i / 1000.0 for i in range(5, 20, 2)]
    }
    grid_search_4 = GridSearchCV(model, param_grid=param_grid_4, verbose=1, n_jobs=-1)
    grid_search_4.fit(X_train.as_matrix(), y_train)
    print(grid_search_4.best_params_, "\n")

    # Add to best parameter map
    for key in grid_search_4.best_params_:
        best[key] = grid_search_4.best_params_[key]

    # Re-initialize with new parameters
    print("Re-initialize...")
    model = xgb.XGBClassifier(
        learning_rate=grid_search_4.best_params_["learning_rate"],
        n_estimators=grid_search_1.best_params_["n_estimators"],
        max_depth=grid_search_1.best_params_["max_depth"],
        min_child_weight=grid_search_1.best_params_["min_child_weight"],
        gamma=grid_search_2.best_params_["gamma"],
        subsample=grid_search_3.best_params_["subsample"],
        colsample_bytree=grid_search_3.best_params_["colsample_bytree"],
        scale_pos_weight=1,
        objective="multi:softprob",
        num_class=num_class,
    )
    model.fit(X_train.as_matrix(), y_train)
    auc = accuracy_score(y_train, model.predict(X_train.as_matrix()))
    print("Model accuracy:", auc, "\n")

    print("--- BEST PARAMETERS ---")
    print(best)


def main():
    """
    XGBoost parameter tuning
    """
    global config
    config = load_config()

    global le
    le = LabelEncoder()

    grid_search_params(stage=1)
    grid_search_params(stage=2)


if __name__ == '__main__':
    main()
