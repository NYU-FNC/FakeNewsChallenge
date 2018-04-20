import pickle
import sys

import pandas as pd

import spacy

from textblob import TextBlob
from scipy.spatial.distance import hamming
from sklearn.feature_extraction.text import CountVectorizer

nlp = spacy.load('en')

def build_features(split, config):
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
        stance_polarity = TextBlob(row["Headline"]).sentiment.polarity
        body_polarity = TextBlob(row["articleBody"]).sentiment.polarity
        similarity = nlp(row["Headline"]).similarity(nlp(row["articleBody"]))

        data.append((dist, stance_polarity, body_polarity, similarity, row["Stance"]))

    # Write training data to feature file
    df = pd.DataFrame(
        data, columns=["hamming_distance", "stance_polarity", "body_polarity", "similarity", "label"])
    df.to_csv(config["{0}_feats".format(split)], index=False)

    return df



