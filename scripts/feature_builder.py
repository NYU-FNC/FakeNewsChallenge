import os
import pickle
import sys

import pandas as pd
import spacy

from scipy.spatial.distance import hamming
from sklearn.feature_extraction.text import CountVectorizer

# nlp = spacy.load("en")
nlp = spacy.load(
    "en",
    disable=[
        "parser",
        "tagger",
        "ner",
    ])


class FeatureBuilder:

    def __init__(self, stance, body):

        self.stance = stance  # Headline
        self.body = body  # Article body

        self.nlpstance = nlp(self.stance)
        self.nlpbody = nlp(self.body)

        # List of extracted features
        self.feats = []

        # Features to use
        self.use = [
            "hamming_distance",
            "stance_polarity",
            "body_polarity",
            "doc_similarity",
            "word_overlap",
        ]

        # Build features
        for x in self.use:
            getattr(self, x)()

    def hamming_distance(self):
        """
        Hamming distance between stance/body binary count vectors
        """
        stancevec = vectorizer.transform([self.stance])
        bodyvec = vectorizer.transform([self.body])
        assert(stancevec.shape == bodyvec.shape)

        # Compute Hamming distance
        dist = hamming(stancevec.toarray(), bodyvec.toarray())
        self.feats.append(dist)

    def stance_polarity(self):
        """
        Sentiment polarity of stance
        """
        self.feats.append(self.nlpstance.sentiment)

    def body_polarity(self):
        """
        Sentiment polarity of body
        """
        self.feats.append(self.nlpbody.sentiment)

    def doc_similarity(self):
        """
        spaCy document similarity
        """
        sim = self.nlpstance.similarity(self.nlpbody)
        self.feats.append(sim)

    def word_overlap(self):
        """
        Word overlap
        """
        sset = set(tok.lemma_ for tok in self.nlpstance)
        bset = set(tok.lemma_ for tok in self.nlpbody)
        intersec = len(sset.intersection(bset))
        union = len(sset.union(bset))
        self.feats.append(intersec / union)


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
    global vectorizer
    if os.path.isfile(config["vectorizer"]):
        # Load vectorizer object from serialized file
        vectorizer = pickle.load(open(config["vectorizer"], "rb"))
    else:
        # Initialize binary count vectorizer
        vectorizer = CountVectorizer(
            binary=True,
            # stop_words="english",
        )
        # Fit count vectorizer to all stance and body text
        vectorizer.fit(df["Headline"].tolist() + df["articleBody"].tolist())
        # Write vectorizer object to serialized file
        pickle.dump(vectorizer, open(config["vectorizer"], "wb"))

    # List of list of features per stance/body pair
    features = []
    # List of features/columns
    cols = []

    # Process each row in merged data frame
    for idx, row in df.iterrows():
        # Build features
        fb = FeatureBuilder(row["Headline"], row["articleBody"])
        # Append label
        features.append(fb.feats + [row["Stance"]])
        # Get list of features
        cols = fb.use

    # Append label
    cols += ["label"]

    # Write training data to feature file
    df = pd.DataFrame(features, columns=cols)
    df.to_csv(config["{0}_feats".format(split)], index=False)

    return df
