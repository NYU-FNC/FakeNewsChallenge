import os
import pickle
import sys

import pandas as pd
import spacy

from tqdm import tqdm

from scipy.spatial.distance import hamming, cosine
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# spaCy
nlp = spacy.load(
    "en_core_web_lg",
    disable=[
        "tagger",
        "parser",
        "ner",
    ])


class FeatureBuilder:

    def __init__(self, row):

        self.stance = row["Headline"]  # Headline
        self.body = row["articleBody"]  # Article body

        self.nlpstance = nlp(self.stance)
        self.nlpbody = nlp(self.body)

        self.stance_sentiment_score = row["stances_sentiment"]
        self.body_sentiment_score = row["bodies_sentiment"]

        # List of extracted features
        self.feats = []

        # Features to use
        self.use = [
            "hamming_distance",
            "stance_sentiment",
            "body_sentiment",
            "doc_similarity",
            "word_overlap",
            # "dep_subject_overlap",
            # "dep_object_overlap",
            "tfidf_cosine",
        ]

        # Build features
        for x in self.use:
            getattr(self, x)()

    def hamming_distance(self):
        """
        Hamming distance between stance/body binary count vectors
        """
        stancevec = count_vec.transform([self.stance])
        bodyvec = count_vec.transform([self.body])
        assert(stancevec.shape == bodyvec.shape)

        # Compute Hamming distance
        dist = hamming(stancevec.toarray(), bodyvec.toarray())
        self.feats.append(dist)

    def stance_sentiment(self):
        """
        Average CoreNLP sentiment score of stance
        """
        self.feats.append(self.stance_sentiment_score)

    def body_sentiment(self):
        """
        Average CoreNLP sentiment score of body
        """
        self.feats.append(self.body_sentiment_score)

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

    def dep_subject_overlap(self):
        """
        Subject overlap between stance/body spaCy subjects (binary)
        """
        SUBJECTS = ["nsubj", "nsubjpass", "csubj", "csubjpass", "agent", "expl"]

        stance_subjects = set([tok for tok in self.nlpstance if tok.dep_ in SUBJECTS])
        body_subjects = set([tok for tok in self.nlpbody if tok.dep_ in SUBJECTS])

        if (len(stance_subjects.intersection(body_subjects)) > 0):
            self.feats.append(1.0)

    def dep_object_overlap(self):
        """
        Object overlap between stance/body spaCy objects (binary)
        """
        OBJECTS = ["dobj", "dative", "attr", "oprd"]

        stance_objects = set([tok for tok in self.nlpstance if tok.dep_ in OBJECTS])
        body_objects = set([tok for tok in self.nlpbody if tok.dep_ in OBJECTS])

        if (len(stance_objects.intersection(body_objects)) > 0):
            self.feats.append(1.0)

    def tfidf_cosine(self):
        """
        Cosine similarity between stance/body TF-IDF vectors
        """
        stancevec = tfidf_vec.transform([self.stance])
        bodyvec = tfidf_vec.transform([self.body])
        assert(stancevec.shape == bodyvec.shape)

        # Compute cosine similarity
        dist = cosine(stancevec.toarray(), bodyvec.toarray())
        self.feats.append(dist)


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

    # Add sentiment scores
    stances_scores = [
        float(s) for s in open(config["{0}_stances_sentiment".format(split)], "r")]
    bodies_scores = [
        float(s) for s in open(config["{0}_bodies_sentiment".format(split)], "r")]

    assert(len(df_stances) == len(stances_scores))
    assert(len(df_bodies) == len(bodies_scores))

    df_stances["stances_sentiment"] = stances_scores
    df_bodies["bodies_sentiment"] = bodies_scores

    # Merge stances and bodies
    df = pd.merge(df_stances, df_bodies, on="Body ID", how="left")

    # Load or initialize CountVectorizer()
    global count_vec
    if os.path.isfile(config["count_vec"]):
        # Load vectorizer object from serialized file
        count_vec = pickle.load(open(config["count_vec"], "rb"))
    else:
        count_vec = CountVectorizer(
            binary=True,
            # stop_words="english",
        )
        # Fit count vectorizer to all stance and body text
        count_vec.fit(df["Headline"].tolist() + df["articleBody"].tolist())
        # Write vectorizer object to serialized file
        pickle.dump(count_vec, open(config["count_vec"], "wb"))

    # Load or initialize TfidfVectorizer()
    global tfidf_vec
    if os.path.isfile(config["tfidf_vec"]):
        # Load vectorizer object from serialized file
        tfidf_vec = pickle.load(open(config["tfidf_vec"], "rb"))
    else:
        tfidf_vec = TfidfVectorizer()
        # Fit TF-IDF vectorizer to all stance and body text
        tfidf_vec.fit(df["Headline"].tolist() + df["articleBody"].tolist())
        # Write vectorizer object to serialized file
        pickle.dump(tfidf_vec, open(config["tfidf_vec"], "wb"))

    # List of list of features per stance/body pair
    features = []
    # List of features/columns
    cols = []

    # Process each row in merged data frame
    for idx, row in tqdm(df.iterrows(), total=len(df.index)):
        # Build features
        fb = FeatureBuilder(row)
        # Append label
        features.append(fb.feats + [row["Stance"]])
        # Get list of features
        cols = fb.use
        # if idx == 100:
        #     break

    # Append label
    cols += ["label"]

    # Write training data to feature file
    df = pd.DataFrame(features, columns=cols)
    df.to_csv(config["{0}_feats".format(split)], index=False)

    return df
