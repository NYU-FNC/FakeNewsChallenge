import os
import pickle
import sys

import pandas as pd
import spacy

from gensim.models import KeyedVectors
from scipy.spatial.distance import (
    cosine,
    hamming,
)
from scipy.stats import entropy
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer,
)
from spacy.lang.en.stop_words import STOP_WORDS
from tqdm import tqdm

from utils import prep_text

# spaCy
nlp = spacy.load(
    "en_core_web_lg",
    disable=[
        # "tagger",
        # "parser",
        "ner",
    ],
)

DEP_SUBJ = ["nsubj", "nsubjpass", "csubj", "csubjpass", "agent", "expl"]
DEP_OBJ = ["dobj", "dative", "attr", "oprd"]

INTRO_LEN = 250

MIN_NGRAM_LEN = 2
MAX_NGRAM_LEN = 4

REFUTE = {"fake", "fraud", "hoax", "not", "deny", "fabricate", "authenticity"}


def tok_dep_is_valid(tok, deps):
    """
    Check if token is valid for dependency overlap features
    """
    if (tok.dep_ in deps) and (tok.pos_ in ["NOUN", "PROPN"]) \
            and (tok.lower_ not in STOP_WORDS) and not tok.is_punct:
        return True
    return False


def generate_ngrams(tok_list, n):
    """
    Generate n-grams
    """
    return zip(*[tok_list[i:] for i in range(n)])


class FeatureBuilder:

    def __init__(self, features, row):

        self.stance = row["Headline"]  # Headline
        self.body = row["articleBody"]  # Article body

        # Process stance/body using spaCy
        self.nlpstance = nlp(self.stance)
        self.nlpbody = nlp(self.body)

        # Get sentiment scores
        self.stance_sentiment_score = row["stances_sentiment"]
        self.body_sentiment_score = row["bodies_sentiment"]

        # Get LDA probability distributions
        stance_bow = lda_dct.doc2bow(prep_text(self.stance))
        body_bow = lda_dct.doc2bow(prep_text(self.body))

        stance_proba = lda_model.get_document_topics(stance_bow)
        body_proba = lda_model.get_document_topics(body_bow)

        self.pk = [prob for topic, prob in stance_proba]
        self.qk = [prob for topic, prob in body_proba]
        assert(len(self.pk) == len(self.qk))

        # List of extracted features
        self.feats = []

        # Features to use
        self.use = features.copy()

        # Build features
        for x in self.use:
            getattr(self, x)()

    def cosine_count(self):
        """
        Cosine similarity between stance/body count vectors
        """
        stancevec = count_vec.transform([self.stance])
        bodyvec = count_vec.transform([self.body])
        assert(stancevec.shape == bodyvec.shape)

        # Compute cosine similarity
        dist = cosine(stancevec.toarray(), bodyvec.toarray())
        self.feats.append(dist)

    def cosine_tdidf(self):
        """
        Cosine similarity between stance/body TF-IDF vectors
        """
        stancevec = tfidf_vec.transform([self.stance])
        bodyvec = tfidf_vec.transform([self.body])
        assert(stancevec.shape == bodyvec.shape)

        # Compute cosine similarity
        dist = cosine(stancevec.toarray(), bodyvec.toarray())
        self.feats.append(dist)

    def dep_object_overlap(self):
        """
        Object overlap between stance/body spaCy objects
        """
        s_obj = set([tok.lemma_ for tok in self.nlpstance
                     if tok_dep_is_valid(tok, DEP_OBJ)])
        b_obj = set([tok.lemma_ for tok in self.nlpbody
                     if tok_dep_is_valid(tok, DEP_OBJ)])
        intersec = s_obj.intersection(b_obj)
        self.feats.append(len(intersec))

    def dep_subject_overlap(self):
        """
        Subject overlap between stance/body spaCy subjects
        """
        s_subj = set([tok.lemma_ for tok in self.nlpstance
                      if tok_dep_is_valid(tok, DEP_SUBJ)])
        b_subj = set([tok.lemma_ for tok in self.nlpbody
                      if tok_dep_is_valid(tok, DEP_SUBJ)])
        intersec = s_subj.intersection(b_subj)
        self.feats.append(len(intersec))

    def doc_similarity(self):
        """
        spaCy document similarity
        """
        sim = self.nlpstance.similarity(self.nlpbody)
        self.feats.append(sim)

    def doc_similarity_intro(self):
        """
        spaCy document similarity (intro)
        """
        sim = self.nlpstance.similarity(self.nlpbody[:INTRO_LEN])
        self.feats.append(sim)

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

    def len_stance(self):
        """
        Length heading
        """
        self.feats.append(len(self.stance.split()))

    def len_body(self):
        """
        Length body
        """
        self.feats.append(len(self.body.split()))

    def ngram_overlap(self):
        """
        N-gram overlap
        """
        stance_toks = [tok.lemma_ for tok in self.nlpstance if
                       not ((tok.lower_ in STOP_WORDS) or tok.is_punct)]
        body_toks = [tok.lemma_ for tok in self.nlpbody if
                     not ((tok.lower_ in STOP_WORDS) or tok.is_punct)]

        stance_ngrams = []
        for n in range(MIN_NGRAM_LEN, MAX_NGRAM_LEN + 1):
            ngrams = generate_ngrams(stance_toks, n)
            stance_ngrams.extend(ngrams)

        body_ngrams = []
        for n in range(MIN_NGRAM_LEN, MAX_NGRAM_LEN + 1):
            ngrams = generate_ngrams(body_toks, n)
            body_ngrams.extend(ngrams)

        sset = set(stance_ngrams)
        bset = set(body_ngrams)

        intersec = len(sset.intersection(bset))
        union = len(sset.union(bset))
        self.feats.append(intersec / union)

    def ngram_overlap_intro(self):
        """
        N-gram overlap (intro)
        """
        stance_toks = [tok.lemma_ for tok in self.nlpstance if
                       not ((tok.lower_ in STOP_WORDS) or tok.is_punct)]
        body_toks = [tok.lemma_ for tok in self.nlpbody[:INTRO_LEN] if
                     not ((tok.lower_ in STOP_WORDS) or tok.is_punct)]

        stance_ngrams = []
        for n in range(MIN_NGRAM_LEN, MAX_NGRAM_LEN + 1):
            ngrams = generate_ngrams(stance_toks, n)
            stance_ngrams.extend(ngrams)

        body_ngrams = []
        for n in range(MIN_NGRAM_LEN, MAX_NGRAM_LEN + 1):
            ngrams = generate_ngrams(body_toks, n)
            body_ngrams.extend(ngrams)

        sset = set(stance_ngrams)
        bset = set(body_ngrams)

        intersec = len(sset.intersection(bset))
        union = len(sset.union(bset))
        self.feats.append(intersec / union)

    def KL_pk_qk(self):
        """
        Kullback-Leibler divergence (pk, qk)
        """
        div = entropy(self.pk, qk=self.qk)
        self.feats.append(div)

    def KL_qk_pk(self):
        """
        Kullback-Leibler divergence (qk, pk)
        """
        div = entropy(self.qk, qk=self.pk)
        self.feats.append(div)

    def refute(self):
        """
        Refute words
        """
        present = 0
        if any(tok.lemma_ in REFUTE for tok in self.nlpbody):
            present = 1
        self.feats.append(present)

    def refute_intro(self):
        """
        Refute words (intro)
        """
        present = 0
        if any(tok.lemma_ in REFUTE for tok in self.nlpbody[:INTRO_LEN]):
            present = 1
        self.feats.append(present)

    def sentiment_body(self):
        """
        Average CoreNLP sentiment score of body
        """
        self.feats.append(self.body_sentiment_score)

    def sentiment_stance(self):
        """
        Average CoreNLP sentiment score of stance
        """
        self.feats.append(self.stance_sentiment_score)

    def word_overlap(self):
        """
        Word overlap
        """
        sset = set(tok.lemma_ for tok in self.nlpstance if
                   not ((tok.lower_ in STOP_WORDS) or tok.is_punct))
        bset = set(tok.lemma_ for tok in self.nlpbody if
                   not ((tok.lower_ in STOP_WORDS) or tok.is_punct))
        intersec = len(sset.intersection(bset))
        union = len(sset.union(bset))
        self.feats.append(intersec / union)

    def word_overlap_intro(self):
        """
        Word overlap (intro)
        """
        sset = set(tok.lemma_ for tok in self.nlpstance if
                   not ((tok.lower_ in STOP_WORDS) or tok.is_punct))
        bset = set(tok.lemma_ for tok in self.nlpbody[:INTRO_LEN] if
                   not ((tok.lower_ in STOP_WORDS) or tok.is_punct))
        intersec = len(sset.intersection(bset))
        union = len(sset.union(bset))
        self.feats.append(intersec / union)

    def wmdistance(self):
        """
        Word movers distance (WMD)
        """
        stance = " ".join(tok.lower_ for tok in self.nlpstance if
                          not ((tok.lower_ in STOP_WORDS) or tok.is_punct))
        body = " ".join(tok.lower_ for tok in self.nlpbody if
                        not ((tok.lower_ in STOP_WORDS) or tok.is_punct))
        dist = w2v_model.wmdistance(stance, body)
        self.feats.append(dist)


def load_or_generate_vectorizer(which, df):
    """
    Load or generate vectorizer
    """
    if which not in ("count", "tfidf"):
        print("Error: {0} is not a valid vectorizer, use count|tfidf".format(which))
        sys.exit(1)

    # Load vectorizer object from serialized file
    if os.path.isfile(config[which + "_vec"]):
        vec = pickle.load(open(config[which + "_vec"], "rb"))
    else:
        # Generate CountVectorizer()
        if which == "count":
            vec = CountVectorizer(
                binary=True,
                min_df=2,
                ngram_range=(1, 3),
                stop_words=STOP_WORDS,
            )
        # Generate TfidfVectorizer()
        if which == "tfidf":
            vec = TfidfVectorizer(
                min_df=2,
                ngram_range=(1, 3),
                stop_words=STOP_WORDS,
            )

        # Fit count vectorizer to all stance and body text
        vec.fit(df["Headline"].tolist() + df["articleBody"].tolist())

        # Write vectorizer object to serialized file
        pickle.dump(vec, open(config[which + "_vec"], "wb"))

    return vec


def build_features(split, config_file):
    """
    Build features for train|test split
    """
    global config
    config = config_file

    global w2v_model
    w2v_model = KeyedVectors.load_word2vec_format(config["embeddings"], binary=True)

    global lda_model
    global lda_dct
    lda_model = pickle.load(open(config["lda_model"], "rb"))
    lda_dct = pickle.load(open(config["lda_dct"], "rb"))

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

    # Get count vectorizer
    global count_vec
    count_vec = load_or_generate_vectorizer("count", df)

    # Get TF-IDF vectorizer
    global tfidf_vec
    tfidf_vec = load_or_generate_vectorizer("tfidf", df)

    # List of list of features per stance/body pair
    features = []
    # List of features/columns
    cols = []

    # Process each row in merged data frame
    for idx, row in tqdm(df.iterrows(), total=len(df.index)):
        # Build features
        fb = FeatureBuilder(config["feats"], row)
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
