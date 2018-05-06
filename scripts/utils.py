import argparse
import spacy
import yaml

from spacy.lang.en.stop_words import STOP_WORDS

# spaCy
nlp = spacy.load(
    "en_core_web_lg",
    disable=[
        "tagger",
        "parser",
        "ner",
    ],
)


def load_config():
    """
    Load configuration file
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


def prep_text(text):
    """
    Preprocess text
    """
    doc = nlp(text)
    prep = []
    for tok in doc:
        if ((tok.lower_ not in STOP_WORDS) and tok.is_alpha):
            prep.append(tok.lemma_.lower())
    return prep
