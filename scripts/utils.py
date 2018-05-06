import argparse
import yaml

from spacy.lang.en.stop_words import STOP_WORDS


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
    prep = []
    for tok in text.split():
        if tok.isalnum() and tok not in STOP_WORDS:
            prep.append(tok.lower())
    return prep
