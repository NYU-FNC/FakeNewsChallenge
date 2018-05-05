import argparse
import spacy
import yaml

# spaCy
nlp = spacy.load("en_core_web_lg")


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
        if not tok.is_stop and not tok.is_punct:
            prep.append(tok.lemma_)
    return " ".join(prep)
