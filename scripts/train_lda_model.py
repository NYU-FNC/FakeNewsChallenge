import gensim.downloader as api
import pickle

from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel

from utils import (
    load_config,
    prep_text,
)

config = load_config()

# Load dataset
print("Loading Wikipedia data...")
data = api.load("wiki-english-20171001")

# Preprocess data
print("Preprocessing data...")
data_prep = []

for idx, x in enumerate(data):
    text_all = []
    for text in x["section_texts"]:
        text_all.extend(prep_text(text))
    data_prep.append(" ".join(text_all))

# Generate dictionary and document-term matrix
dct = Dictionary([article.split() for article in data_prep])
doc_term_matrix = [dct.doc2bow(article.split()) for article in data_prep]

# Train model
print("Training LDA model..")
lda = LdaModel(
    doc_term_matrix,
    id2word=dct,
    num_topics=100,
    minimum_probability=0.0,
)

pickle.dump(lda, open(config["lda_model"], "wb"))
pickle.dump(dct, open(config["lda_dct"], "wb"))
