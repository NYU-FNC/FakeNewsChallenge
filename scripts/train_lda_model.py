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
print("Loading text8 data...")
data = api.load("text8")

# Preprocess data
print("Preprocessing data...")
text = " ".join(str(x) for x in data)
prep = prep_text(text)
data = prep.split()

print(data)

# Generate corpus and dictionary
dct = Dictionary(data)
corpus = [dct.doc2bow(line) for line in data]

# Train model
print("Training LDA model..")
lda = LdaModel(
    corpus=corpus,
    id2word=dct,
    num_topics=5,
    minimum_probability=0.0,
)

pickle.dump(lda, open(config["lda_model"], "wb"))
pickle.dump(dct, open(config["lda_dct"], "wb"))
