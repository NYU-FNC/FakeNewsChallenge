import pickle

from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
from tqdm import tqdm

from utils import (
    load_config,
    prep_text,
)

config = load_config()

# Preprocessing text
print("Preprocessing text...")
data = []
for doc in tqdm(open(config["gigaword"], "r")):
    data.append(prep_text(doc))

# Generate dictionary and document-term matrix
dct = Dictionary(data)
doc_term_matrix = [dct.doc2bow(doc) for doc in data]

# Train model
print("Training LDA model...")
lda = LdaModel(
    doc_term_matrix,
    id2word=dct,
    num_topics=300,
    minimum_probability=0.0,
)

pickle.dump(lda, open(config["lda_model"], "wb"))
pickle.dump(dct, open(config["lda_dct"], "wb"))
