import pickle
import wikipedia

from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel

from utils import (
    load_config,
    prep_text,
)

config = load_config()

data = []
random_articles = wikipedia.random(pages=100000)

for article in random_articles:
    prep = prep_text(wikipedia.page(article).content)
    data.append(prep)

# Generate dictionary and document-term matrix
dct = Dictionary(data)
doc_term_matrix = [dct.doc2bow(article) for article in data]

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
