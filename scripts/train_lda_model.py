import pickle
import wikipedia

from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
from tqdm import tqdm

from utils import (
    load_config,
    prep_text,
)

config = load_config()

data = []

n = 100000
random_articles = wikipedia.random(pages=n)

# Preprocess articles
print("Preprocessing articles..")
for article in tqdm(random_articles, total=n):
    try:
        content = wikipedia.page(article).content
    except wikipedia.exceptions.DisambiguationError as e:
        content = wikipedia.page(e.options[0]).content
    prep = prep_text(content)
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
