import gensim
from gensim import corpora
import pickle

NUM_TOPICS = 5

dictionary = corpora.Dictionary.load("dictionary.gensim")
corpus = pickle.load(open("corpus.pkl",'rb'))
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=15)
ldamodel.save('model5.gensim')
topics = ldamodel.print_topics(num_words=4)
for topic in topics:
    print(topic)
