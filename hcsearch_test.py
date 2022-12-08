from abc import ABC

import numpy as np
import random
from simpleai.search import SearchProblem, hill_climbing
from gensim.models import Word2Vec

model = Word2Vec.load("latest-ja-word2vec-gensim-model/word2vec.gensim.model")
vector = model.wv["Python"]
sims = model.wv.similar_by_vector(vector)

print(sims)
