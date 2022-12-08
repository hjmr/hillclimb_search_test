from abc import ABC

import numpy as np
import random
from simpleai.search import SearchProblem, hill_climbing
from gensim.models import KeyedVectors

vectors = KeyedVectors.load("../chive-1.2-mc5_gensim/chive-1.2-mc5.kv")

query_vector = vectors["python"]
sims   = vectors.similar_by_vector(query_vector)

print(sims)
