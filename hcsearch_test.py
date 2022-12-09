import numpy as np
import random
from simpleai.search import SearchProblem, hill_climbing
from gensim.models import KeyedVectors

NUM_CANDIDATES = 5 # 一度に探索する単語の数
NUM_MOD_RATIO = 0.33 # 33%の要素を変更
TARGET = "Java"

vectors = KeyedVectors.load("../chive-1.2-mc5_gensim/chive-1.2-mc5.kv")
# ↓ 必ずしも TARGET に設定した単語が含まれているとは限らないので
target_vector = vectors[vectors.similar_by_key(TARGET)[0][0]]

class QuerySearchProblem(SearchProblem):
    def actions(self, curr_state):
        num_mod_indices = int(len(curr_state) * NUM_MOD_RATIO)
        actions = []
        for _ in range(NUM_CANDIDATES):
            idx_candidates = list(range(len(curr_state)))
            pos_dif = []
            for _ in range(num_mod_indices):
                rnd = random.randrange(len(idx_candidates))
                pos = idx_candidates.pop(rnd)
                dif = random.random() * 1.0 - 0.5 # -0.5〜0.5の範囲で変更
                pos_dif.append((pos, dif))
            actions.append(pos_dif)
        return actions

    def value(self, curr_state):
        keywords = vectors.similar_by_vector(curr_state, topn=5)
        print(f"keywords = {keywords}")
        keyword = keywords[0][0] 
        keyword_vector = vectors[keyword]
        dvec = keyword_vector - target_vector
        d = 1.0 / (1.0 + np.linalg.norm(dvec)) + random.random() * 0.1
        print(f"value = {d}")
        return d

    def result(self, curr_state, action):
        new_state = np.array(curr_state)
        # action = list[(pos, dif)]
        # pos: ベクトル上の位置（要素番号）
        # dif: ベクトルの要素に加算する乱数の値
        for p, d in action:
            new_state[p] += d
        return new_state

    def is_goal(self, curr_state):
        keyword = vectors.similar_by_vector(curr_state)[0][0]
        return keyword == TARGET


initial_query = "python"
initial_query_vector = vectors[vectors.similar_by_word(initial_query)[0][0]]
problem = QuerySearchProblem(initial_state = initial_query_vector)
result = hill_climbing(problem)

print(result.path())
