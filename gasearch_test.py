import numpy as np
import random
from simpleai.search import SearchProblem
from simpleai.search.local import genetic
from simpleai.search.viewers import ConsoleViewer
from gensim.models import KeyedVectors

vectors: KeyedVectors = KeyedVectors.load("../chive-1.2-mc30_gensim/chive-1.2-mc30.kv")

start = "若手"
start_vector = vectors[start]

target = "クリスマス"
target_vector = vectors[target]


class QuerySearchProblem(SearchProblem):
    def generate_random_state(self):
        new_vec = start_vector + np.random.standard_normal(size=start_vector.size)
        candidates = vectors.similar_by_vector(new_vec, topn=100)
        return candidates[-1][0]

    def crossover(self, query1, query2):
        vec1 = vectors[query1]
        vec2 = vectors[query2]
        new_vec = (vec1 + vec2) / 2
        candidates = vectors.similar_by_vector(new_vec, topn=10)
        new_query = None
        for q, _ in candidates:
            if q != query1 and q != query2:
                new_query = q
                break
        return new_query

    def mutate(self, query):
        vec = vectors[query]
        new_vec = vec + np.random.uniform(low=-1.0, high=1.0, size=vec.size)
        candidates = vectors.similar_by_vector(new_vec, topn=10)
        new_query = None
        for q, _ in candidates:
            if q != query:
                new_query = q
                break

        return new_query

    # 状態の価値を計算
    # ここではベクトルの差（ノルム）の自乗の逆数を価値としている
    def value(self, curr_query):
        v = 0
        try:
            curr_vector = vectors[curr_query]
            d = np.linalg.norm(curr_vector - target_vector) ** 2
            v = 100.0 / (1.0 + d)
        except Exception as e:
            print(e)
            print(curr_query)
        return v


problem = QuerySearchProblem()
# result = simulated_annealing(problem, iterations_limit=100, viewer=ConsoleViewer())
result = genetic(
    problem,
    population_size=1000,
    crossover_rate=0.8,
    mutation_chance=0.1,
    iterations_limit=100,
    viewer=ConsoleViewer(),
)
print(result.state, result.path())
