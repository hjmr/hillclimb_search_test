import numpy as np
import random
from simpleai.search import SearchProblem
from simpleai.search.local import hill_climbing_stochastic, simulated_annealing
from simpleai.search.viewers import ConsoleViewer
from gensim.models import KeyedVectors

NUM_CANDIDATES = 20  # 一度に探索する単語の数

vectors = KeyedVectors.load("../chive-1.2-mc30_gensim/chive-1.2-mc30.kv")

target = "クリスマス"
target_vector = vectors[target]


class QuerySearchProblem(SearchProblem):
    # 次の状態に至るための action のリストを生成
    # ここでは次のクエリ候補を生成している
    def actions(self, curr_query):
        actions = []
        if curr_query != target:
            # actions.append(curr_query)
            curr_vector = vectors[curr_query]
            for _ in range(NUM_CANDIDATES):
                new_vector = curr_vector + np.random.uniform(size=curr_vector.size, low=-0.5, high=0.5)
                new_keywords = vectors.similar_by_vector(new_vector, topn=NUM_CANDIDATES)
                for k, _ in new_keywords:
                    # 現在のqueryと異なり，かつすでにクエリ候補に入っていない単語をクエリ候補とする
                    if k != curr_query and k not in actions:
                        actions.append(k)
                        break
        return actions

    # 状態にアクションを適用した際の次の状態を生成
    # アクション＝次のクエリ候補なのでそのまま戻す
    def result(self, curr_query, action):
        return action

    # 状態の価値を計算
    # ここではベクトルの差（ノルム）の逆数を価値としている
    def value(self, curr_query):
        curr_vector = vectors[curr_query]
        d = curr_vector - target_vector
        v = 100.0 / (1.0 + np.linalg.norm(d))
        print(f"query = {curr_query}, value={v}")
        return v


initial_query = "若手"
problem = QuerySearchProblem(initial_state=initial_query)
# result = simulated_annealing(problem, iterations_limit=100, viewer=ConsoleViewer())
result = hill_climbing_stochastic(problem, iterations_limit=100, viewer=ConsoleViewer())

print(result.state, result.path())
