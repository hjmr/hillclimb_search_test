from simpleai.search import SearchProblem
from simpleai.search.traditional import astar
from simpleai.search.local import hill_climbing, simulated_annealing
from simpleai.search.viewers import ConsoleViewer

GOAL = "HELLO WORLD"


class HelloProblem(SearchProblem):
    def actions(self, state):
        if len(state) < len(GOAL):
            return list("- ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        else:
            return []

    def result(self, state, action):
        if 1 < len(state) and action == "-":
            new_state = state[:-1]
        else:
            new_state = state + action
        return new_state

    def is_goal(self, state):
        return state == GOAL

    def heuristic(self, state):
        # how far are we from the goal?
        wrong = sum([1 if state[i] != GOAL[i] else 0 for i in range(len(state))])
        missing = len(GOAL) - len(state)
        return wrong + missing

    def value(self, state):
        # how close are we to the goal?
        match = sum([1 if state[i] == GOAL[i] else 0 for i in range(len(state))])
        missing = len(GOAL) - len(state)
        return match - missing


problem = HelloProblem(initial_state="")
# result = astar(problem, viewer=ConsoleViewer())
# result = hill_climbing(problem, viewer=ConsoleViewer())
result = simulated_annealing(problem, iterations_limit=100, viewer=ConsoleViewer())

print(result.state)
print(result.path())
