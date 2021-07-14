

from collections import defaultdict
import itertools
import bisect
import random


import functools


@functools.total_ordering
class Solution:

    def __init__(self, graph, start, ant=None):
        self.graph = graph
        self.start = start
        self.ant = ant
        self.current = start
        self.cost = 0
        self.path = []
        self.nodes = [start]
        self.visited = set(self.nodes)

    def __iter__(self):
        return iter(self.path)

    def __eq__(self, other):
        return self.cost == other.cost

    def __lt__(self, other):
        return self.cost < other.cost

    def __contains__(self, item):
        return item in self.visited or item == self.current

    def __repr__(self):
        easy_id = self.get_easy_id(sep=',', monospace=False)
        return '{}\t{}'.format(self.cost, easy_id)

    def get_easy_id(self, sep=' ', monospace=True):
        nodes = [str(n) for n in self.get_id()]
        if monospace:
            size = max([len(n) for n in nodes])
            nodes = [n.rjust(size) for n in nodes]
        return sep.join(nodes)

    def get_id(self):
        first = min(self.nodes)
        index = self.nodes.index(first)
        return tuple(self.nodes[index:] + self.nodes[:index])

    def add_node(self, node):
        self.nodes.append(node)
        self.visited.add(node)
        self._add_node(node)

    def _add_node(self, node):
        edge = self.current, node
        data = self.graph.edges[edge]
        self.path.append(edge)
        self.cost += data['weight']
        self.current = node

    def close(self):
        self._add_node(self.start)


class Ant:

    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
        self.sales = None
        self.graph = None
        self.n = None

    def tour(self, graph, sales, start, opt2):
        self.graph = graph
        self.sales = sales
        self.n = len(graph.nodes)

        solutions = [Solution(graph, start, self) for _ in range(sales)]

        # できる限り均等にセールスマンが通る
        # as possible,  salesman travels vertexes as same
        saleses = [(self.n - 1) // sales for i in range(sales)]
        for i in range((self.n - 1) % sales):
            saleses[i] += 1

        unvisited = [i for i in range(0, self.n) if i != start]
        for i in range(sales):
            for j in range(saleses[i]):
                next_node = self.choose_destination(solutions[i].current, unvisited)
                solutions[i].add_node(next_node)
                unvisited.remove(next_node)
            solutions[i].close()

        if opt2:
            self.opt2_update(graph, opt2, sales, saleses, solutions)

        return solutions

    def opt2_update(self, graph, opt2, sales, saleses, solutions):
        for i in range(sales):
            for j in range(opt2):
                k = saleses[i] + 1
                while True:
                    a = random.randint(0, k - 1)
                    b = random.randint(0, k - 1)
                    if a != b:
                        break
                if a > b:
                    a, b = b, a
                dist_a = graph.edges[solutions[i].nodes[a], solutions[i].nodes[a + 1]]['weight']
                dist_b = graph.edges[solutions[i].nodes[b], solutions[i].nodes[(b + 1) % k]]['weight']
                dist_c = graph.edges[solutions[i].nodes[a], solutions[i].nodes[b]]['weight']
                dist_d = graph.edges[solutions[i].nodes[a + 1], solutions[i].nodes[(b + 1) % k]]['weight']
                if dist_a + dist_b > dist_c + dist_d:
                    solutions[i].nodes[a + 1:b + 1] = reversed(solutions[i].nodes[a + 1: b + 1])
                    solutions[i].cost += (dist_c + dist_d - dist_a - dist_b)
                    solutions[i].path = []
                    for l in range(k):
                        solutions[i].path.append((solutions[i].nodes[l], solutions[i].nodes[(l + 1) % k]))

    def choose_destination(self, current, unvisited):
        if len(unvisited) == 1:
            return unvisited[0]
        scores = self.get_scores(current, unvisited)
        return self.choose_node(unvisited, scores)

    def choose_node(self, unvisited, scores):
        total = sum(scores)
        cumdist = list(itertools.accumulate(scores))
        index = bisect.bisect(cumdist, random.random() * total)
        return unvisited[min(index, len(unvisited) - 1)]

    def get_scores(self, current, unvisited):
        scores = []
        for node in unvisited:
            edge = self.graph.edges[current, node]
            score = self.score_edge(edge)
            scores.append(score)
        return scores

    def score_edge(self, edge):
        weight = edge.get('weight', 1)
        if weight == 0:
            return 1e200
        phe = edge['pheromone']
        return phe ** self.alpha * (1 / weight) ** self.beta



class Colony:
    def __init__(self, alpha=1, beta=3):
        self.alpha = alpha
        self.beta = beta

    def get_ants(self, size):
        return [Ant(self.alpha, self.beta) for i in range(size)]


class Solver:

    def __init__(self, rho=0.03, q=1, top=None):
        self.rho = rho
        self.q = q
        self.top = top

    def solve(self, *args, **kwargs):
        best = None
        for solution in self.optimize(*args, **kwargs):
            if best is None:
                best = solution
            elif sum([s.cost for s in best]) > sum([s.cost for s in solution]):
                best = solution
        return best

    def optimize(self, graph, colony, sales, start=1, gen_size=None, limit=50, opt2=None):
        gen_size = gen_size or len(graph.nodes)
        ants = colony.get_ants(gen_size)

        for u, v in graph.edges:
            weight = graph.edges[u, v]['weight']
            # print(u,v, weight)

            if weight == 0:
                weight = 1e100
            graph.edges[u, v].setdefault('pheromone', 1 / weight)

        for _ in range(limit):
            sales_solutions = self.find_solutions(graph, ants, sales, start, opt2)
            for solutions in sales_solutions:
                solutions.sort()
            sales_solutions.sort(key=lambda x: sum([y.cost for y in x]))
            self.global_update(sales_solutions, graph)

            yield sales_solutions[0]

    def find_solutions(self, graph, ants, sales, start, opt2):
        return [ant.tour(graph, sales, start, opt2) for ant in ants]

    def global_update(self, sales_solutions, graph):
        next_pheromones = defaultdict(float)
        for solutions in sales_solutions:
            cost = sum([solution.cost for solution in solutions])
            for solution in solutions:
                for path in solution:
                    next_pheromones[path] += 1 / cost

        for edge in graph.edges:
            p = graph.edges[edge]['pheromone']
            graph.edges[edge]['pheromone'] = p * (1 - self.rho) + next_pheromones[edge]


def multi_agent_tsp(sales_persons, graph):
    colony = Colony(1, 3)
    solver = Solver()
    ans = solver.solve(graph, colony, sales_persons, start=0)#, limit=50, opt2=20)

    time_cost = max(s.cost for s in ans)
    return time_cost

# # draw
# colors = ['black', 'blue', 'green', 'red', 'pink', 'orange']
# plt.figure(dpi=300)
# _, ax = plt.subplots()
# pos = problem.display_data or problem.node_coords
# nx.draw_networkx_nodes(G, pos=pos, ax=ax, node_color=(0.4157, 0.3529, 0.3490))
# nx.draw_networkx_labels(G, pos=pos, labels={i: str(i) for i in range(1, len(G.nodes) + 1)}, font_size=8, font_color='white')
# for i in range(len(ans)):
#     solution = ans[i]
#     path = solution.path
#     nx.draw_networkx_edges(G, pos=pos, edgelist=path, arrows=True, edge_color=colors[i])
#     # nx.draw_networkx_edges(G, pos=pos, edgelist=path, arrows=True, edge_color=[random.random() for i in range(3)])

# # If this doesn't exsit, x_axis and y_axis's numbers are not there.
# ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
# plt.show()
