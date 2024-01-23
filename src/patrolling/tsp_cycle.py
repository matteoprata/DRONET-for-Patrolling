

from src.patrolling.meta_patrolling import PrecomputedPolicy

import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict
from src.utilities.utilities import Christofides
from src.utilities.utilities import euclidean_distance


class Cycle(PrecomputedPolicy):
    name = "Cycle"
    identifier = 8
    line_tick = 8
    marker = 8

    def __init__(self, set_drones, set_targets):
        super().__init__(set_drones=set_drones, set_targets=set_targets)

    def set_tour(self) -> dict:
        plan, cost = self.my_solution()
        self.route_info = {"cost": cost}
        return plan  # {0: [0, 1, 2, 4, 2, 3, 0, 3], 1: [3]}

    def my_solution(self):
        targets_coo = np.array([np.array(list(t.coords)) for t in self.set_targets][1:])
        targets_ids = np.array([t.identifier for t in self.set_targets][1:])
        n_drones = len(self.set_drones)

        n_clusters = n_drones
        print("n_clusters", n_clusters)
        tsp_path = Christofides().compute_from_coordinates(targets_coo, 0)
        tsp_cost = np.sum([euclidean_distance(targets_coo[tsp_path[i]], targets_coo[tsp_path[i+1]]) for i in range(len(tsp_path)-1)])
        tsp_cost += np.sum(euclidean_distance(targets_coo[tsp_path[0]], targets_coo[tsp_path[-1]]))

        # TSP simple
        # drones assignment
        plan = dict()
        # path optimization
        for d in range(n_drones):
            #is_not_first = 1 if d > 0 else 0
            shift = 0  # ((len(tsp_path) // n_drones) * d)
            print(shift, tsp_path, np.roll(tsp_path, shift=-shift))
            plan[d] = [targets_ids[tindex] for tindex in np.roll(tsp_path, shift=-shift)]
        return plan, tsp_cost
