

from src.patrolling.meta_patrolling import PrecomputedPolicy

import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict
from src.utilities.utilities import Christofides


class ClusteringTSP(PrecomputedPolicy):

    def __init__(self, set_drones, set_targets):
        super().__init__(set_drones=set_drones, set_targets=set_targets)

    def set_tour(self) -> dict:
        return self.my_solution()  # {0: [0, 1, 2, 4, 2, 3, 0, 3], 1: [3]}

    def my_solution(self) -> dict:
        targets_coo = np.array([np.array(list(t.coords) + []) for t in self.set_targets][1:])  # + [t.maximum_tolerated_idleness]
        n_drones = len(self.set_drones)

        n_clusters = n_drones
        print("n_clusters", n_clusters)

        kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=0)
        kmeans.fit(targets_coo)
        labels = kmeans.labels_
        target_clusters = np.array(labels)  # [0, 0, 0, 1, 1, 1, ...]

        # TSP simple
        # drones assignment
        plan = defaultdict(list)
        # plan[0].append(0)
        for id_target, id_drone in enumerate(target_clusters):
            plan[id_drone].append(id_target + 1)

        # path optimization
        for d in plan:
            target_to_visit = [self.set_targets[tid].coords for tid in plan[d]]
            if len(target_to_visit) >= 2:
                tsp_path = Christofides().compute_from_coordinates(target_to_visit, 0)
                plan[d] = [plan[d][tp] for tp in tsp_path]

        return plan
