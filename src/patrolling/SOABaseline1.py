
from src.patrolling.Baselines import PrecomputedPolicy

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
        targets_coo = [np.array(t.coords) for t in self.set_targets]

        n_drones = len(self.set_drones)
        kmeans_vals = KMeans(n_clusters=n_drones, random_state=0, n_init="auto").fit(targets_coo)
        clusters = np.array(kmeans_vals.labels_)

        # drones assignment
        plan = defaultdict(list)
        for id_target, id_drone in enumerate(clusters):
            plan[id_drone].append(id_target)

        # path optimization
        for d in plan:
            target_to_visit = [self.set_targets[tid].coords for tid in plan[d]]
            if len(target_to_visit) >= 2:
                tsp_path = Christofides().compute_from_coordinates(target_to_visit, 0)
                plan[d] = [plan[d][tp] for tp in tsp_path]

        return {0: [1, 2, 3]}  # plan  # {0: [1, 2]}


class Clustering(PrecomputedPolicy):

    def __init__(self, set_drones, set_targets):
        super().__init__(set_drones=set_drones, set_targets=set_targets)

    def set_tour(self) -> dict:
        return self.my_solution() # {0: [0, 1, 2, 4, 2, 3, 0, 3], 1: [3]}

    def my_solution(self) -> dict:
        targets_coo = [np.array(t.coords) for t in self.set_targets]
        # targets_ids = [t.identifier for t in self.set_targets]

        n_drones = len(self.set_drones)
        kmeans_vals = KMeans(n_clusters=n_drones, random_state=0, n_init="auto").fit(targets_coo)
        clusters = np.array(kmeans_vals.labels_)

        # drones assignment
        plan = defaultdict(list)
        for id_target, id_drone in enumerate(clusters):
            plan[id_drone].append(id_target)

        return plan
