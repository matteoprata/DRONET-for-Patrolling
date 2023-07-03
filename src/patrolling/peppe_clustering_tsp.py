from src.patrolling.meta_patrolling import PrecomputedPolicy

import numpy as np
from sklearn.cluster import SpectralClustering
from collections import defaultdict
from src.utilities.utilities import Christofides
from sklearn.metrics.pairwise import euclidean_distances


class PeppeClusteringTSP(PrecomputedPolicy):

    def __init__(self, set_drones, set_targets):
        super().__init__(set_drones=set_drones, set_targets=set_targets)

    def set_tour(self) -> dict:
        return self.my_solution()  # {0: [0, 1, 2, 4, 2, 3, 0, 3], 1: [3]}

    def my_solution(self) -> dict:
        n_drones = len(self.set_drones)
        n_targets = len(self.set_targets)-1

        print('n_drones:', n_drones)
        print('n_targets:', n_targets)

        targets_theta = np.asarray([t.maximum_tolerated_idleness for t in self.set_targets[1:]])
        targets_coo = np.asarray([np.array(t.coords) for t in self.set_targets[1:]])
        distances = euclidean_distances(targets_coo, targets_coo)

        distances = distances.T - targets_theta
        np.fill_diagonal(distances, 0)

        # rendere distances simmetrica
        distances = distances / np.linalg.norm(distances)

        affinities = 1-distances

        spectral_clustering_vals = SpectralClustering(
            n_clusters=n_drones,
            random_state=0,
            affinity='precomputed',
        ).fit(affinities)
        clusters = np.array(spectral_clustering_vals.labels_)

        # drones assignment
        plan = defaultdict(list)
        for id_target, id_drone in enumerate(clusters):
            plan[id_drone].append(id_target+1)

        # path optimization
        for d in plan:
            target_to_visit = [self.set_targets[tid].coords for tid in plan[d]]
            if len(target_to_visit) >= 2:
                tsp_path = Christofides().compute_from_coordinates(target_to_visit, 0)
                plan[d] = [plan[d][tp] for tp in tsp_path]

        return plan  # {0: [1, 2, 1, 3]}  # plan  # {0: [1, 2]}
