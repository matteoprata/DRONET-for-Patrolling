import time

from src.patrolling.meta_patrolling import PrecomputedPolicy

import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict
from src.utilities.utilities import Christofides


import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from mpl_toolkits.mplot3d import Axes3D
from kneed import KneeLocator


class INFOCOM_Patrol(PrecomputedPolicy):

    def __init__(self, set_drones, set_targets):
        super().__init__(set_drones=set_drones, set_targets=set_targets)

    def set_tour(self) -> dict:
        return self.my_solution()  # {0: [0, 1, 2, 4, 2, 3, 0, 3], 1: [3]}

    def my_solution(self) -> dict:
        ids_targets = {e: t for e, t in enumerate(self.set_targets)}
        targets_coo = np.array([np.array(list(t.coords) + [t.maximum_tolerated_idleness]) for t in self.set_targets][1:])  # +
        n_drones = len(self.set_drones)

        k_values = range(1, n_drones)

        # Plot the elbow curve
        # ax = fig.add_subplot(122)
        inertia_values = []
        clusterings = []
        for k in k_values:
            kmeans = KMeans(n_clusters=k, n_init="auto", random_state=0)
            kmeans.fit(targets_coo)
            labels = kmeans.labels_

            clusterings.append(labels)
            inertia_values.append(kmeans.inertia_)

        kneedle = KneeLocator(list(k_values), inertia_values, curve='convex', direction='decreasing')
        elbow_index = kneedle.knee

        if elbow_index:
            n_clusters = elbow_index + 1
            n_clusters = min(n_clusters, n_drones)
            # n_clusters = n_drones
        else:
            n_clusters = n_drones

        print("n_clusters", n_clusters)
        # np.argmin()

        kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=0)
        kmeans.fit(targets_coo)
        labels = kmeans.labels_
        target_clusters = np.array(labels)  # [0, 0, 0, 1, 1, 1, ...]

        if n_drones > n_clusters:
            clusters_tolerances = []
            for i in range(n_clusters):
                targets_cluster_i = [ids_targets[tid] for tid in np.where(target_clusters == i)[0]]
                cluster_tolerance = np.min([t.maximum_tolerated_idleness for t in targets_cluster_i])  # low is more urgent
                clusters_tolerances.append(cluster_tolerance)
            clusters_tolerances = 1 / np.array(clusters_tolerances)  # highest means [.5, .3, .2] high priority (low thresholds)
            clusters_tolerances /= np.sum(clusters_tolerances)
            print(clusters_tolerances)
            clusters_assignments = clusters_tolerances * (n_drones - n_clusters)
            clusters_assignments = np.floor(clusters_assignments) + 1

            spare = int(n_drones - np.sum(clusters_assignments))
            if spare > 0:  # there are spare drones
                sorted_clusters_tolerances_ix = np.argsort(list(clusters_tolerances))[::-1]
                for idx in range(spare):
                    clusters_assignments[sorted_clusters_tolerances_ix[idx]] += 1

            # clusters_assignments how many drone per cluster [ 1. 10.  2.  1.  1.]
            print(clusters_assignments)

            # drones assignment
            plan = defaultdict(list)

            clid_tars = {}   # map cluster id : target ids
            for i in range(n_clusters):
                targets_cluster_i = [ids_targets[tid+1] for tid in np.where(target_clusters == i)[0]]  # +1 for depot
                cluster_tids = [t.identifier for t in targets_cluster_i]
                clid_tars[i] = cluster_tids

            print(clid_tars)

            id_drone_so_far = 0
            for i in range(n_clusters):
                # print("CLUSTER i", i)
                target_to_visit = [self.set_targets[tid].coords for tid in clid_tars[i]]
                if len(target_to_visit) >= 2:
                    tsp_path = Christofides().compute_from_coordinates(target_to_visit, 0)
                else:
                    print("Problem! Christofides")
                    exit()

                n_drones_in_cluster = int(clusters_assignments[i])
                tsp = np.array([clid_tars[i][tp] for tp in tsp_path])
                for nd in range(n_drones_in_cluster):
                    shift_am = min((len(tsp) // n_drones_in_cluster) * nd, len(tsp))
                    plan[id_drone_so_far] = list(np.roll(tsp, shift=shift_am))
                    # print(nd, n_drones_in_cluster, tsp, plan[id_drone_so_far])
                    id_drone_so_far += 1
                # print()

            print(plan)
        else:
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

        print(plan)
        return plan
