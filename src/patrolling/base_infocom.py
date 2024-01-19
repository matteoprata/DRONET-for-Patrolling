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
from sklearn.metrics.pairwise import euclidean_distances


class INFOCOM_Patrol(PrecomputedPolicy):

    def __init__(self, set_drones, set_targets):
        super().__init__(set_drones=set_drones, set_targets=set_targets)

    def set_tour(self) -> dict:
        return self.my_solution()  # {0: [0, 1, 2, 4, 2, 3, 0, 3], 1: [3]}

    def my_solution(self) -> dict:
        ids_targets = {e: t for e, t in enumerate(self.set_targets)}
        targets_coo = np.array([np.array(list(t.coords)) for t in self.set_targets][1:])  # [t.maximum_tolerated_idleness]
        n_drones = len(self.set_drones)
        n_targets = len(self.set_targets) - 1

        distances = self.distances_matrix_targets(self.set_targets, n_targets)
        n_clusters = self.elbow_k_search(n_drones, distances)
        target_clusters = self.kmeans(n_clusters, distances)

        print("n_clusters", n_clusters)

        if n_drones > n_clusters:
            clusters_assignments = self.n_drones_cluster(n_clusters, n_drones, ids_targets, target_clusters)
        else:
            clusters_assignments = [1 for _ in range(n_clusters)]

        plan = self.plan_given_clusters(n_clusters, target_clusters, clusters_assignments, ids_targets)
        return plan

    def n_drones_cluster(self, n_clusters, n_drones, ids_targets, target_clusters):
        clusters_tolerances = []

        for i in range(n_clusters):
            targets_cluster_i = [ids_targets[tid] for tid in np.where(target_clusters == i)[0]]
            cluster_tolerance = np.min([t.maximum_tolerated_idleness for t in targets_cluster_i])  # low is more urgent
            clusters_tolerances.append(cluster_tolerance)

        clusters_tolerances = 1 / np.array(clusters_tolerances)  # + DRONI = - TOLL highest means [.5, .3, .2] high priority (low thresholds)
        clusters_tolerances /= np.sum(clusters_tolerances)  # [.5, .2, .3]
        print(clusters_tolerances)
        clusters_assignments = clusters_tolerances * (n_drones - n_clusters)
        clusters_assignments = np.floor(clusters_assignments) + 1

        spare = int(n_drones - np.sum(clusters_assignments))
        if spare > 0:  # there are spare drones
            sorted_clusters_tolerances_ix = np.argsort(list(clusters_tolerances))[::-1]
            for idx in range(spare):
                clusters_assignments[sorted_clusters_tolerances_ix[idx]] += 1

        # clusters_assignments how many drone per cluster [ 1. 10.  2.  1.  1.]
        return clusters_assignments

    def plan_given_clusters(self, n_clusters, target_clusters, clusters_assignments, ids_targets):
        """ Assign clusters to drones and shifts the path accordingly """
        # drones assignment
        plan = defaultdict(list)

        clid_tars = {}  # map cluster id : target ids
        for i in range(n_clusters):
            targets_cluster_i = [ids_targets[tid + 1] for tid in np.where(target_clusters == i)[0]]  # +1 for depot
            cluster_tids = [t.identifier for t in targets_cluster_i]
            clid_tars[i] = cluster_tids

        print(clid_tars)

        id_drone_so_far = 0
        for i in range(n_clusters):
            # print("CLUSTER i", i)
            target_to_visit = [self.set_targets[tid].coords for tid in clid_tars[i]]
            if len(target_to_visit) > 1:
                tsp_path = Christofides().compute_from_coordinates(target_to_visit, 0)
            elif len(target_to_visit) == 1:
                tsp_path = [0]
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
        return plan

    @staticmethod
    def distances_matrix_targets(set_targets, n_targets):
        """ given a set of targets it returns the matrix of residual times"""
        targets_theta = np.asarray([t.maximum_tolerated_idleness for t in set_targets[1:]])  # 40
        targets_coo = np.asarray([np.array(t.coords) for t in set_targets[1:]])
        distances = euclidean_distances(targets_coo, targets_coo)  # 40 x 40 symmetrical

        for i in range(n_targets):
            for j in range(n_targets):
                if i != j:
                    distances[i, j] = distances[i, j] - min(targets_theta[i], targets_theta[j])
        return distances

    def elbow_k_search(self, n_drones, data):
        """ Given the max number of drones available and a set of points, it returns the clusters"""

        k_values = range(1, n_drones)  # [1, 2, 3, 4, ]

        # Plot the elbow curve
        inertia_values = []
        clusterings = []
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=0)
            kmeans.fit(data)  # targets_coo
            labels = kmeans.labels_

            clusterings.append(labels)
            inertia_values.append(kmeans.inertia_)

        kneedle = KneeLocator(list(k_values), inertia_values, curve='convex', direction='decreasing')
        elbow_index = kneedle.knee

        elbow_index = elbow_index if elbow_index else n_drones - 1
        n_clusters = elbow_index + 1
        n_clusters = min(n_clusters, n_drones)
        # n_clusters = _drones

        print("n_clusters", n_clusters)
        return n_clusters

    @staticmethod
    def kmeans(k, to_fit):
        """ given k it computes the kmeans"""
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(to_fit)
        labels = kmeans.labels_
        target_clusters = np.array(labels)  # [0, 0, 0, 1, 1, 1, ...]
        return target_clusters
