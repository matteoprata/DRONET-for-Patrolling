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
    name = "BilClust"
    identifier = 2
    line_tick = 7
    marker = 7

    def __init__(self, set_drones, set_targets):
        super().__init__(set_drones=set_drones, set_targets=set_targets)

    def set_tour(self) -> dict:
        return self.my_solution()  # {0: [0, 1, 2, 4, 2, 3, 0, 3], 1: [3]}

    def my_solution(self) -> dict:
        self.set_targets = self.set_targets[1:]

        n_drones = len(self.set_drones)
        n_targets = len(self.set_targets)

        distances = self.distances_matrix_targets(self.set_targets, n_targets)
        n_clusters = self.elbow_k_search(n_drones, distances)
        target_clusters = self.kmeans(n_clusters, distances)  # [00000000 1111] len of the target NO BS

        print("n_clusters", n_clusters)
        # number of drones in each cluster
        if n_drones > n_clusters:
            clusters_assignments = self.n_drones_cluster(n_clusters, n_drones, self.set_targets, target_clusters)
        else:
            clusters_assignments = [1 for _ in range(n_clusters)]

        plan = {}
        drones_so_far = 0
        # second clustering
        for c in range(n_clusters):
            n_drones_cluster = int(clusters_assignments[c])
            # targets objects in the cluster
            new_targets_subset = [self.set_targets[tid] for tid in np.where(target_clusters == c)[0]]
            # targets_coo = np.asarray([np.array(t.coords) for t in targets_cluster_i[:]])
            print(c+1, n_clusters, new_targets_subset)
            distances_intern = self.distances_matrix_targets(new_targets_subset, len(new_targets_subset))

            assert n_drones_cluster <= len(new_targets_subset), str(n_drones_cluster)  + "_" + str(len(new_targets_subset))
            target_clusters_int = self.kmeans(n_drones_cluster, distances_intern)  # cluster nel cluster [0 0 0 1 1 1]

            print("siamo qui", target_clusters_int)
            for dr in range(n_drones_cluster):
                tar_ids_for_drone = [new_targets_subset[tid] for tid in np.where(target_clusters_int == dr)[0]]
                tar_coo = np.asarray([np.array(t.coords) for t in tar_ids_for_drone])
                if len(tar_coo) > 1:
                    tsp = Christofides().compute_from_coordinates(tar_coo, 0)
                else:
                    tsp = [0]
                plan[drones_so_far] = [tar_ids_for_drone[i].identifier-1 for i in tsp]
                drones_so_far += 1
        print(plan)
        # plan = self.plan_given_clusters(n_clusters, target_clusters, clusters_assignments, self.set_targets)
        return plan

    def n_drones_cluster(self, n_clusters, n_drones, set_targets, target_clusters):
        UB = 10000
        clusters_tolerances = []
        clusters_crowd = []
        for i in range(n_clusters):
            targets_cluster_i = [set_targets[tid] for tid in np.where(target_clusters == i)[0]]
            cluster_tolerance = np.min([t.maximum_tolerated_idleness for t in targets_cluster_i])  # low is more urgent
            clusters_tolerances.append(cluster_tolerance)
            clusters_crowd.append(len(targets_cluster_i))

        clusters_crowd = np.array(clusters_crowd)
        print(clusters_tolerances)
        clusters_tolerances = 1 / np.array(clusters_tolerances)  # + DRONI = - TOLL highest means [.5, .3, .2] high priority (low thresholds)
        clusters_tolerances /= np.sum(clusters_tolerances)  # [.5, .2, .3]
        print(clusters_tolerances)
        clusters_assignments = clusters_tolerances * (n_drones - n_clusters)
        clusters_assignments = np.floor(clusters_assignments) + 1

        viol = clusters_assignments >= clusters_crowd
        # print(clusters_assignments, clusters_crowd, viol)
        # print("crucial")
        # exit()
        clusters_assignments[viol] = np.minimum(clusters_crowd, clusters_assignments)[viol]
        # clusters_tolerances[viol] = (np.ones(shape=clusters_tolerances.shape) * - np.inf)[viol]

        # print(clusters_assignments, clusters_crowd)

        # spare = int(n_drones - np.sum(clusters_assignments))
        sorted_clusters_tolerances_ix = np.argsort(clusters_tolerances)[::-1]
        spare = int(n_drones - np.sum(clusters_assignments))
        glob_last_end = 0
        for _ in range(spare):
            for idx in range(glob_last_end, len(sorted_clusters_tolerances_ix)):
                n_el = clusters_assignments[sorted_clusters_tolerances_ix[idx]]
                glob_last_end += 1
                glob_last_end %= len(sorted_clusters_tolerances_ix)

                if n_el + 1 <= clusters_crowd[sorted_clusters_tolerances_ix[idx]]:
                    clusters_assignments[sorted_clusters_tolerances_ix[idx]] += 1
                    break

        # clusters_assignments how many drone per cluster [ 1. 10.  2.  1.  1.]
        return clusters_assignments

    def plan_given_clusters(self, n_clusters, target_clusters, clusters_assignments, set_targets):
        """ Assign clusters to drones and shifts the path accordingly """
        # drones assignment
        plan = defaultdict(list)

        clid_tars = {}  # map cluster id : target ids
        for i in range(n_clusters):
            targets_cluster_i = [set_targets[tid] for tid in np.where(target_clusters == i)[0]]  # +1 for depot
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
        targets_theta = np.asarray([t.maximum_tolerated_idleness for t in set_targets])  # 40
        targets_coo = np.asarray([np.array(t.coords) for t in set_targets])
        distances = euclidean_distances(targets_coo, targets_coo)  # 40 x 40 symmetrical

        for i in range(n_targets):
            for j in range(n_targets):
                if i != j:
                    distances[i, j] = distances[i, j] - min(targets_theta[i], targets_theta[j])
        return distances

    def elbow_k_search(self, n_drones, data):
        """ Given the max number of drones available and a set of points, it returns the clusters"""

        k_values = range(1, n_drones+1)  # [1, 2, 3, 4, ]

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
        kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto")
        kmeans.fit(to_fit)
        labels = kmeans.labels_
        target_clusters = np.array(labels)  # [0, 0, 0, 1, 1, 1, ...]
        return target_clusters
