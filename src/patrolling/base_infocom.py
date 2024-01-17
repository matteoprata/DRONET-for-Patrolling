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
        targets_coo = np.array([np.array(list(t.coords) + [t.maximum_tolerated_idleness]) for t in self.set_targets][1:])
        n_drones = len(self.set_drones)

        # n_drones = len(self.set_drones)
        # kmeans_vals = KMeans(n_clusters=n_drones, random_state=0, n_init="auto").fit(targets_coo)
        # target_clusters = np.array(kmeans_vals.labels_)  # [0, 0, 0, 1, 1, 1, ...]
        # print(target_clusters)
        #
        # # drones assignment
        # plan = defaultdict(list)
        # # plan[0].append(0)
        # for id_target, id_drone in enumerate(target_clusters):
        #     plan[id_drone].append(id_target+1)
        #
        # # path optimization
        # for d in plan:
        #     target_to_visit = [self.set_targets[tid].coords for tid in plan[d]]
        #     if len(target_to_visit) >= 2:
        #         tsp_path = Christofides().compute_from_coordinates(target_to_visit, 0)
        #         plan[d] = [plan[d][tp] for tp in tsp_path]
        #
        # #return plan  # {0: [1, 2, 1, 3]}  # plan  # {0: [1, 2]}

        # fig = plt.figure()
        # ax = fig.add_subplot(121, projection='3d')
        # for x, y, z in targets_coo:
        #     ax.scatter(x, y, z)

        k_values = range(1, 15)

        # Plot the elbow curve
        # ax = fig.add_subplot(122)
        inertia_values = []
        clusterings = []
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(targets_coo)
            labels = kmeans.labels_

            clusterings.append(labels)
            inertia_values.append(kmeans.inertia_)

        kneedle = KneeLocator(list(k_values), inertia_values, curve='convex', direction='decreasing')
        elbow_index = kneedle.knee
        n_clusters = elbow_index + 1
        n_clusters = min(n_clusters, n_drones)

        print("n_clusters", n_clusters)
        # np.argmin()

        # ax.plot(k_values, inertia_values, marker='o')
        # ax.set_xlabel('Number of Clusters (k)')
        # ax.set_ylabel('Inertia')
        # ax.set_title('Elbow Method for Optimal k')
        #
        # plt.show()

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(targets_coo)
        labels = kmeans.labels_
        target_clusters = np.array(labels)  # [0, 0, 0, 1, 1, 1, ...]

        if n_drones > n_clusters:

            clusters_tolerances = []
            for i in range(n_clusters):
                targets_cluster_i = [ids_targets[tid] for tid in np.where(target_clusters == i)[0]]
                cluster_tolerance = np.min([t.maximum_tolerated_idleness for t in targets_cluster_i])  # low is more urgent
                clusters_tolerances.append(cluster_tolerance)
            clusters_tolerances = 1 / np.array(clusters_tolerances)
            clusters_tolerances /= np.sum(clusters_tolerances)
            print(clusters_tolerances)
            clusters_assignments = clusters_tolerances * (n_drones - n_clusters)
            clusters_assignments = np.floor(clusters_assignments) + 1

            while n_drones - np.sum(clusters_assignments) > 0:
                spare = n_drones - np.sum(clusters_assignments)
                ma = np.max(clusters_tolerances)
                if np.floor(spare * ma) < 1:
                    iv = np.argmax(clusters_tolerances)
                    clusters_assignments[iv] += spare
                    continue

                print("iter", spare, clusters_assignments)
                clusters_assignments_new = clusters_tolerances * spare
                clusters_assignments_new = np.floor(clusters_assignments_new) + clusters_assignments
                clusters_assignments = clusters_assignments_new

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

                target_to_visit = [self.set_targets[tid].coords for tid in clid_tars[i]]
                if len(target_to_visit) >= 2:
                    tsp_path = Christofides().compute_from_coordinates(target_to_visit, 0)
                else:
                    print("Problem! Christofides")
                    exit()

                n_drones_in_cluster = int(clusters_assignments[i])
                for nd in range(n_drones_in_cluster):
                    tsp = np.array([clid_tars[i][tp] for tp in tsp_path])
                    shift_am = min((len(tsp) // n_drones_in_cluster) *nd, len(tsp))
                    plan[id_drone_so_far] = list(np.roll(tsp, shift=shift_am))
                    id_drone_so_far += 1

            # print(plan)
            #
            # # path optimization
            # for d in plan:
            #     target_to_visit = [self.set_targets[tid].coords for tid in plan[d]]
            #     if len(target_to_visit) >= 2:
            #         tsp_path = Christofides().compute_from_coordinates(target_to_visit, 0)
            #         plan[d] = [plan[d][tp] for tp in tsp_path]

            print(plan)
        else:
            # TSP simple
            print(target_clusters)

            # drones assignment
            plan = defaultdict(list)
            # plan[0].append(0)
            for id_target, id_drone in enumerate(target_clusters):
                plan[id_drone].append(id_target + 1)

            print(plan)
            # path optimization
            for d in plan:
                target_to_visit = [self.set_targets[tid].coords for tid in plan[d]]
                if len(target_to_visit) >= 2:
                    tsp_path = Christofides().compute_from_coordinates(target_to_visit, 0)
                    plan[d] = [plan[d][tp] for tp in tsp_path]

        return plan
