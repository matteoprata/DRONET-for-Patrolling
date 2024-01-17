
from src.patrolling.meta_patrolling import PrecomputedPolicy

import numpy as np
from sklearn.cluster import SpectralClustering
from collections import defaultdict
from src.utilities.utilities import Christofides
from src.utilities.utilities import euclidean_distance
from sklearn.cluster import MeanShift
from sklearn.cluster import KMeans


class Ours(PrecomputedPolicy):

    def __init__(self, set_drones, set_targets):
        super().__init__(set_drones=set_drones, set_targets=set_targets)

    def set_tour(self) -> dict:
        return self.my_solution()  # {0: [0, 1, 2, 4, 2, 3, 0, 3], 1: [3]}

    def my_solution(self) -> dict:

        def benefit_of_cluster(target_to_visit_coo, target_to_visit, ndrones):
            tsp_path = Christofides().compute_from_coordinates(target_to_visit_coo, 0)
            # min_period = np.min([t.maximum_tolerated_idleness for t in target_to_visit]) * self.set_drones[0].simulator.cf.SIM_TS_DURATION
            sum_period = np.sum([t.maximum_tolerated_idleness for t in target_to_visit])  # * self.set_drones[0].simulator.cf.SIM_TS_DURATION

            tsp_coo = np.array(target_to_visit_coo)[tsp_path]
            tsp_period = self.tsp_length(tsp_coo) / self.set_drones[0].speed
            # print(tsp_path, min_period, tsp_period)

            benefit = sum_period - len(target_to_visit) * tsp_period / ndrones
            return tsp_path, benefit

        def split_tsp(tsp_path, tsp_coo, tid, nsplit):
            assert(nsplit > 0)
            if nsplit == 1:
                return {0: tsp_path[0]}

            drones_split = {k:None for k in range(nsplit)} # map the drone to the starting node
            # tsp_path = tsp_per_cluster #Christofides().compute_from_coordinates(coords, 0)  # [0,2,4,5,3,2,1]
            tsp_cum_len = np.zeros(shape=len(tsp_path))

            for i in range(len(tsp_path) - 1):
                t1, t2 = tsp_coo[i], tsp_coo[i + 1]
                tsp_cum_len[i] = tsp_cum_len[i-1] + euclidean_distance(t1, t2)

            t1, t2 = tsp_coo[0], tsp_coo[-1]
            tsp_cum_len[-1] = tsp_cum_len[-2] + euclidean_distance(t1, t2)

            share = tsp_cum_len[-1] / nsplit

            print()
            print(tsp_path)
            print(tsp_cum_len)
            print(share, nsplit)
            print()

            drones_split[0] = tsp_path[0]
            npicked = 1
            for ic, c in enumerate(tsp_cum_len):
                if c >= share * npicked:
                    drones_split[npicked] = tsp_path[ic]
                    npicked += 1
                if npicked >= nsplit:
                    break
            return drones_split   # id del tsp path

        def shift_tsp(tsp: list, start: int):  # OK
            stind = tsp.index(start)
            new_tsp = [start] + tsp[stind+1:] + tsp[:stind]
            return new_tsp

        targets_coo = [np.array(t.coords) for t in self.set_targets]  # coords ALL

        clustering = MeanShift().fit(targets_coo)
        OUTER_clusters = np.array(clustering.labels_)   # CLUSTER

        plan = defaultdict(list)

        clusterID_to_ndrones = {c: 1 for c in set(OUTER_clusters)}
        tsp_per_cluster = {c: None for c in set(OUTER_clusters)}

        def drones_less_cluster():
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
            return plan

        def drones_more_clusters():
            avail_drones = len(self.set_drones) - len(set(OUTER_clusters))

            while avail_drones > 0:
                priority_clus, cluster_augment = np.inf, None  # MAX, cluster to feed with one drone
                for clu in set(OUTER_clusters):
                    # targets in cluster
                    targets_OUTER = [self.set_targets[i] for i, val in enumerate(OUTER_clusters == clu) if val]
                    target_coo_OUTER = [t.coords for t in targets_OUTER]
                    # target_ids_OUTER = [t.identifier for t in targets_OUTER]

                    if len(targets_OUTER) >= 2:
                        tsp_path, clu_benefit = benefit_of_cluster(target_coo_OUTER, targets_OUTER, clusterID_to_ndrones[clu])
                        tsp_per_cluster[clu] = tsp_path
                        print(clu, tsp_path, clu_benefit)
                        # print(target_ids_OUTER)
                        # print()

                        if clu_benefit < priority_clus:
                            priority_clus = clu_benefit
                            cluster_augment = clu
                    else:
                        print("check edge case")
                        exit()

                print("to augment:", cluster_augment)
                clusterID_to_ndrones[cluster_augment] += 1
                avail_drones -= 1
                print(clusterID_to_ndrones)

            print(tsp_per_cluster)

            # Now in clusterID_to_ndrones it is clear how many drones must be added to the same
            # cluster
            used_uavs = 0
            for clu in set(OUTER_clusters):
                print("inizio con cluster", clu)
                N_DRONES_CLU = clusterID_to_ndrones[clu]
                # targets in cluster
                targets_OUTER = [self.set_targets[i] for i, val in enumerate(OUTER_clusters == clu) if val]
                target_coo_OUTER = np.array([t.coords for t in targets_OUTER])
                target_ids_OUTER = [t.identifier for t in targets_OUTER]

                # returns the initial point for every drone among those in this cluster
                drones_split = split_tsp(tsp_per_cluster[clu], target_coo_OUTER[tsp_per_cluster[clu]], target_ids_OUTER, N_DRONES_CLU)
                # where each drone should start
                print("i suoi", N_DRONES_CLU, "droni dovrebbero partire così", drones_split, "ok", tsp_per_cluster[clu])

                for i in range(N_DRONES_CLU):
                    new_tsp = shift_tsp(tsp_per_cluster[clu], drones_split[i])  # tsp path to convert
                    plan[used_uavs] = [targets_OUTER[i].identifier for i in new_tsp if targets_OUTER[i].identifier != 0]
                    used_uavs += 1
            print(plan)
            return plan  # {0: [1, 2, 1, 3]}  # plan  # {0: [1, 2]}

        if len(self.set_drones) > len(set(OUTER_clusters)):
            return drones_more_clusters()
        else:
            return drones_less_cluster()

    @staticmethod
    def tsp_length(path):
        """ meters distance of the tsp"""
        tot_dis = 0
        for i in range(len(path)-1):
            t1, t2 = path[i], path[i+1]
            tot_dis += euclidean_distance(t1, t2)

        t1, t2 = path[0], path[-1]
        tot_dis += euclidean_distance(t1, t2)
        return tot_dis
