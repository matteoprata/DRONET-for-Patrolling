
import numpy as np
from src.patrolling.meta_patrolling import PatrollingPolicy
from sklearn.cluster import KMeans
from collections import defaultdict
from src.utilities.utilities import euclidean_distance

class ClusterMaxAOIRatioPolicy(PatrollingPolicy):
    name = "Go-cluster-Max-AOI-Ratio"
    identifier = 2
    line_tick = "-"
    marker = "<"

    def __init__(self, patrol_drone, set_drones, set_targets):
        super().__init__(patrol_drone=patrol_drone, set_drones=set_drones, set_targets=set_targets)

        # assigns every drone to a cluster
        targets_coo = [np.array(t.coords) for t in self.set_targets]

        n_drones = len(self.set_drones)
        kmeans_vals = KMeans(n_clusters=n_drones, random_state=0, n_init="auto").fit(targets_coo)
        clusters = np.array(kmeans_vals.labels_)

        # drones assignment
        self.clusters = defaultdict(list)
        for id_target, id_drone in enumerate(clusters):
            self.clusters[id_drone].append(id_target)

    def next_visit(self):
        """ Returns the target with the lowest percentage residual. """

        max_min_res_list = [np.inf] * len(self.set_targets)
        for ti in self.clusters[self.patrol_drone.identifier]:
            target_1 = self.set_targets[ti]
            if self.patrol_drone.current_target() == target_1 or target_1.lock is not None:
                continue

            rel_time_arrival = euclidean_distance(target_1.coords,
                                                  self.patrol_drone.current_target().coords) / self.patrol_drone.speed
            sec_arrival = self.patrol_drone.sim.cur_step * self.patrol_drone.sim.ts_duration_sec + rel_time_arrival

            min_res_list = []
            for target_2 in self.set_targets:
                ls_visit = target_2.last_visit_ts * self.patrol_drone.sim.ts_duration_sec if target_1.identifier != target_2.identifier else sec_arrival
                RES = (sec_arrival - ls_visit) / target_2.maximum_tolerated_idleness
                min_res_list.append(RES)

            # print("min ->", min_res_list, target_1.identifier)
            max_min_res_list[ti] = np.sum(min_res_list)
        max_min_res_tar = self.set_targets[np.argmin(max_min_res_list)]

        # print("max_min ->", max_min_res_list, max_min_res_tar.identifier)
        return max_min_res_tar


