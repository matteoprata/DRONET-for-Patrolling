
import numpy as np
from src.utilities.utilities import euclidean_distance
from src.patrolling.meta_patrolling import PatrollingPolicy


class MaxSumResidualPolicy(PatrollingPolicy):
    name = "Max-Residual-Ratio"
    identifier = 4
    line_tick = 4
    marker = 4

    def __init__(self, patrol_drone, set_drones, set_targets):
        super().__init__(patrol_drone=patrol_drone, set_drones=set_drones, set_targets=set_targets)

    def next_visit(self):
        """ Returns the target leading to the maximum minimum residual upon having reached it. """

        max_min_res_list = [np.inf] * len(self.set_targets)
        for ti, target_1 in enumerate(self.set_targets):
            if self.patrol_drone.current_target() == target_1 or target_1.lock is not None:
                continue

            rel_time_arrival = euclidean_distance(target_1.coords, self.patrol_drone.current_target().coords) / self.patrol_drone.speed
            sec_arrival = self.patrol_drone.sim.cur_step * self.patrol_drone.sim.ts_duration_sec + rel_time_arrival

            min_res_list = []
            for target_2 in self.set_targets:
                ls_visit = target_2.last_visit_ts * self.patrol_drone.sim.ts_duration_sec if target_1.identifier != target_2.identifier else sec_arrival
                RES = (sec_arrival - ls_visit) / target_2.maximum_tolerated_idleness
                min_res_list.append(RES)

            # print("min ->", min_res_list, target_1.identifier)
            max_min_res_list[ti] = np.sum(min_res_list)
            max_min_res_list[0] = np.inf  # ignore the base station
        max_min_res_tar = self.set_targets[np.argmin(max_min_res_list)]

        # print("max_min ->", max_min_res_list, max_min_res_tar.identifier)
        return max_min_res_tar
