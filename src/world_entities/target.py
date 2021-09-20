
from src.utilities.utilities import euclidean_distance
from src.world_entities.entity import SimulatedEntity
import numpy as np


class Target(SimulatedEntity):

    def __init__(self, identifier, coords, maximum_tolerated_idleness, simulator, is_depot=False, n_drones=1):
        SimulatedEntity.__init__(self, identifier, coords, simulator)
        self.maximum_tolerated_idleness = maximum_tolerated_idleness
        self.last_visit_ts = [0] * n_drones if is_depot else [0]  # each drone ha its own view SECONDS SINCE START

        self.furthest_target = None
        self.closest_target = None

        self.lock = None  # drone id
        self.active = True  # if False means that his target should not be considered
        self.is_depot = is_depot

    # ------ AGE OF INFORMATION -- RESIDUAL OF INFORMATION

    def age_of_information(self, next=0, drone_id=None):
        """ seconds since the last visit  """
        drone_id = 0 if not self.is_depot else drone_id
        return (self.simulator.cur_step - self.last_visit_ts[drone_id])*self.simulator.ts_duration_sec + next * self.simulator.ts_duration_sec

    def residual_of_information(self, next=0, drone_id=None):
        return 1 - self.age_of_information(next, drone_id) / self.maximum_tolerated_idleness

    def aoi_idleness_ratio(self, next=0, drone_id=None):
        return self.age_of_information(next, drone_id) / self.maximum_tolerated_idleness

    # ------ AGE OF INFORMATION -- RESIDUAL OF INFORMATION

    def set_last_visit_ts(self, step, drone_id=0):
        if self.is_depot:
            self.last_visit_ts[drone_id] = step
        else:
            self.last_visit_ts[0] = step

    def get_last_visit_ts(self, drone_id):
        if self.is_depot:
            return self.last_visit_ts[drone_id]
        return self.last_visit_ts[0]

    @staticmethod
    def max_aoi(set_targets, cur_tar, drone_id):
        """ Returns the the target with the oldest age. """
        max_aoi = np.argmax([target.age_of_information(drone_id=drone_id) for target in set_targets])
        return set_targets[max_aoi]

    @staticmethod
    def min_residual(set_targets, cur_tar, drone_id):
        """ Returns the target with the lowest percentage residual. """
        # TODO FIX
        min_res = np.argmin([target.residual_of_information(drone_id=drone_id) for target in set_targets])
        return set_targets[min_res]

    @staticmethod
    def min_sum_residual(set_targets, cur_tar, speed, cur_step, ts_duration_sec, drone_id):
        """ Returns the target leading to the maximum minimum residual upon having reached it. """
        set_targets = list(target for target in set_targets if target.active)

        max_min_res_list = [np.inf] * len(set_targets)
        for ti, target_1 in enumerate(set_targets):
            if cur_tar == target_1 or target_1.lock is not None or not target_1.active:
                continue

            rel_time_arrival = euclidean_distance(target_1.coords, cur_tar.coords) / speed
            sec_arrival = cur_step * ts_duration_sec + rel_time_arrival

            min_res_list = []
            for target_2 in set_targets:
                # TODO check if drone_id is really ok
                ls_visit = target_2.get_last_visit_ts(drone_id) * ts_duration_sec if target_1.identifier != target_2.identifier else sec_arrival
                RES = (sec_arrival - ls_visit) / target_2.maximum_tolerated_idleness
                min_res_list.append(RES)

            # print("min ->", min_res_list, target_1.identifier)
            max_min_res_list[ti] = np.sum(min_res_list)
        max_min_res_tar = set_targets[np.argmin(max_min_res_list)]

        # print("max_min ->", max_min_res_list, max_min_res_tar.identifier)
        return max_min_res_tar