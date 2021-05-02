
from src.utilities.utilities import euclidean_distance
from src.world_entities.entity import SimulatedEntity
import numpy as np


class Target(SimulatedEntity):

    def __init__(self, identifier, coords, maximum_tolerated_idleness, simulator):
        SimulatedEntity.__init__(self, identifier, coords, simulator)
        self.maximum_tolerated_idleness = maximum_tolerated_idleness
        self.last_visit_ts = 0  # -maximum_tolerated_idleness / self.simulator.ts_duration_sec

        self.furthest_target = None
        self.closest_target = None

        self.lock = None  # drone id

    # ------ AGE OF INFORMATION -- RESIDUAL OF INFORMATION

    def age_of_information(self, next=0):
        return (self.simulator.cur_step - self.last_visit_ts)*self.simulator.ts_duration_sec + next * self.simulator.ts_duration_sec

    def residual_of_information(self):
        return 1 - self.age_of_information() / self.maximum_tolerated_idleness

    def aoi_idleness_ratio(self):
        return self.age_of_information() / self.maximum_tolerated_idleness

    # ------ AGE OF INFORMATION -- RESIDUAL OF INFORMATION

    @staticmethod
    def max_aoi(set_targets, cur_tar):
        """ Returns the the target with the oldest age. """
        max_aoi = np.argmax([target.age_of_information() for target in set_targets])
        return set_targets[max_aoi]

    @staticmethod
    def min_residual(set_targets, cur_tar):
        """ Returns the target with the lowest percentage residual. """
        min_res = np.argmin([target.residual_of_information() for target in set_targets])
        return set_targets[min_res]

    @staticmethod
    def min_sum_residual(set_targets, cur_tar, speed, cur_step, ts_duration_sec):
        """ Returns the target leading to the maximum minimum residual upon having reached it. """

        max_min_res_list = [np.inf] * len(set_targets)
        for ti, target_1 in enumerate(set_targets):
            if cur_tar == target_1 or target_1.lock is not None:
                continue

            rel_time_arrival = euclidean_distance(target_1.coords, cur_tar.coords) / speed
            sec_arrival = cur_step * ts_duration_sec + rel_time_arrival

            min_res_list = []
            for target_2 in set_targets:
                ls_visit = target_2.last_visit_ts * ts_duration_sec if target_1.identifier != target_2.identifier else sec_arrival
                RES = (sec_arrival - ls_visit) / target_2.maximum_tolerated_idleness
                min_res_list.append(RES)

            # print("min ->", min_res_list, target_1.identifier)
            max_min_res_list[ti] = np.sum(min_res_list)
        max_min_res_tar = set_targets[np.argmin(max_min_res_list)]

        # print("max_min ->", max_min_res_list, max_min_res_tar.identifier)
        return max_min_res_tar