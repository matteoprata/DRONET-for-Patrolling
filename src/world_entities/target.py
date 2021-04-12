
from src.utilities.utilities import euclidean_distance
from src.world_entities.entity import SimulatedEntity
import numpy as np


class Target(SimulatedEntity):

    def __init__(self, identifier, coords, maximum_tolerated_idleness, simulator):
        SimulatedEntity.__init__(self, identifier, coords, simulator)
        self.maximum_tolerated_idleness = maximum_tolerated_idleness
        self.last_visit_ts = 0  #-maximum_tolerated_idleness / self.simulator.ts_duration_sec

        self.furthest_target = None
        self.closest_target = None

    # ------ AGE OF INFORMATION -- RESIDUAL OF INFORMATION

    def age_of_information(self):
        return (self.simulator.cur_step - self.last_visit_ts)*self.simulator.ts_duration_sec

    def residual_of_information(self):
        # self.maximum_tolerated_idleness - self.age_of_information()
        return 1 - self.age_of_information() / self.maximum_tolerated_idleness

    # ------ AGE OF INFORMATION -- RESIDUAL OF INFORMATION

    @staticmethod
    def oldest(set_targets, cur_tar):
        """ Returns the the target with the oldest age. """
        max_aoi = np.argmax([target.age_of_information() for target in set_targets])
        target = set_targets[max_aoi]
        assert(target != cur_tar)
        return target

    @staticmethod
    def lowest_residual(set_targets, cur_tar):
        """ Returns the target with the lowest percentage residual. """
        min_res = np.argmin([target.residual_of_information() for target in set_targets])
        target = set_targets[min_res]
        assert(target != cur_tar)
        return target

    @staticmethod
    def lowest_residual_nearest(set_targets, cur_tar, speed):
        def residual_nearest(target_pivot, target_other):
            travel_time = euclidean_distance(target_pivot.coords, target_other.coords) / speed
            return 1 - (target_other.age_of_information() + travel_time) / target_other.maximum_tolerated_idleness

        min_aoi = np.argmin([residual_nearest(cur_tar, target) for target in set_targets])
        min_aoi_target = set_targets[min_aoi]
        assert(min_aoi_target != cur_tar)
        return min_aoi_target
