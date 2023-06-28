import numpy as np

from src.world_entities.entity import SimulatedEntity
import src.constants as cst


class Target(SimulatedEntity):

    def __init__(self, identifier, coords, maximum_tolerated_idleness, simulator, family=cst.TargetFamily.BLUE):
        SimulatedEntity.__init__(self, identifier, coords, simulator)
        # THETA
        self.maximum_tolerated_idleness = maximum_tolerated_idleness  # how much time until next visit tolerated SECONDS

        self.last_visit_ts = 0  # - maximum_tolerated_idleness / self.sim.ts_duration_sec
        self.last_visit_ts_by_drone = np.zeros(simulator.n_drones)  # NEW: every target knows the last time since every drone visisted it

        self.lock = None    # drone
        self.active = True  # if False means that his target should not be considered

        self.furthest_target = None
        self.closest_target = None
        self.family = family

    # ------ AGE OF INFORMATION -- RESIDUAL OF INFORMATION

    def AOI_absolute(self, next=0, drone_id_view=None):
        """ Returns seconds since the last visit to a target (if specified this becomes relative to a particular drone).
            E.g. Last visit to X was at time last_visit_ts = 5. Now is cur_step = 10 and drone visits target X.
            this returns the time elapsed in seconds, that is (10-5) step * duration of a step in seconds.
        """
        last_visit_ts = self.last_visit_ts                              # time since last visit to the target by any drone
        if drone_id_view is not None:                                   # if specified this becomes relative to a particular drone
            last_visit_ts = self.last_visit_ts_by_drone[drone_id_view]  # delay of drone on target (self)
        return ((self.simulator.cur_step - last_visit_ts) + next) * self.simulator.ts_duration_sec

    def AOI_ratio(self, next=0, drone_id_view=None):
        """ Returns percentage of AOI.
            E.g. Last visit to X was at time last_visit_ts = 5. Now is cur_step = 10 and drone visits target X.
            Maximum_tolerated_idleness is 15. Percentage of AOI is (10-5) / 15 that is 33.33%.
        """
        return self.AOI_absolute(next, drone_id_view) / self.maximum_tolerated_idleness

    def AOI_tolerance_ratio(self, next=0, drone_id_view=None):
        """ Returns percentage of residual tolerance.
            E.g. Last visit to X was at time last_visit_ts = 5. Now is cur_step = 10 and drone visits target X.
            Maximum_tolerated_idleness is 15. Percentage of residual tolerance is 1 - ((10-5) / 15) that is 66.67%.
        """
        return 1 - self.AOI_ratio(next, drone_id_view)

    # ------ AGE OF INFORMATION -- RESIDUAL OF INFORMATION

    def is_base_station(self):
        """ returns true if the target is the base station. """
        return self.identifier == 0

    def reset(self):
        self.last_visit_ts = 0
        self.last_visit_ts_by_drone = np.zeros(self.simulator.n_drones)
        self.lock = None
        self.active = True

    def __repr__(self):
        return "tar:id_{}-tol_f{}-coo_{}".format(self.identifier, int(self.maximum_tolerated_idleness), self.coords)
